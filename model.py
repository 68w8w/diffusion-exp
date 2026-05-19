"""MDLM Multi-Head Student Model with LoRA."""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from einops import rearrange
import flash_attn.flash_attn_interface
from config import Config


# ---------------------------------------------------------------------------
# We load the HuggingFace MDLM model (kuleshov-group/mdlm-owt) which
# exposes:  MDLM -> backbone (DITBackbone) -> blocks, output_layer, etc.
# The DITBackbone has the same architecture as the local DIT but uses
# config.hidden_dim (not config.model.hidden_size).
# ---------------------------------------------------------------------------


def _import_hf_functions():
    """Import JIT-fused functions from the HF model code (cached by trust_remote_code)."""
    # These are defined inside the HF model's modeling_mdlm.py which
    # gets cached. We re-implement the key ones we need inline.
    pass


# We need modulate_fused and apply_rotary_pos_emb. Since the HF model
# defines these as torch.jit.script functions, we import them from the
# loaded module at runtime. For now we define them locally:

@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    # shift/scale are already [B, 1, dim] from adaLN_modulation(c)[:, None].chunk(...)
    return x * (1 + scale) + shift


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, :cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, :sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


# ======================================================================
# LoRA
# ======================================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation. B is zero-initialized → output is 0 at init."""

    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


# ======================================================================
# Head-index embedding
# ======================================================================

class HeadIndexEmbedding(nn.Module):
    """Sinusoidal head-index embedding + 2-layer MLP (last layer zero-init)."""

    def __init__(self, K: int, d_model: int):
        super().__init__()
        emb = self._sinusoidal(K, d_model)
        self.register_buffer('embeddings', emb)  # [K, d_model]
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _sinusoidal(n: int, d: int) -> torch.Tensor:
        half = d // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        pos = torch.arange(n, dtype=torch.float32)
        args = pos[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if d % 2:
            emb = torch.cat([emb, torch.zeros(n, 1)], dim=-1)
        return emb

    def forward(self, h: int) -> torch.Tensor:
        return self.mlp(self.embeddings[h])  # [d_model]


# ======================================================================
# MultiHead
# ======================================================================

class MultiHead(nn.Module):
    """K heads sharing backbone hidden and W_out (DDitFinalLayer)."""

    def __init__(self, config: Config, output_layer: nn.Module):
        super().__init__()
        self.K = config.K
        self.mask_token_id = config.mask_token_id
        self.neg_infinity = config.neg_infinity

        # Shared frozen output_layer (reference, NOT re-registered)
        self._output_layer = output_layer

        self.head_index_emb = HeadIndexEmbedding(config.K, config.hidden_dim)
        self.head_loras = nn.ModuleList([
            LoRALayer(config.hidden_dim, config.hidden_dim, config.head_lora_rank)
            for _ in range(config.K)
        ])

    def compute_one_head(self, hidden: torch.Tensor, c: torch.Tensor,
                         head_idx: int) -> torch.Tensor:
        """Compute logits for head `head_idx`.

        Args:
            hidden: [B, L, d] block output (before output_layer)
            c: [B, cond_dim] time conditioning
            head_idx: 0..K-1

        Returns:
            logits: [B, L, V] with MASK column = -inf
        """
        h = hidden + self.head_index_emb(head_idx)           # broadcast
        h = h + self.head_loras[head_idx](h)                 # LoRA + residual
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self._output_layer(h, c)                # shared DDitFinalLayer
        logits = logits.float()
        logits[:, :, self.mask_token_id] = self.neg_infinity
        return logits


# ======================================================================
# Student
# ======================================================================

class MultiHeadStudent(nn.Module):
    """Frozen MDLM backbone + LoRA (rank 128) + K heads."""

    def __init__(self, config: Config, hf_model):
        """
        Args:
            config: Config
            hf_model: MDLM (HuggingFace PreTrainedModel)
                      hf_model.backbone is a DITBackbone with
                      .vocab_embed, .sigma_map, .rotary_emb,
                      .blocks (ModuleList of DDiTBlock),
                      .output_layer (DDitFinalLayer)
        """
        super().__init__()
        self.config = config
        self.hf_model = hf_model          # full MDLM HF model
        self.dit = hf_model.backbone      # DITBackbone

        # Freeze all original parameters + force eval (no dropout)
        for p in self.hf_model.parameters():
            p.requires_grad = False
        self.hf_model.eval()

        # Backbone LoRA
        self.backbone_loras = nn.ModuleDict()
        self._inject_backbone_lora(config.backbone_lora_rank)

        # MultiHead (shares dit.output_layer)
        self.heads = MultiHead(config, self.dit.output_layer)

    def train(self, mode=True):
        """Override: keep frozen backbone (hf_model) in eval mode always."""
        super().train(mode)
        self.hf_model.eval()
        return self

    def _inject_backbone_lora(self, rank: int):
        for i, blk in enumerate(self.dit.blocks):
            self.backbone_loras[f'b{i}_qkv'] = LoRALayer(
                blk.attn_qkv.in_features, blk.attn_qkv.out_features, rank)
            self.backbone_loras[f'b{i}_out'] = LoRALayer(
                blk.attn_out.in_features, blk.attn_out.out_features, rank)
            self.backbone_loras[f'b{i}_up'] = LoRALayer(
                blk.mlp[0].in_features, blk.mlp[0].out_features, rank)
            self.backbone_loras[f'b{i}_down'] = LoRALayer(
                blk.mlp[2].in_features, blk.mlp[2].out_features, rank)

    def forward_backbone(self, indices: torch.Tensor, sigma: torch.Tensor):
        """Forward through backbone+LoRA → (hidden, c).

        Args:
            indices: [B, L] token ids
            sigma: [B] time values

        Returns:
            hidden: [B, L, d] — output of transformer blocks, BEFORE output_layer
            c: [B, cond_dim] — conditioning vector (for output_layer & heads)
        """
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)

        dit = self.dit

        # Handle time_conditioning=False (zero out sigma, same as HF model)
        if not dit.config.time_conditioning:
            sigma = torch.zeros_like(sigma)

        x = dit.vocab_embed(indices)
        c = F.silu(dit.sigma_map(sigma))
        rotary_cos_sin = dit.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i, blk in enumerate(dit.blocks):
                B, S = x.shape[0], x.shape[1]
                bds = blk._get_bias_dropout_scale()   # returns inference (no-drop) fn

                (shift_msa, scale_msa, gate_msa,
                 shift_mlp, scale_mlp, gate_mlp) = (
                    blk.adaLN_modulation(c)[:, None].chunk(6, dim=2))

                # --- Attention ---
                x_skip = x
                x_norm = modulate_fused(blk.norm1(x), shift_msa, scale_msa)
                qkv = blk.attn_qkv(x_norm) + self.backbone_loras[f'b{i}_qkv'](x_norm)
                qkv = rearrange(qkv, 'b s (three h d) -> b s three h d',
                                three=3, h=blk.n_heads)
                with torch.cuda.amp.autocast(enabled=False):
                    cos, sin = rotary_cos_sin
                    qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                cu = torch.arange(0, (B + 1) * S, step=S,
                                  dtype=torch.int32, device=qkv.device)
                attn = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                    qkv, cu, S, 0., causal=False)
                attn = rearrange(attn, '(b s) h d -> b s (h d)', b=B)
                proj = blk.attn_out(attn) + self.backbone_loras[f'b{i}_out'](attn)
                x = bds(proj, None, gate_msa, x_skip, blk.dropout)

                # --- MLP ---
                mlp_in = modulate_fused(blk.norm2(x), shift_mlp, scale_mlp)
                up = blk.mlp[0](mlp_in) + self.backbone_loras[f'b{i}_up'](mlp_in)
                mid = blk.mlp[1](up)       # GELU
                down = blk.mlp[2](mid) + self.backbone_loras[f'b{i}_down'](mid)
                x = bds(down, None, gate_mlp, x, blk.dropout)

        return x, c

    # ------------------------------------------------------------------
    def get_trainable_parameters(self) -> list:
        params = list(self.backbone_loras.parameters())
        params += list(self.heads.head_index_emb.parameters())
        params += list(self.heads.head_loras.parameters())
        return params

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.get_trainable_parameters())


# ======================================================================
# Teacher
# ======================================================================

class Teacher(nn.Module):
    """Frozen teacher: independently loaded MDLM checkpoint, eval mode."""

    def __init__(self, config: Config, hf_model):
        super().__init__()
        self.config = config
        self.hf_model = hf_model
        self.mask_token_id = config.mask_token_id
        self.neg_infinity = config.neg_infinity
        for p in self.hf_model.parameters():
            p.requires_grad = False
        self.hf_model.eval()

    @torch.no_grad()
    def forward_log_probs(self, z: torch.Tensor, t) -> torch.Tensor:
        """[B,L,V] log-probs (MASK col = -inf)."""
        if isinstance(t, (int, float)):
            t = torch.full((z.shape[0],), t, device=z.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(z.shape[0])
        if t.ndim > 1:
            t = t.squeeze(-1)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.hf_model(input_ids=z, timesteps=t)
            if isinstance(logits, tuple):
                logits = logits[0]

        logits = logits.float()
        logits[:, :, self.mask_token_id] = self.neg_infinity
        return F.log_softmax(logits, dim=-1)


# ======================================================================
# Factory
# ======================================================================

def _load_hf_model(config: Config, device: str = "cuda"):
    """Load the MDLM HuggingFace model."""
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        config.hf_model_id, trust_remote_code=True)
    return model.to(device)


def build_student(config: Config, device: str = "cuda") -> MultiHeadStudent:
    hf_model = _load_hf_model(config, device)
    return MultiHeadStudent(config, hf_model).to(device)


def build_teacher(config: Config, device: str = "cuda") -> Teacher:
    hf_model = _load_hf_model(config, device)
    teacher = Teacher(config, hf_model).to(device)
    teacher.eval()
    return teacher
