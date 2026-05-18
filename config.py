"""Configuration for MDLM Multi-Head Distillation."""

from dataclasses import dataclass


@dataclass
class Config:
    # Model
    K: int = 4                          # Number of heads
    T_outer: int = 4                    # Outer steps (target NFE)
    backbone_lora_rank: int = 128       # LoRA rank for backbone
    head_lora_rank: int = 64            # LoRA rank for per-head
    hidden_dim: int = 768               # DiT hidden dimension
    vocab_size: int = 50258             # GPT-2 vocab + MASK token
    mask_token_id: int = 50257          # MASK token ID
    seq_len: int = 1024                 # Sequence length

    # Training
    lr: float = 1e-4
    warmup_steps: int = 1000
    total_steps: int = 30000
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_size: int = 8
    seed: int = 42

    # Numerical
    numerical_eps: float = 1e-5         # eps for clamping t
    neg_infinity: float = -1e6          # -inf replacement

    # Logging
    log_every: int = 50
    sample_every: int = 5000
    save_every: int = 5000
    num_sample: int = 8                 # Samples to generate during training

    # Paths
    hf_model_id: str = "kuleshov-group/mdlm-owt"
    output_dir: str = "./outputs"

    # Hardware
    device: str = "cuda"
    precision: str = "bf16"             # bf16 mixed precision

    # Eval
    gen_ppl_model: str = "gpt2-large"
    num_eval_samples: int = 1024
