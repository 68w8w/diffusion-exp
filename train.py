"""Training loop for MDLM Multi-Head Distillation."""

import os
import math
import json
import torch
import torch.nn.functional as F
import transformers

from config import Config
from model import MultiHeadStudent, Teacher, build_student, build_teacher
from diffusion_utils import forward_noise, absorbing_reverse_step
from data import get_dataloader


def compute_targets(
    student: MultiHeadStudent,
    teacher: Teacher,
    z_t: torch.Tensor,
    t_src: torch.Tensor,
    hidden_detached: torch.Tensor,
    c_detached: torch.Tensor,
    config: Config,
) -> tuple:
    """
    Construct targets via rollout (no_grad).

    Args:
        student: student model (used for detached heads)
        teacher: teacher model
        z_t: [B, L] initial noised sequence
        t_src: [B] starting time for the segment
        hidden_detached: [B, L, d] detached backbone hidden
        c_detached: conditioning vector (detached)
        config: configuration

    Returns:
        targets: list of K tensors [B, L, V] (teacher log-probs, bfloat16 to save memory)
        masks: list of K tensors [B, L] (bool, True where z==MASK)
    """
    K = config.K
    T_outer = config.T_outer
    Delta = 1.0 / (T_outer * K)
    eps = config.numerical_eps
    mask_token_id = config.mask_token_id

    targets = []
    masks = []
    z = z_t.clone()

    with torch.no_grad():
        for h in range(K):
            t_curr = t_src - h * Delta  # [B]
            t_next = t_src - (h + 1) * Delta  # [B]
            t_next = t_next.clamp(min=eps)

            # Teacher log-probs at current z, t_curr
            target_h = teacher.forward_log_probs(z, t_curr)  # [B, L, V] fp32
            # Store in bf16 to save memory (cast back to fp32 during loss)
            targets.append(target_h.bfloat16())

            # Mask: which positions are MASK
            mask_h = (z == mask_token_id)  # [B, L]
            masks.append(mask_h)

            # Advance z using student's detached head h
            logits_h = student.heads.compute_one_head(hidden_detached, c_detached, h)
            log_p_h = F.log_softmax(logits_h.float(), dim=-1)
            z = absorbing_reverse_step(z, log_p_h, t_curr, t_next, mask_token_id, eps)

    return targets, masks


def compute_kl_loss_one_head(
    student: MultiHeadStudent,
    hidden: torch.Tensor,
    c: torch.Tensor,
    head_idx: int,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """
    Compute KL loss for one head.

    Args:
        student: student model
        hidden: [B, L, d] hidden with gradient
        c: conditioning (for output layer)
        head_idx: which head
        target: [B, L, V] teacher log-probs
        mask: [B, L] bool, True where position was MASK
        config: configuration

    Returns:
        loss_h: scalar KL loss for this head
    """
    mask_token_id = config.mask_token_id

    # Student logits for this head
    logits_h = student.heads.compute_one_head(hidden, c, head_idx)
    log_pS_h = F.log_softmax(logits_h.float(), dim=-1)  # fp32

    # Teacher log-probs (stored in bf16, cast back to fp32 for precision)
    log_pT_h = target.float()  # [B, L, V]

    # KL divergence: sum_v p_T(v) * (log p_T(v) - log p_S(v))
    p_T = log_pT_h.exp()
    term = p_T * (log_pT_h - log_pS_h)

    # Handle 0 * -inf = NaN: where p_T == 0, set term to 0
    term = torch.where(p_T > 0, term, torch.zeros_like(term))

    # MASK token column contribution = 0
    term[..., mask_token_id] = 0

    # Sum over vocab
    kl_per_pos = term.sum(dim=-1)  # [B, L]

    # Average over MASK positions only
    mask_float = mask.float()
    loss_h = (kl_per_pos * mask_float).sum() / mask_float.sum().clamp(min=1)

    return loss_h


def compute_head_divergence(
    student: MultiHeadStudent,
    z: torch.Tensor,
    hidden: torch.Tensor,
    c: torch.Tensor,
    config: Config,
) -> dict:
    """
    Compute KL(head_0 || head_h) for h in 1..K-1 on MASK positions.

    Returns dict: {h: kl_value}
    """
    mask_token_id = config.mask_token_id
    is_mask = (z == mask_token_id)  # [B, L]

    if is_mask.sum() == 0:
        return {h: 0.0 for h in range(1, config.K)}

    with torch.no_grad():
        logits_0 = student.heads.compute_one_head(hidden, c, 0)
        log_p0 = F.log_softmax(logits_0.float(), dim=-1)
        p0 = log_p0.exp()

        divergences = {}
        for h_idx in range(1, config.K):
            logits_h = student.heads.compute_one_head(hidden, c, h_idx)
            log_ph = F.log_softmax(logits_h.float(), dim=-1)

            # KL(head_0 || head_h)
            term = p0 * (log_p0 - log_ph)
            term = torch.where(p0 > 0, term, torch.zeros_like(term))
            term[..., mask_token_id] = 0
            kl_per_pos = term.sum(dim=-1)  # [B, L]

            # Mean over MASK positions
            mask_float = is_mask.float()
            kl_val = (kl_per_pos * mask_float).sum() / mask_float.sum().clamp(min=1)
            divergences[h_idx] = kl_val.item()

    return divergences


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config: Config):
    """Main training loop."""
    torch.manual_seed(config.seed)

    device = config.device
    os.makedirs(config.output_dir, exist_ok=True)

    # Build models
    print("Building student model...")
    student = build_student(config, device)
    print(f"Trainable parameters: {student.count_trainable_parameters():,}")

    print("Building teacher model...")
    teacher = build_teacher(config, device)

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    # Data
    print("Loading data...")
    train_loader = get_dataloader(config, tokenizer, split="train")

    # Optimizer
    trainable_params = student.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, config.total_steps)

    # Val batch for head divergence (fixed)
    val_batch = next(iter(train_loader))['input_ids'].to(device)

    # Training loop
    print(f"Starting training for {config.total_steps} steps...")
    step = 0
    data_iter = iter(train_loader)
    log_data = []

    K = config.K
    T_outer = config.T_outer
    Delta = 1.0 / (T_outer * K)

    while step < config.total_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x0 = batch['input_ids'].to(device)  # [B, L]

        # Sample random outer segment start: t_src ~ Uniform({1/T_outer, ..., 1})
        segment_indices = torch.randint(1, T_outer + 1, (x0.shape[0],), device=device)
        t_src = segment_indices.float() / T_outer  # [B]

        # Forward noise
        z_t = forward_noise(x0, t_src, config.mask_token_id)

        # Student backbone forward
        student.train()
        hidden, c = student.forward_backbone(z_t, t_src)

        # Detached versions for rollout
        hidden_detached = hidden.detach()
        c_detached = c.detach()

        # Compute targets (no_grad rollout)
        targets, masks = compute_targets(
            student, teacher, z_t, t_src,
            hidden_detached, c_detached, config)

        # Compute loss per head with sequential backward
        optimizer.zero_grad()
        total_loss = 0.0
        head_losses = []

        for h in range(K):
            loss_h = compute_kl_loss_one_head(
                student, hidden, c, h, targets[h], masks[h], config)
            weighted_h = (1.0 / K) * loss_h
            weighted_h.backward(retain_graph=(h < K - 1))
            total_loss += weighted_h.item()
            head_losses.append(loss_h.item())
            # Free target memory
            targets[h] = None

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        step += 1

        # Logging
        if step % config.log_every == 0:
            current_lr = scheduler.get_last_lr()[0]

            # Head divergence
            with torch.no_grad():
                t_val = torch.rand(val_batch.shape[0], device=device) * 0.8 + 0.1
                z_val = forward_noise(val_batch, t_val, config.mask_token_id)
                hidden_val, c_val = student.forward_backbone(z_val, t_val)
                divergences = compute_head_divergence(student, z_val, hidden_val, c_val, config)

            log_entry = {
                'step': step,
                'total_loss': total_loss,
                'lr': current_lr,
            }
            for h in range(K):
                log_entry[f'loss_head_{h}'] = head_losses[h]
            for h in range(1, K):
                log_entry[f'div_head_0_vs_{h}'] = divergences.get(h, 0.0)

            log_data.append(log_entry)

            # Write to file immediately (防中断丢数据)
            with open(os.path.join(config.output_dir, 'train_log.json'), 'w') as f:
                json.dump(log_data, f, indent=2)

            div_str = " | ".join(
                [f"div_{h}={divergences.get(h, 0):.4f}" for h in range(1, K)])
            head_loss_str = " | ".join(
                [f"h{h}={head_losses[h]:.4f}" for h in range(K)])
            print(f"[Step {step}] loss={total_loss:.4f} | {head_loss_str} | "
                  f"{div_str} | lr={current_lr:.6f}")

        # Sample generation
        if step % config.sample_every == 0:
            from inference import generate_samples
            print(f"\n[Step {step}] Generating samples...")
            student.eval()
            samples = generate_samples(
                student, config, num_samples=config.num_sample,
                device=device)
            text_samples = tokenizer.batch_decode(samples)
            print("--- Generated samples ---")
            for i, text in enumerate(text_samples[:4]):
                print(f"  Sample {i}: {text[:200]}...")
            print("---")

            # Save samples
            sample_path = os.path.join(
                config.output_dir, f"samples_step{step}.txt")
            with open(sample_path, 'w') as f:
                for text in text_samples:
                    f.write(text + '\n\n---\n\n')
            student.train()

        # Save checkpoint
        if step % config.save_every == 0:
            ckpt_path = os.path.join(config.output_dir, f"checkpoint_step{step}.pt")
            torch.save({
                'step': step,
                'student_backbone_loras': student.backbone_loras.state_dict(),
                'student_heads': student.heads.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': vars(config),
            }, ckpt_path)
            print(f"[Step {step}] Checkpoint saved to {ckpt_path}")

    # Save final checkpoint
    final_ckpt_path = os.path.join(config.output_dir, "checkpoint_final.pt")
    torch.save({
        'step': step,
        'student_backbone_loras': student.backbone_loras.state_dict(),
        'student_heads': student.heads.state_dict(),
        'config': vars(config),
    }, final_ckpt_path)

    # Save training log
    log_path = os.path.join(config.output_dir, "train_log.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"Training complete! Final checkpoint: {final_ckpt_path}")
    return student


if __name__ == "__main__":
    config = Config()
    train(config)
