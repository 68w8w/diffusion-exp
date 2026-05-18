"""500-step training run with full spec params.
B=8, L=1024, LR=1e-4, warmup=100, T_outer=4, K=4
Log every 50 steps to file.
"""

import os
import json
import math
import time
import torch
import torch.nn.functional as F
import datasets

from config import Config
from model import build_student, build_teacher
from diffusion_utils import forward_noise
from train import compute_targets, compute_kl_loss_one_head, compute_head_divergence

# ── Config ──
TOTAL_STEPS = 500
LOG_EVERY = 50
WARMUP = 100
LR = 1e-4
OUTPUT = "./outputs"
LOG_FILE = os.path.join(OUTPUT, "run_500_log.jsonl")

device = "cuda"
config = Config()  # K=4, T_outer=4, B=8, L=1024
K = config.K

os.makedirs(OUTPUT, exist_ok=True)

# ── LR schedule: linear warmup → cosine decay ──
def lr_lambda(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    progress = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

# ── Build models ──
print("Building student...")
student = build_student(config, device)
print(f"Trainable params: {student.count_trainable_parameters():,}")

print("Building teacher...")
teacher = build_teacher(config, device)

# ── Optimizer ──
opt = torch.optim.AdamW(
    student.get_trainable_parameters(),
    lr=LR, betas=(config.beta1, config.beta2),
    eps=config.eps, weight_decay=config.weight_decay,
)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# ── Data ──
print("Loading data...")
ds = datasets.load_from_disk(
    '/data1/wulingdan/data/diffusion/mdlm/cache/openwebtext-train_train_bs1024_wrapped.dat')
# Pre-load a chunk into GPU for speed
N_PRELOAD = 20000
data_ids = torch.tensor([ds[i]['input_ids'] for i in range(N_PRELOAD)])  # [N, 1024] on CPU
print(f"Preloaded {N_PRELOAD} sequences (L=1024)")

# ── Fixed val batch for divergence (spec §3.6) ──
torch.manual_seed(config.seed)
val_batch = data_ids[torch.randint(0, N_PRELOAD, (config.batch_size,))].to(device)

# ── Training loop ──
torch.manual_seed(config.seed)
print(f"\nStarting training: {TOTAL_STEPS} steps, B={config.batch_size}, L={config.seq_len}")
print(f"Logging every {LOG_EVERY} steps to {LOG_FILE}\n")

f_log = open(LOG_FILE, 'w')
t0 = time.time()

for step in range(1, TOTAL_STEPS + 1):
    student.train()

    # Sample batch
    idx = torch.randint(0, N_PRELOAD, (config.batch_size,))
    x0 = data_ids[idx].to(device)  # [B, 1024]

    # Sample t_src ∈ {1/T_outer, ..., 1}
    seg = torch.randint(1, config.T_outer + 1, (config.batch_size,), device=device)
    t_src = seg.float() / config.T_outer

    # Forward noise
    z_t = forward_noise(x0, t_src, config.mask_token_id)

    # Student backbone
    hidden, c = student.forward_backbone(z_t, t_src)

    # Targets (no_grad rollout)
    targets, masks = compute_targets(
        student, teacher, z_t, t_src,
        hidden.detach(), c.detach(), config)

    # Loss per head with sequential backward
    opt.zero_grad()
    total_loss = 0.0
    head_losses = []
    for h in range(K):
        loss_h = compute_kl_loss_one_head(
            student, hidden, c, h, targets[h], masks[h], config)
        weighted = (1.0 / K) * loss_h
        weighted.backward(retain_graph=(h < K - 1))
        total_loss += weighted.item()
        head_losses.append(loss_h.item())
        targets[h] = None  # free memory

    torch.nn.utils.clip_grad_norm_(student.get_trainable_parameters(), config.grad_clip)
    opt.step()
    scheduler.step()

    # ── Logging ──
    if step % LOG_EVERY == 0:
        cur_lr = scheduler.get_last_lr()[0]

        # Head divergence on fixed val batch
        with torch.no_grad():
            t_val = torch.rand(config.batch_size, device=device) * 0.8 + 0.1
            z_val = forward_noise(val_batch, t_val, config.mask_token_id)
            h_val, c_val = student.forward_backbone(z_val, t_val)
            divs = compute_head_divergence(student, z_val, h_val, c_val, config)

        elapsed = time.time() - t0
        entry = {
            'step': step,
            'total_loss': round(total_loss, 6),
            'lr': round(cur_lr, 8),
            'elapsed_s': round(elapsed, 1),
        }
        for h in range(K):
            entry[f'loss_h{h}'] = round(head_losses[h], 6)
        for h in range(1, K):
            entry[f'div_0v{h}'] = round(divs.get(h, 0.0), 6)

        # Write to file (flush immediately)
        f_log.write(json.dumps(entry) + '\n')
        f_log.flush()

        # Also print
        hl = " ".join(f"h{h}={head_losses[h]:.4f}" for h in range(K))
        dv = " ".join(f"d{h}={divs.get(h,0):.4f}" for h in range(1, K))
        print(f"[{step:4d}] loss={total_loss:.4f} | {hl} | {dv} | lr={cur_lr:.6f} | {elapsed:.0f}s")

f_log.close()
elapsed_total = time.time() - t0
print(f"\nDone. {TOTAL_STEPS} steps in {elapsed_total:.0f}s ({elapsed_total/TOTAL_STEPS:.1f}s/step)")
print(f"Log saved to {LOG_FILE}")
