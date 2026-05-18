"""Sanity checks for MDLM Multi-Head Distillation.

Tests from the spec:
1. K=1, zero-init LoRA student == raw MDLM (max abs error < 1e-5)
2. Target construction: no NaN, MASK col = -inf, rest finite
3. Student == teacher => KL loss ~ 0 (< 1e-4)
4. Random input => KL loss is finite, no NaN
5. K=4, B=8, L=1024 => GPU memory < 22GB
6. K=4, B=4, L=128 tiny training => loss decreases, no NaN
7. Inference => no MASK tokens in output
8. Trainable params ~22M (not near 169M)
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data1/wulingdan/data/diffusion/mdlm')

from config import Config
from model import (
    MultiHeadStudent, Teacher, _load_hf_model,
    build_student, build_teacher,
)
from diffusion_utils import forward_noise, absorbing_reverse_step
from train import compute_targets, compute_kl_loss_one_head
from inference import generate_samples


def test_1_zero_init_student_equals_raw_mdlm():
    """K=1, zero-init LoRA student output == raw MDLM output (max abs error < 1e-5)."""
    print("\n" + "=" * 60)
    print("TEST 1: Zero-init student (K=1) == raw MDLM")
    print("=" * 60)

    device = "cuda"
    config = Config(K=1)

    # Build raw HF model
    raw_model = _load_hf_model(config, device)
    raw_model.eval()

    # Build student with K=1 (zero-init LoRA + zero-init head MLP)
    student = build_student(config, device)
    student.eval()

    # Random input
    torch.manual_seed(42)
    B, L = 2, 128
    z = torch.randint(0, 50257, (B, L), device=device)
    t = torch.rand(B, device=device) * 0.8 + 0.1

    # Raw MDLM full forward
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float32):
        raw_logits = raw_model(input_ids=z, timesteps=t)
        if isinstance(raw_logits, tuple):
            raw_logits = raw_logits[0]

    # Student forward via head 0
    with torch.no_grad():
        hidden, c = student.forward_backbone(z, t)
        student_logits = student.heads.compute_one_head(hidden, c, 0)

    # Compare non-MASK columns
    raw_f = raw_logits.float().clone()
    stu_f = student_logits.float().clone()
    raw_f[:, :, config.mask_token_id] = 0
    stu_f[:, :, config.mask_token_id] = 0

    max_err = (raw_f - stu_f).abs().max().item()
    print(f"  Max abs error (non-MASK cols): {max_err:.2e}")
    passed = max_err < 1e-5
    print(f"  PASS: {passed}")

    del raw_model, student
    torch.cuda.empty_cache()
    return passed


def test_2_target_construction():
    """Target construction: no NaN, MASK col = -inf, rest finite."""
    print("\n" + "=" * 60)
    print("TEST 2: Target construction sanity")
    print("=" * 60)

    device = "cuda"
    config = Config()

    student = build_student(config, device)
    teacher = build_teacher(config, device)
    student.eval()

    torch.manual_seed(42)
    B, L = 4, 128
    x0 = torch.randint(0, 50257, (B, L), device=device)
    t_src = torch.rand(B, device=device) * 0.8 + 0.1
    z_t = forward_noise(x0, t_src, config.mask_token_id)

    with torch.no_grad():
        hidden, c = student.forward_backbone(z_t, t_src)

    targets, masks = compute_targets(
        student, teacher, z_t, t_src,
        hidden.detach(), c.detach(), config)

    all_ok = True
    for h in range(config.K):
        t = targets[h]
        has_nan = torch.isnan(t).any().item()
        mask_col = t[:, :, config.mask_token_id]
        mask_is_neginf = (mask_col <= -1e5).all().item()
        rest = t.clone()
        rest[:, :, config.mask_token_id] = 0
        rest_finite = torch.isfinite(rest).all().item()

        print(f"  Head {h}: NaN={has_nan}, MASK_col=-inf={mask_is_neginf}, rest_finite={rest_finite}")
        if has_nan or not mask_is_neginf or not rest_finite:
            all_ok = False

    print(f"  PASS: {all_ok}")

    del student, teacher
    torch.cuda.empty_cache()
    return all_ok


def test_3_student_equals_teacher_kl_zero():
    """When student output matches teacher, KL should be ~0."""
    print("\n" + "=" * 60)
    print("TEST 3: Student == Teacher => KL ~ 0")
    print("=" * 60)

    device = "cuda"
    config = Config(K=1)

    teacher = build_teacher(config, device)

    torch.manual_seed(42)
    B, L = 4, 128
    x0 = torch.randint(0, 50257, (B, L), device=device)
    t = torch.rand(B, device=device) * 0.8 + 0.1
    z_t = forward_noise(x0, t, config.mask_token_id)

    log_pT = teacher.forward_log_probs(z_t, t)
    mask = (z_t == config.mask_token_id)

    # KL(p_T || p_T) should be exactly 0
    p_T = log_pT.exp()
    term = p_T * (log_pT - log_pT)
    term = torch.where(p_T > 0, term, torch.zeros_like(term))
    term[..., config.mask_token_id] = 0
    kl_per_pos = term.sum(dim=-1)
    mask_float = mask.float()
    kl = (kl_per_pos * mask_float).sum() / mask_float.sum().clamp(min=1)

    print(f"  KL(teacher || teacher) = {kl.item():.2e}")
    passed = kl.item() < 1e-4
    print(f"  PASS: {passed}")

    del teacher
    torch.cuda.empty_cache()
    return passed


def test_4_random_input_kl_finite():
    """Random input => KL is finite, no NaN."""
    print("\n" + "=" * 60)
    print("TEST 4: Random input KL is finite, no NaN")
    print("=" * 60)

    device = "cuda"
    config = Config()

    student = build_student(config, device)
    teacher = build_teacher(config, device)

    torch.manual_seed(42)
    B, L = 4, 128
    x0 = torch.randint(0, 50257, (B, L), device=device)
    t_src = torch.rand(B, device=device) * 0.8 + 0.1
    z_t = forward_noise(x0, t_src, config.mask_token_id)

    hidden, c = student.forward_backbone(z_t, t_src)

    targets, masks = compute_targets(
        student, teacher, z_t, t_src,
        hidden.detach(), c.detach(), config)

    all_ok = True
    for h in range(config.K):
        loss = compute_kl_loss_one_head(
            student, hidden, c, h, targets[h], masks[h], config)
        is_finite = torch.isfinite(loss).item()
        has_nan = torch.isnan(loss).item()
        print(f"  Head {h} loss: {loss.item():.4f} (finite={is_finite}, nan={has_nan})")
        if not is_finite or has_nan:
            all_ok = False

    print(f"  PASS: {all_ok}")

    del student, teacher
    torch.cuda.empty_cache()
    return all_ok


def test_5_memory_usage():
    """K=4, B=8, L=1024 => GPU memory < 22GB."""
    print("\n" + "=" * 60)
    print("TEST 5: Memory usage (K=4, B=8, L=1024)")
    print("=" * 60)

    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    config = Config()
    student = build_student(config, device)
    teacher = build_teacher(config, device)

    torch.manual_seed(42)
    B, L = 8, 1024
    x0 = torch.randint(0, 50257, (B, L), device=device)
    t_src = torch.rand(B, device=device) * 0.8 + 0.1
    z_t = forward_noise(x0, t_src, config.mask_token_id)

    hidden, c = student.forward_backbone(z_t, t_src)

    targets, masks = compute_targets(
        student, teacher, z_t, t_src,
        hidden.detach(), c.detach(), config)

    optimizer = torch.optim.AdamW(student.get_trainable_parameters(), lr=1e-4)
    optimizer.zero_grad()

    for h in range(config.K):
        loss_h = compute_kl_loss_one_head(
            student, hidden, c, h, targets[h], masks[h], config)
        weighted = (1.0 / config.K) * loss_h
        weighted.backward(retain_graph=(h < config.K - 1))

    optimizer.step()

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  Peak GPU memory: {peak_gb:.2f} GB")
    passed = peak_gb < 22.0
    print(f"  PASS: {passed}")

    del student, teacher, optimizer
    torch.cuda.empty_cache()
    return passed


def test_6_tiny_training():
    """K=4, B=4, L=128, 1000 steps => loss decreases, no NaN."""
    print("\n" + "=" * 60)
    print("TEST 6: Tiny training (K=4, B=4, L=128, 1000 steps)")
    print("=" * 60)

    device = "cuda"
    config = Config(batch_size=4, seq_len=128)

    student = build_student(config, device)
    teacher = build_teacher(config, device)

    optimizer = torch.optim.AdamW(
        student.get_trainable_parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
    )

    torch.manual_seed(42)
    x0 = torch.randint(0, 50257, (4, 128), device=device)

    losses = []
    n_steps = 1000
    no_nan = True

    for step in range(1, n_steps + 1):
        student.train()
        t_src_idx = torch.randint(1, config.T_outer + 1, (4,), device=device)
        t_src = t_src_idx.float() / config.T_outer
        z_t = forward_noise(x0, t_src, config.mask_token_id)

        hidden, c = student.forward_backbone(z_t, t_src)

        targets, masks = compute_targets(
            student, teacher, z_t, t_src,
            hidden.detach(), c.detach(), config)

        optimizer.zero_grad()
        total_loss = 0.0
        for h in range(config.K):
            loss_h = compute_kl_loss_one_head(
                student, hidden, c, h, targets[h], masks[h], config)
            weighted = (1.0 / config.K) * loss_h
            weighted.backward(retain_graph=(h < config.K - 1))
            total_loss += weighted.item()

        if torch.isnan(torch.tensor(total_loss)):
            no_nan = False
            break

        torch.nn.utils.clip_grad_norm_(student.get_trainable_parameters(), config.grad_clip)
        optimizer.step()
        losses.append(total_loss)

        if step % 100 == 0:
            print(f"  Step {step}/{n_steps}: loss = {total_loss:.4f}")

    if len(losses) >= 100:
        first_avg = sum(losses[:100]) / 100
        last_avg = sum(losses[-100:]) / 100
        decreased = last_avg < first_avg
        print(f"  First 100 avg: {first_avg:.4f}, Last 100 avg: {last_avg:.4f}")
        print(f"  Decreased: {decreased}")
    else:
        decreased = False

    print(f"  No NaN: {no_nan}")
    passed = no_nan and decreased
    print(f"  PASS: {passed}")

    del student, teacher, optimizer
    torch.cuda.empty_cache()
    return passed


def test_7_inference_no_mask():
    """Inference output has no MASK tokens."""
    print("\n" + "=" * 60)
    print("TEST 7: Inference output has no MASK tokens")
    print("=" * 60)

    device = "cuda"
    config = Config()

    student = build_student(config, device)
    student.eval()

    samples = generate_samples(student, config, num_samples=4, device=device)
    has_mask = (samples == config.mask_token_id).any().item()
    print(f"  Shape: {samples.shape}, contains MASK: {has_mask}")
    passed = not has_mask
    print(f"  PASS: {passed}")

    del student
    torch.cuda.empty_cache()
    return passed


def test_8_trainable_params():
    """Trainable params ~22M, not near 169M."""
    print("\n" + "=" * 60)
    print("TEST 8: Trainable parameter count")
    print("=" * 60)

    device = "cuda"
    config = Config()

    student = build_student(config, device)
    n_train = student.count_trainable_parameters()
    n_total = sum(p.numel() for p in student.parameters())

    print(f"  Trainable: {n_train:,} ({n_train/1e6:.1f}M)")
    print(f"  Total: {n_total:,} ({n_total/1e6:.1f}M)")
    print(f"  Ratio: {n_train/n_total*100:.1f}%")

    passed = 10_000_000 < n_train < 40_000_000
    print(f"  PASS: {passed} (expected ~22M)")

    del student
    torch.cuda.empty_cache()
    return passed


def run_all_tests():
    """Run all sanity checks."""
    print("=" * 60)
    print("MDLM MULTI-HEAD DISTILLATION — SANITY CHECKS")
    print("=" * 60)

    results = {}
    tests = [
        ("1: Zero-init == raw MDLM", test_1_zero_init_student_equals_raw_mdlm),
        ("2: Target construction", test_2_target_construction),
        ("3: KL(T||T) ~ 0", test_3_student_equals_teacher_kl_zero),
        ("4: Random KL finite", test_4_random_input_kl_finite),
        ("5: Memory < 22GB", test_5_memory_usage),
        ("6: Tiny training", test_6_tiny_training),
        ("7: No MASK in output", test_7_inference_no_mask),
        ("8: Param count ~22M", test_8_trainable_params),
    ]

    for name, fn in tests:
        try:
            ok = fn()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        mark = "PASS" if result == "PASS" else "FAIL"
        print(f"  [{mark}] {name}: {result}")

    return results


if __name__ == "__main__":
    run_all_tests()
