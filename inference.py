"""Inference for MDLM Multi-Head Student."""

import torch
import torch.nn.functional as F

from config import Config
from diffusion_utils import absorbing_reverse_step


@torch.no_grad()
def generate_samples(
    student,
    config: Config,
    num_samples: int = 8,
    T_outer: int = None,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate samples using multi-head inference.

    Total backbone NFE = T_outer.
    Total reverse substeps = T_outer * K.

    Args:
        student: MultiHeadStudent model
        config: configuration
        num_samples: number of samples to generate
        T_outer: outer steps (override config if provided)
        device: device

    Returns:
        z: [num_samples, L] generated token sequences
    """
    student.eval()

    if T_outer is None:
        T_outer = config.T_outer
    K = config.K
    L = config.seq_len
    mask_token_id = config.mask_token_id
    eps = config.numerical_eps

    # Start with all MASK
    z = torch.full((num_samples, L), mask_token_id, dtype=torch.long, device=device)

    # Time schedule: linspace from 1.0 to eps, total T_outer*K+1 points
    times = torch.linspace(1.0, eps, T_outer * K + 1, device=device)

    for outer in range(T_outer):
        # Time for backbone conditioning
        t_outer = times[outer * K]
        t_sigma = torch.full((num_samples,), t_outer.item(), device=device)

        # ONE backbone forward per outer iteration
        hidden, c = student.forward_backbone(z, t_sigma)

        for h in range(K):
            t_curr = times[outer * K + h]
            t_next = times[outer * K + h + 1]

            # Get logits from head h
            logits_h = student.heads.compute_one_head(hidden, c, h)
            log_p_h = F.log_softmax(logits_h.float(), dim=-1)

            # Absorbing reverse step
            t_curr_batch = torch.full((num_samples,), t_curr.item(), device=device)
            t_next_batch = torch.full((num_samples,), t_next.item(), device=device)
            z = absorbing_reverse_step(
                z, log_p_h, t_curr_batch, t_next_batch, mask_token_id, eps)

    return z


@torch.no_grad()
def generate_samples_baseline_mdlm(
    teacher,
    config: Config,
    num_samples: int = 8,
    num_steps: int = 16,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate samples using vanilla MDLM (teacher) with given number of steps.

    Args:
        teacher: Teacher model (frozen MDLM)
        config: configuration
        num_samples: number of samples to generate
        num_steps: total number of reverse steps
        device: device

    Returns:
        z: [num_samples, L] generated token sequences
    """
    teacher.eval()
    L = config.seq_len
    mask_token_id = config.mask_token_id
    eps = config.numerical_eps

    # Start with all MASK
    z = torch.full((num_samples, L), mask_token_id, dtype=torch.long, device=device)

    # Time schedule
    times = torch.linspace(1.0, eps, num_steps + 1, device=device)

    for i in range(num_steps):
        t_curr = times[i]
        t_next = times[i + 1]

        # Teacher forward
        log_p = teacher.forward_log_probs(z, t_curr)

        # Absorbing reverse step
        t_curr_batch = torch.full((num_samples,), t_curr.item(), device=device)
        t_next_batch = torch.full((num_samples,), t_next.item(), device=device)
        z = absorbing_reverse_step(
            z, log_p, t_curr_batch, t_next_batch, mask_token_id, eps)

    return z
