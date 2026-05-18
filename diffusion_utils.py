"""Core diffusion utilities: forward noising, absorbing reverse step."""

import torch


def forward_noise(x0: torch.Tensor, t: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Forward noising: replace each token with MASK with probability t.

    Args:
        x0: [B, L] clean token sequence
        t: [B] or [B, 1] noise probability per sample
        mask_token_id: MASK token index

    Returns:
        z_t: [B, L] noised sequence
    """
    if t.ndim == 1:
        t = t[:, None]  # [B, 1]
    move_mask = torch.rand_like(x0.float()) < t  # [B, L]
    z_t = torch.where(move_mask, mask_token_id, x0)
    return z_t


def absorbing_reverse_step(
    z: torch.Tensor,
    log_p: torch.Tensor,
    t_curr: torch.Tensor,
    t_next: torch.Tensor,
    mask_token_id: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Absorbing reverse step: unmask positions from z.

    For each MASK position:
      - With probability s/t: stay MASK
      - With probability (t-s)/t: sample a token from softmax(log_p)
    Non-MASK positions carry over unchanged.

    Args:
        z: [B, L] current sequence (with MASK tokens)
        log_p: [B, L, V] log-probabilities (log_softmax'd, MASK col = -inf)
        t_curr: [B] or scalar, current time
        t_next: [B] or scalar, target time (< t_curr)
        mask_token_id: MASK token index
        eps: numerical floor for division

    Returns:
        z_new: [B, L] updated sequence
    """
    if isinstance(t_curr, (int, float)):
        t_curr = torch.full((z.shape[0],), t_curr, device=z.device)
    if isinstance(t_next, (int, float)):
        t_next = torch.full((z.shape[0],), t_next, device=z.device)

    t_curr = t_curr.clamp(min=eps)
    t_next = t_next.clamp(min=0.0)

    # Probability of staying masked: s/t
    stay_prob = (t_next / t_curr)[:, None]  # [B, 1]
    # Probability of unmasking: (t-s)/t
    unmask_prob = 1.0 - stay_prob  # [B, 1]

    # Positions that are currently MASK
    is_mask = (z == mask_token_id)  # [B, L]

    # Sample tokens using Gumbel-max trick (equivalent to Categorical sampling)
    probs = log_p.exp()  # [B, L, V]; MASK col = 0 because log_p MASK = -inf
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
    sampled_tokens = (probs / (gumbel_noise + 1e-10)).argmax(dim=-1)  # [B, L]

    # Decide which MASK positions to unmask
    unmask_decision = torch.rand(z.shape, device=z.device) < unmask_prob  # [B, L]
    should_unmask = is_mask & unmask_decision

    # Build result: carry over non-MASK, optionally unmask MASK positions
    z_new = z.clone()
    z_new[should_unmask] = sampled_tokens[should_unmask]

    return z_new
