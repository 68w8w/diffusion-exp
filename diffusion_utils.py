"""Core diffusion utilities: forward noising, absorbing reverse step.

Aligned with MDLM original implementation.
"""

import torch


def _sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
    """Sample from categorical distribution using Gumbel-max trick.

    Exactly matches MDLM's _sample_categorical.
    """
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def forward_noise(x0: torch.Tensor, t: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Forward noising: replace each token with MASK with probability t.

    Matches MDLM's q_xt (for linear schedule where move_chance ≈ t).
    """
    if t.ndim == 1:
        t = t[:, None]  # [B, 1]
    move_mask = torch.rand(*x0.shape, device=x0.device) < t  # [B, L]
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
    """Absorbing reverse step, aligned with MDLM's _ddpm_caching_update.

    Constructs a joint categorical distribution over {all tokens, MASK} where:
      - prob(token v) = p_x0(v) * (t - s)     (unmask to v)
      - prob(MASK)    = s                       (stay masked)
    Then samples from this distribution. Non-MASK positions carry over.

    Args:
        z: [B, L] current sequence
        log_p: [B, L, V] log-probabilities (log_softmax'd, MASK col = -inf)
        t_curr: [B] or scalar, current time (= t)
        t_next: [B] or scalar, target time (= s, s < t)
        mask_token_id: MASK token index
        eps: numerical floor
    """
    if isinstance(t_curr, (int, float)):
        t_curr = torch.full((z.shape[0],), t_curr, device=z.device)
    if isinstance(t_next, (int, float)):
        t_next = torch.full((z.shape[0],), t_next, device=z.device)

    t_curr = t_curr.clamp(min=eps)
    t_next = t_next.clamp(min=0.0)  # upstream already clamps to eps; this is a safety floor

    # move_chance_t ≈ t, move_chance_s ≈ s  (linear schedule: alpha_t = 1 - t)
    move_chance_t = t_curr[:, None, None]  # [B, 1, 1]
    move_chance_s = t_next[:, None, None]  # [B, 1, 1]

    # p_x0: [B, L, V] probabilities
    p_x0 = log_p.exp()

    # Joint categorical: unmask probs + stay-as-MASK prob
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, mask_token_id] = move_chance_s[:, :, 0]

    # Sample from joint distribution (Gumbel-max)
    _x = _sample_categorical(q_xs)

    # Non-MASK positions carry over
    copy_flag = (z != mask_token_id).to(z.dtype)
    return (copy_flag * z + (1 - copy_flag) * _x).long()
