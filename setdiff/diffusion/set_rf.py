"""SetRectifiedFlow — Rectified Flow on [B, N, 4] joint states.

The key insight: x_t is a JOINT state [B, N, 4] = [B, 4N] conceptually.
Diffusion is over the entire set, not per-proposal.
"""

from typing import Tuple

import torch
from torch import Tensor

from ldmdet.diffusion.rectified_flow import RectifiedFlow


class SetRectifiedFlow:
    """Set-level Rectified Flow: operates on [B, N, 4] joint states.

    Wraps :class:`ldmdet.diffusion.rectified_flow.RectifiedFlow`, reshaping
    the set state to a flat joint vector for the underlying RF, then
    reshaping back. The reshape is semantically meaningful (joint state)
    but numerically a no-op because RF operations are element-wise.
    """

    def __init__(self, snr_scale: float = 2.0):
        self.rf = RectifiedFlow(snr_scale=snr_scale)

    def q_sample(
        self,
        x_start: Tensor,
        x_noise: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward diffusion on joint state.

        Args:
            x_start: [B, N, 4] GT boxes (matched to noise slots).
            x_noise: [B, N, 4] noise proposals.
            t: [B] time steps.

        Returns:
            x_t: [B, N, 4] noisy joint state.
            velocity: [B, N, 4] target velocity = x_noise - x_start.
        """
        B, N, D = x_start.shape
        # Reshape to [B, N*D] — conceptually a single 4N-dim joint state.
        x_start_flat = x_start.reshape(B, N * D)
        x_noise_flat = x_noise.reshape(B, N * D)
        x_t_flat, velocity_flat = self.rf.q_sample(
            x_start_flat, x_noise_flat, t
        )
        x_t = x_t_flat.reshape(B, N, D)
        velocity = velocity_flat.reshape(B, N, D)
        return x_t, velocity

    def step(
        self,
        x_t: Tensor,
        x_0_pred: Tensor,
        t_curr: float,
        t_next: float,
    ) -> Tensor:
        """Euler step on joint state.

        Args:
            x_t: [B, N, 4] current noisy state.
            x_0_pred: [B, N, 4] predicted x_0 (GT boxes).
            t_curr: current time (float).
            t_next: next time (float, < t_curr).

        Returns:
            x_next: [B, N, 4] state at t_next.
        """
        B, N, D = x_t.shape
        x_t_flat = x_t.reshape(B, N * D)
        x_0_pred_flat = x_0_pred.reshape(B, N * D)
        x_next_flat = self.rf.step(x_t_flat, x_0_pred_flat, t_curr, t_next)
        return x_next_flat.reshape(B, N, D)

    def heun_step(
        self,
        x_t: Tensor,
        x_0_pred: Tensor,
        t_curr: float,
        t_next: float,
        model_fn,
    ) -> Tensor:
        """Heun (2nd order) step on joint state.

        Args:
            x_t: [B, N, 4] current noisy state.
            x_0_pred: [B, N, 4] predicted x_0 at t_curr.
            t_curr: current time.
            t_next: next time.
            model_fn: callable(x, t) -> (x0_pred, _) for the Heun corrector.

        Returns:
            x_next: [B, N, 4] state at t_next.
        """
        B, N, D = x_t.shape

        def flat_model_fn(x_flat: Tensor, t: float):
            x = x_flat.reshape(B, N, D)
            x0_pred, extra = model_fn(x, t)
            return x0_pred.reshape(B, N * D), extra

        x_t_flat = x_t.reshape(B, N * D)
        x_0_pred_flat = x_0_pred.reshape(B, N * D)
        x_next_flat = self.rf.heun_step(
            x_t_flat, x_0_pred_flat, t_curr, t_next, flat_model_fn
        )
        return x_next_flat.reshape(B, N, D)
