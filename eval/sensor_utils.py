# FreqNO-DPS: Correcting Neural Operator Spectral Bias via Diffusion 
# Posterior Sampling with Sparse Observations.
# Copyright (C) 2026 Niccolò Perrone, Fanny Lehmann, Stefania Fresca,
# Filippo Gatti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Sensor mask construction for sparse-observation experiments."""

from __future__ import annotations

import torch


@torch.no_grad()
def random_sensor_mask(
    targ: torch.Tensor,
    keep_frac: float = 0.05,
) -> torch.Tensor:
    """Create an IID random sparse-sensor mask

    Parameters
    ----------
    targ : Tensor (B, C, X, Y, T)
        Only its shape and device are used
    keep_frac : float
        Fraction of sensors to observe

    Returns
    -------
    Tensor (B, C, X, Y, 1) with 1 at observed sensors, 0 elsewhere
    The same spatial pattern is shared across all channels
    """
    B, C, X, Y, _ = targ.shape
    N = X * Y
    k = max(1, int(round(keep_frac * N)))

    mask = torch.zeros((B, 1, N), device=targ.device)
    for b in range(B):
        idx = torch.randperm(N, device=targ.device)[:k]
        mask[b, 0, idx] = 1.0

    return mask.expand(B, C, N).reshape(B, C, X, Y, 1)