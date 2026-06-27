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

"""Linear measurement operators for diffusion posterior sampling"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MeasurementOperator(nn.Module):
    """Abstract base class: y = H(x)

    'adjoint' returns H^T r, used for gradients w.r.t. x when the loss
    depends on H x
    """

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def adjoint(self, r: Tensor) -> Tensor:
        raise NotImplementedError


class SparseSensorOperator(MeasurementOperator):
    """Keeps only selected sensors (per-channel)

    Mask shape:
        (1, C, X, Y, 1)  broadcast to batch
        (B, C, X, Y, 1)  per-sample mask
    """

    def __init__(self, mask: Tensor):
        super().__init__()
        assert mask.ndim == 5, f"mask must be 5D, got {mask.shape}"
        self.register_buffer("mask", mask)

    def set_mask(self, mask: Tensor):
        """Update the mask buffer, supports changing the batch dimension"""
        assert mask.ndim == 5, f"mask must be 5D, got {mask.shape}"
        self._buffers["mask"] = mask

    def forward(self, x: Tensor) -> Tensor:
        m = self.mask
        if m.shape[0] == 1 and x.shape[0] != 1:
            m = m.expand(x.shape[0], -1, -1, -1, -1)
        return x * m

    def adjoint(self, r: Tensor) -> Tensor:
        m = self.mask
        if m.shape[0] == 1 and r.shape[0] != 1:
            m = m.expand(r.shape[0], -1, -1, -1, -1)
        return r * m