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

"""HDF5 streaming utilities for residual-statistics estimators

Shared between estimate_spectral_model.py and estimate_sigma2_iso.py
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt


def make_lowpass_sos(lowpass_hz: float, dt: float, order: int = 4):
    """Butterworth lowpass SOS coefficients, or None if disabled or invalid"""
    if lowpass_hz <= 0:
        return None
    fs = 1.0 / dt
    Wn = lowpass_hz / (fs / 2.0)
    if Wn >= 1.0:
        print(f"  [lowpass] cutoff {lowpass_hz} Hz >= Nyquist - disabled")
        return None
    print(f"  [lowpass] Butterworth order={order}, cutoff={lowpass_hz} Hz, Wn={Wn:.4f}")
    return butter(order, Wn, btype="low", output="sos")


def load_zscore_stats(stats_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load per-channel z-score stats reshaped to broadcast as (1, C, 1, 1, 1)"""
    s = np.load(stats_path)
    mean = torch.from_numpy(s["mean"].astype(np.float32)).view(1, -1, 1, 1, 1)
    std  = torch.from_numpy(s["std"].astype(np.float32)).view(1, -1, 1, 1, 1)
    return mean, std


def iter_h5_chunks(h5_path, chunk_size, mean, std, sos):
    """Yield z-score-normalized, optionally lowpass-filtered (u, u_no) chunks

    Yields tuples (start, end, N, u, u_no) where:
        start, end : current chunk indices into the full dataset
        N          : total number of samples
        u, u_no    : (chunk, 3, Nx, Ny, Nt) torch float32 on CPU
    """
    with h5py.File(h5_path, "r") as f:
        N = f["uE"].shape[0]
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            u_np = np.stack(
                [f["uE"][start:end], f["uN"][start:end], f["uZ"][start:end]],
                axis=1,
            ).astype(np.float32, copy=False)
            u_no_np = np.stack(
                [f["outE"][start:end], f["outN"][start:end], f["outZ"][start:end]],
                axis=1,
            ).astype(np.float32, copy=False)

            if sos is not None:
                u_np    = sosfiltfilt(sos, u_np,    axis=-1).astype(np.float32, copy=False)
                u_no_np = sosfiltfilt(sos, u_no_np, axis=-1).astype(np.float32, copy=False)

            u    = (torch.from_numpy(u_np)    - mean) / std
            u_no = (torch.from_numpy(u_no_np) - mean) / std
            yield start, end, N, u, u_no