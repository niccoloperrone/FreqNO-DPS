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

"""Velocity z-score (de)normalization and training-stats loaders

Entry points:
- load_vel_stats   : load per-channel velocity z-score statistics
- denorm_vel       : invert z-score on (B,3,X,Y,T) tensors
- unscale_fields   : invert the norm_traces scheme used during training
- load_train_stats_from_val_h5 : pull stats embedded in HDF5 val files
"""

from __future__ import annotations

import numpy as np
import torch
import h5py

EPS = 1e-12

# Default location for the precomputed velocity z-score stats
VEL_STATS_PATH = "./checkpoints/vel_zscore_stats_train.npz"


def load_vel_stats(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    path: str = VEL_STATS_PATH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load per-channel (mean, std) for velocity z-score normalization

    Returns tensors of shape (3,) on *device*
    """
    stats = np.load(path)
    mean = torch.from_numpy(stats["mean"].astype(np.float32)).to(device=device, dtype=dtype)
    std  = torch.from_numpy(stats["std"].astype(np.float32)).to(device=device, dtype=dtype)
    return mean, std


def denorm_vel(
    v_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Invert z-score normalization on velocity fields

    Parameters
    ----------
    v_norm : Tensor (B, 3, X, Y, T)
    mean, std : Tensor (3,) or already broadcastable (1, 3, 1, 1, 1)
    """
    if mean.dim() == 1:
        mean = mean.view(1, 3, 1, 1, 1)
    if std.dim() == 1:
        std = std.view(1, 3, 1, 1, 1)
    return v_norm * std.clamp_min(eps) + mean


def _to_b11111(n: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Reshape norm_traces to (B, 1, 1, 1, 1)"""
    if isinstance(n, np.ndarray):
        n = torch.from_numpy(n)
    if n.dim() == 1:
        return n.view(-1, 1, 1, 1, 1)
    if n.dim() == 4:
        return n.view(n.shape[0], 1, 1, 1, 1)
    if n.dim() == 5:
        return n
    raise ValueError(f"Unexpected norm_traces shape: {tuple(n.shape)}")


def load_train_stats_from_val_h5(val_h5_path: str) -> dict:
    """Load training normalization statistics embedded in an HDF5 file

    Supports both per-voxel (geo_mean/geo_std) and per-channel
    (geo_global_mean/geo_global_std) geology stats schemas
    """
    with h5py.File(val_h5_path, "r") as f:
        if "stats" not in f:
            raise KeyError("Group '/stats' not found in HDF5")
        st = f["stats"]

        u_mean = torch.from_numpy(np.asarray(st["u_mean"][...],   dtype=np.float32))
        u_std  = torch.from_numpy(np.asarray(st["u_std"][...],    dtype=np.float32))
        o_mean = torch.from_numpy(np.asarray(st["out_mean"][...], dtype=np.float32))
        o_std  = torch.from_numpy(np.asarray(st["out_std"][...],  dtype=np.float32))

        if "geo_global_mean" in st and "geo_global_std" in st:
            g_mean = torch.from_numpy(np.asarray(st["geo_global_mean"][...], dtype=np.float32))
            g_std  = torch.from_numpy(np.asarray(st["geo_global_std"][...],  dtype=np.float32))
            geo_scalar = True
        elif "geo_mean" in st and "geo_std" in st:
            g_mean = torch.from_numpy(np.asarray(st["geo_mean"][...], dtype=np.float32))
            g_std  = torch.from_numpy(np.asarray(st["geo_std"][...],  dtype=np.float32))
            geo_scalar = False
        elif "geo_mean_scalar" in st and "geo_std_scalar" in st:
            g_mean = torch.tensor(float(np.asarray(st["geo_mean_scalar"][...], dtype=np.float32)))
            g_std  = torch.tensor(float(np.asarray(st["geo_std_scalar"][...],  dtype=np.float32)))
            geo_scalar = True
        else:
            raise KeyError("No recognizable geology stats found in '/stats'")

    return {
        "u_mean": u_mean, "u_std": u_std,
        "out_mean": o_mean, "out_std": o_std,
        "geo_mean": g_mean, "geo_std": g_std,
        "geo_scalar": geo_scalar,
    }


@torch.no_grad()
def unscale_fields(
    *,
    x_u_norm: torch.Tensor | None,
    x_out_norm: torch.Tensor | None,
    norm_traces: torch.Tensor | np.ndarray,
    stats: dict,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Invert normalised = (raw * norm - mean) / std

    Returns (x_u_phys, x_out_phys); either may be None
    """
    n = _to_b11111(norm_traces).to(device=device, dtype=dtype)

    u_mean = stats["u_mean"].to(device=device, dtype=dtype).view(1, 3, 1, 1, 1)
    u_std  = torch.clamp(stats["u_std"].to(device=device, dtype=dtype), min=EPS).view(1, 3, 1, 1, 1)
    o_mean = stats["out_mean"].to(device=device, dtype=dtype).view(1, 3, 1, 1, 1)
    o_std  = torch.clamp(stats["out_std"].to(device=device, dtype=dtype), min=EPS).view(1, 3, 1, 1, 1)

    x_u_phys, x_out_phys = None, None
    if x_u_norm is not None:
        x_u_phys = (x_u_norm.to(device=device, dtype=dtype) * u_std + u_mean) / torch.clamp(n, min=EPS)
    if x_out_norm is not None:
        x_out_phys = (x_out_norm.to(device=device, dtype=dtype) * o_std + o_mean) / torch.clamp(n, min=EPS)

    return x_u_phys, x_out_phys