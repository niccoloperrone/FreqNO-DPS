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

"""Evaluation metrics for seismic velocity fields"""

from __future__ import annotations

import numpy as np


def myL1(x, axis=None):
    """Mean absolute value"""
    return np.mean(np.abs(x), axis=axis)


def myL2(x, axis=None):
    """Root-mean-square"""
    return np.sqrt(np.mean(x ** 2, axis=axis))


def significant_duration_maps_all_sensors(
    v5d: np.ndarray,
    dt: float = 0.02,
    p1: float = 0.05,
    p2: float = 0.95,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arias-intensity significant-duration maps for all sensors

    Parameters
    ----------
    v5d : ndarray (N, 3, X, Y, T)
        Velocity fields in physical units

    Returns
    -------
    D, t1, t2 : ndarray (N, X, Y) each
        Duration D = t2 - t1 and the bounding times
        Sensors with zero energy are NaN
    """
    _N, _C, _X, _Y, T = v5d.shape
    v2  = np.sum(v5d ** 2, axis=1)
    cum = np.cumsum(v2, axis=-1) * dt
    tot = cum[..., -1]
    valid = np.isfinite(tot) & (tot > 0)

    thr1 = p1 * tot
    thr2 = p2 * tot
    idx1 = (cum >= thr1[..., None]).argmax(axis=-1)
    idx2 = (cum >= thr2[..., None]).argmax(axis=-1)

    def _interp(cum_arr, thr, idx):
        idx0 = np.clip(idx - 1, 0, T - 1)
        c0 = np.take_along_axis(cum_arr, idx0[..., None], axis=-1)[..., 0]
        c1 = np.take_along_axis(cum_arr, idx[..., None],  axis=-1)[..., 0]
        t0 = idx0 * dt
        w = (thr - c0) / (c1 - c0 + eps)
        return t0 + np.clip(w, 0.0, 1.0) * dt

    t1 = _interp(cum, thr1, idx1)
    t2 = _interp(cum, thr2, idx2)
    D  = t2 - t1

    t1 = np.where(valid, t1, np.nan)
    t2 = np.where(valid, t2, np.nan)
    D  = np.where(valid, D,  np.nan)
    return D, t1, t2