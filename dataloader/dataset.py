# Copyright 2024 The CAM Lab at ETH Zurich.
# Modifications made by Niccolò Perrone, Politecnico di Milano /
# CentraleSupélec LMPS, 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


VEL_STATS_PATH = "./checkpoints/vel_zscore_stats_train.npz"


class Normalization:
    """Normalization wrapper following the MIFNO convention

    X : array (n, p) is a matrix of samples
    x : array (p,)   is one sample to normalize
    y : array (p,)   is one normalized sample to denormalize
    """

    def __init__(self, X, x_mean=None, x_std=None, norm_type=None, eps=1e-6):
        if x_mean is None:
            if norm_type == "log-normal":
                self.x_mean = np.mean(np.log(X))
            elif norm_type == "point-normal":
                self.x_mean = np.mean(X, axis=0)
            else:
                self.x_mean = X.mean()
        else:
            self.x_mean = x_mean

        if x_std is None:
            if norm_type == "log-normal":
                self.x_std = 4 * np.std(np.log(X))
            elif norm_type == "point-normal":
                self.x_std = 4 * np.std(X, axis=0)
            else:
                self.x_std = 4 * X.std()
        else:
            self.x_std = x_std

        self.norm_type = norm_type
        self.eps = eps

    def forward(self, x):
        if self.norm_type in ("normal", "point-normal"):
            return (x - self.x_mean) / (self.x_std + self.eps)
        if self.norm_type == "log":
            return np.log(x)
        if self.norm_type == "log-normal":
            return (np.log(x) - self.x_mean) / self.x_std
        return x

    def inverse(self, y):
        if self.norm_type in ("normal", "point-normal"):
            return self.x_mean + self.x_std * y
        if self.norm_type == "log":
            return np.exp(y)
        if self.norm_type == "log-normal":
            return np.exp(self.x_mean + self.x_std * y)
        return y


def norm_constant_distance_Vs(
    s: np.ndarray,
    a: np.ndarray,
    S_in: int = 32,
    S_in_z: int = 32,
    Dx: float = 9600.0,
    Dy: float = 9600.0,
    Dz: float = 9600.0,
) -> float:
    """MIFNO normalization constant based on Vs at the source and source depth

    Inputs:
      s : array (3,)                       source coordinates in meters
      a : array (S_in, S_in, S_in_z)       Vs grid
    """
    s = s.flatten()
    if a.ndim == 4 and a.shape[3] == 1:
        a = a[:, :, :, 0]

    hx = Dx / S_in
    hy = Dy / S_in
    hz = Dz / S_in_z

    ix = int(s[0] // hx)
    iy = int(s[1] // hy)
    iz = int(S_in_z - 1 + s[2] // hz)

    Vs_source = a[ix, iy, iz]
    R = np.sqrt((1e-3 * s[2]) ** 2 + (1e-3 * Dx / 4) ** 2)
    return float(1e-8 * Vs_source ** 2 * R)


class MIFNOGenCFDDataset(Dataset):
    def __init__(
        self,
        dir_data: Sequence[str],
        T_out: int = 320,
        S_in: int = 32,
        S_in_z: int = 32,
        S_out: int = 32,
        transform_a: str = "normal",
        transform_traces: Optional[str] = "distance_Vs",
        N: Optional[int] = None,
        orientation: str = "moment",
        transform_position: Sequence[float] = (9600.0, 9600.0, -9600.0),
        temporal_downsample: int = 1,
        vel_stats_path: str = VEL_STATS_PATH,
    ):
        super().__init__()
        if temporal_downsample < 1 or (T_out % temporal_downsample) != 0:
            raise ValueError(
                f"temporal_downsample={temporal_downsample} must divide T_out={T_out}"
            )

        self.temporal_downsample = int(temporal_downsample)
        self.T_out = int(T_out)
        self.T_eff = self.T_out // self.temporal_downsample

        self.S_in = S_in
        self.S_in_z = S_in_z
        self.S_out = S_out
        self.transform_a = transform_a
        self.transform_traces = transform_traces

        stats = np.load(vel_stats_path)
        self.vel_mean = stats["mean"].astype(np.float32)
        self.vel_std  = stats["std"].astype(np.float32)
        self.eps = 1e-8

        self.orientation = orientation
        self.transform_position = (
            np.array(transform_position, dtype=np.float32)
            if transform_position is not None
            else None
        )

        self.input_channel  = 3
        self.output_channel = 3
        self.spatial_resolution = (self.S_out, self.S_out, self.T_eff)
        self.input_shape  = (3, self.S_out, self.S_out, self.T_eff)
        self.output_shape = (3, self.S_out, self.S_out, self.T_eff)

        # Geology normalization from a_mean.npy and a_std.npy (MIFNO convention)
        a_mean = np.load(os.path.join(dir_data[0], "a_mean.npy"))
        a_std  = np.load(os.path.join(dir_data[0], "a_std.npy"))
        if self.transform_a == "scalar_normal":
            self.a_mean = float(np.mean(a_mean))
            self.a_std  = float(np.mean(a_std))
        else:
            self.a_mean = a_mean[: self.S_in, : self.S_in, : self.S_in_z]
            self.a_std  = a_std[: self.S_in, : self.S_in, : self.S_in_z]

        if self.transform_a in ("normal", "scalar_normal"):
            self.ANorm = Normalization(
                1, norm_type="normal", x_mean=self.a_mean, x_std=4 * self.a_std,
            )
        else:
            self.ANorm = None

        # Collect sample files across all dirs, globally sorted by index
        self.all_files: List[str] = []
        for indiv_dir in dir_data:
            if not os.path.isdir(indiv_dir):
                raise ValueError(f"Data directory does not exist: {indiv_dir}")
            entries = [e for e in os.listdir(indiv_dir) if e.startswith("sample")]
            if not entries:
                raise RuntimeError(f"Directory {indiv_dir} has no 'sample*.h5' files")
            entries = sorted(entries, key=lambda s: int(re.search(r"\d+", s).group()))
            self.all_files += [os.path.join(indiv_dir, e) for e in entries]

        numbers = [re.search(r"(\d+)", os.path.basename(p)).group() for p in self.all_files]
        idx_sorted = np.argsort(np.array(numbers).astype(int))
        self.all_files = list(np.array(self.all_files)[idx_sorted])

        if N is not None:
            self.all_files = self.all_files[:N]

    def __len__(self) -> int:
        return len(self.all_files)

    def _load_h5(self, path: str) -> Dict[str, Any]:
        with h5py.File(path, "r") as f:
            a  = f["a"] [: self.S_in,  : self.S_in,  : self.S_in_z]
            uE = f["uE"][: self.S_out, : self.S_out, : self.T_out]
            uN = f["uN"][: self.S_out, : self.S_out, : self.T_out]
            uZ = f["uZ"][: self.S_out, : self.S_out, : self.T_out]

            s_raw     = (f["s"][:]     if "s"      in f.keys()
                         else np.array([4800.0, 4800.0, -8400.0], dtype=np.float32))
            angle_deg = (f["angle"][:].astype(np.float32) if "angle" in f.keys()
                         else np.array([48.0, 45.0, 88.0], dtype=np.float32))
            moment    = (f["moment"][:].astype(np.float32) if "moment" in f.keys()
                         else np.zeros(6, dtype=np.float32))

        return dict(a=a, uE=uE, uN=uN, uZ=uZ,
                    s_raw=s_raw, angle_deg=angle_deg, moment=moment)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self._load_h5(self.all_files[idx])
        a, uE, uN, uZ = data["a"], data["uE"], data["uN"], data["uZ"]

        if self.temporal_downsample != 1:
            ds = self.temporal_downsample
            uE, uN, uZ = uE[..., ::ds], uN[..., ::ds], uZ[..., ::ds]
            assert uE.shape[-1] == self.T_eff

        s_raw     = data["s_raw"]
        angle_deg = data["angle_deg"]
        moment    = data["moment"]

        # Geology normalization
        if self.ANorm is not None:
            a = self.ANorm.forward(a)
        a = np.expand_dims(a, axis=0)  # channel first

        # MIFNO normalization constant for traces
        if self.transform_traces == "distance_Vs":
            norm_cst = norm_constant_distance_Vs(s_raw, data["a"])
        else:
            norm_cst = 1.0

        # Stack components, scale by norm_cst, then z-score per channel
        # (z-scoring is safe here because the GenCFD loss is on z-scored targets)
        uE = np.expand_dims(uE * norm_cst, axis=0)
        uN = np.expand_dims(uN * norm_cst, axis=0)
        uZ = np.expand_dims(uZ * norm_cst, axis=0)
        vel   = np.concatenate([uE, uN, uZ], axis=0)
        vel_z = (vel - self.vel_mean[:, None, None, None]) / (self.vel_std[:, None, None, None] + self.eps)

        # Source / angles
        s         = s_raw.astype(np.float32)
        angle_rad = np.deg2rad(angle_deg).astype(np.float32)

        # MIFNO source vector: 3 normalized coords + 6 moment components
        s_pos = (s_raw.astype(np.float32) / self.transform_position
                 if self.transform_position is not None
                 else s_raw.astype(np.float32))
        if self.orientation == "moment":
            s_mifno = np.concatenate([s_pos, moment],    axis=0).astype(np.float32)
        elif self.orientation == "angle":
            s_mifno = np.concatenate([s_pos, angle_rad], axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unknown orientation {self.orientation!r}")

        # Tensors
        vel_z_t        = torch.from_numpy(vel_z).to(torch.float32)
        geo_t          = torch.from_numpy(a).to(torch.float32)
        s_t            = torch.from_numpy(s).to(torch.float32)
        angle_t        = torch.from_numpy(angle_rad).to(torch.float32)
        s_mifno_t      = torch.from_numpy(s_mifno).to(torch.float32)
        norm_traces_t  = torch.tensor(norm_cst, dtype=torch.float32).view(1, 1, 1, 1)
        initial_cond   = torch.zeros_like(vel_z_t)  # dummy, kept for interface compatibility

        return {
            "initial_cond": initial_cond,
            "target_cond":  vel_z_t,
            "norm_traces":  norm_traces_t,
            "cond": {
                "a":       geo_t,
                "s":       s_t,
                "angle":   angle_t,
                "s_mifno": s_mifno_t,
            },
        }