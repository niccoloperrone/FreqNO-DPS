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

"""Compute per-channel z-score stats (mean, std) for MIFNO+GenCFD targets

Run once to produce vel_zscore_stats_train.npz, then both training and
inference will pick it up
"""

import argparse
import os

import h5py
import numpy as np

from dataloader.dataset import norm_constant_distance_Vs


def iter_sample_paths(dirs):
    for d in dirs:
        if not os.path.isdir(d):
            raise ValueError(f"Directory does not exist: {d}")
        for name in os.listdir(d):
            if name.startswith("sample") and name.endswith(".h5"):
                yield os.path.join(d, name)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-channel z-score stats for MIFNO+GenCFD targets",
    )
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="Directories with sample_*.h5 (TRAIN dirs only)")
    parser.add_argument("--out",  type=str, required=True,
                        help="Output .npz file (e.g. checkpoints/vel_zscore_stats_train.npz)")
    parser.add_argument("--S_out", type=int, default=32)
    parser.add_argument("--T_out", type=int, default=320)
    args = parser.parse_args()

    S_out, T_out = args.S_out, args.T_out
    sum_c   = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    n_c     = np.zeros(3, dtype=np.int64)
    n_samples = 0

    for path in iter_sample_paths(args.dirs):
        with h5py.File(path, "r") as f:
            a_raw = f["a"][:]
            uE = f["uE"][:S_out, :S_out, :T_out]
            uN = f["uN"][:S_out, :S_out, :T_out]
            uZ = f["uZ"][:S_out, :S_out, :T_out]
            s_raw = (f["s"][:] if "s" in f.keys()
                     else np.array([4800.0, 4800.0, -8400.0], dtype=np.float32))

        norm_cst = norm_constant_distance_Vs(s_raw, a_raw)
        for ch, u_raw in enumerate([uE, uN, uZ]):
            flat = (u_raw * norm_cst).ravel().astype(np.float64)
            sum_c[ch]   += flat.sum()
            sumsq_c[ch] += np.square(flat).sum()
            n_c[ch]     += flat.size

        n_samples += 1
        if n_samples % 100 == 0:
            print(f"Processed {n_samples} samples", flush=True)

    mean = sum_c / n_c
    var  = sumsq_c / n_c - mean ** 2
    std  = np.sqrt(np.maximum(var, 1e-12))

    print("Per-channel stats (after norm_traces):")
    print("  mean:", mean)
    print("  std :", std)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"Saved stats to {args.out}")


if __name__ == "__main__":
    main()