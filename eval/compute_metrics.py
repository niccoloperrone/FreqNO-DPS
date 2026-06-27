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

"""Aggregate per-rank partials into a metrics CSV.

Usage:
    python eval/compute_metrics.py --partials_dir <path> [--out <csv>]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.fft import rfft, rfftfreq

from eval.metrics import myL1, myL2, significant_duration_maps_all_sensors

DT = 0.02


def _fourier_band_spectra(values_4d, low_band, mid_band, high_band, dt=DT):
    freqs = rfftfreq(values_4d.shape[-1], d=dt)
    mag = np.abs(rfft(values_4d, axis=-1))
    lo  = mag[..., (freqs >= low_band[0])  & (freqs <= low_band[1])].mean(-1)
    mid = mag[..., (freqs >  mid_band[0])  & (freqs <= mid_band[1])].mean(-1)
    hi  = mag[..., (freqs >  high_band[0]) & (freqs <= high_band[1])].mean(-1)
    return lo, mid, hi


def _bands_over_components(x5d, dt=DT):
    lo, mi, hi = zip(*[
        _fourier_band_spectra(x5d[:, c], (0., 1.), (1., 2.), (2., 5.), dt=dt)
        for c in range(3)
    ])
    return sum(lo) / 3, sum(mi) / 3, sum(hi) / 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partials_dir", type=str, required=True,
                   help="Directory containing part_rank*.pt files")
    p.add_argument("--out", type=str, default=None,
                   help="CSV output path (default: <partials_dir>/../metrics.csv)")
    args = p.parse_args()

    partials_dir = Path(args.partials_dir)
    part_files = sorted(partials_dir.glob("part_rank*.pt"))
    if not part_files:
        raise FileNotFoundError(f"No part_rank*.pt files in {partials_dir}")
    print(f"Loading {len(part_files)} partials from {partials_dir}")

    parts = [torch.load(f, map_location="cpu") for f in part_files]
    targ_all  = torch.cat([p["targ"]  for p in parts])
    mifno_all = torch.cat([p["mifno"] for p in parts])
    gen_all   = torch.cat([p["gen"]   for p in parts])
    norm_all  = torch.cat([p["norm"]  for p in parts])
    N = targ_all.shape[0]
    print(f"Total samples: {N}")

    meta = parts[0]
    method  = meta.get("method", "unknown")
    sampler = meta.get("sampler", "unknown")
    steps   = meta.get("sampling_steps", "?")

    targ_np = targ_all.view(N, 3, -1).numpy()
    gen_np  = gen_all.view(N, 3, -1).numpy()
    init_np = mifno_all.view(N, 3, -1).numpy()

    eps = 1e-2
    base = np.abs(targ_np) + eps
    rel_gen  = (gen_np  - targ_np) / base
    rel_init = (init_np - targ_np) / base

    df = pd.DataFrame(index=np.arange(N), dtype=np.float32)
    df["rMAE_gen"]   = myL1(rel_gen,  axis=2).mean(axis=1)
    df["rRMSE_gen"]  = myL2(rel_gen,  axis=2).mean(axis=1)
    df["rMAE_init"]  = myL1(rel_init, axis=2).mean(axis=1)
    df["rRMSE_init"] = myL2(rel_init, axis=2).mean(axis=1)

    targ_5d = (targ_all  / norm_all).numpy()
    init_5d = (mifno_all / norm_all).numpy()
    gen_5d  = (gen_all   / norm_all).numpy()
    t_lo, t_mid, t_hi = _bands_over_components(targ_5d)
    g_lo, g_mid, g_hi = _bands_over_components(gen_5d)
    i_lo, i_mid, i_hi = _bands_over_components(init_5d)

    eps_fft = 1e-12
    df["rFFTlow_gen"]   = ((g_lo  - t_lo)  / (t_lo  + eps_fft)).mean(axis=(1, 2))
    df["rFFTmid_gen"]   = ((g_mid - t_mid) / (t_mid + eps_fft)).mean(axis=(1, 2))
    df["rFFThigh_gen"]  = ((g_hi  - t_hi)  / (t_hi  + eps_fft)).mean(axis=(1, 2))
    df["rFFTlow_init"]  = ((i_lo  - t_lo)  / (t_lo  + eps_fft)).mean(axis=(1, 2))
    df["rFFTmid_init"]  = ((i_mid - t_mid) / (t_mid + eps_fft)).mean(axis=(1, 2))
    df["rFFThigh_init"] = ((i_hi  - t_hi)  / (t_hi  + eps_fft)).mean(axis=(1, 2))

    D595_t, _, _ = significant_duration_maps_all_sensors(targ_5d, dt=DT)
    D595_g, _, _ = significant_duration_maps_all_sensors(gen_5d,  dt=DT)
    D595_i, _, _ = significant_duration_maps_all_sensors(init_5d, dt=DT)
    df["SD595_true_meanXY"]  = np.nanmean(D595_t.reshape(N, -1), axis=1)
    df["SD595_gen_meanXY"]   = np.nanmean(D595_g.reshape(N, -1), axis=1)
    df["SD595_init_meanXY"]  = np.nanmean(D595_i.reshape(N, -1), axis=1)
    df["SD595_abs_err_gen"]  = np.abs(df["SD595_gen_meanXY"]  - df["SD595_true_meanXY"])
    df["SD595_abs_err_init"] = np.abs(df["SD595_init_meanXY"] - df["SD595_true_meanXY"])

    df_final = pd.concat([
        df,
        df.mean(axis=0).rename("Average").to_frame().T,
        df.std(axis=0).rename("Std").to_frame().T,
    ])

    out_path = (Path(args.out) if args.out is not None
                else partials_dir.parent / f"metrics_{method}_{sampler}.csv")
    df_final.to_csv(out_path, index=True)
    print(f"Metrics saved -> {out_path}")
    print(df_final.tail(3).to_string())


if __name__ == "__main__":
    main()