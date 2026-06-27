"""Estimate the isotropic NO residual variance sigma2_NO,iso

The ablation experiment DPS_NO_iso uses a scalar sigma2_NO instead of the
per-mode spectral model. This script computes that scalar:

    sigma2_iso = (1 / (N * |K|)) sum_n sum_k |F(u_NO - u)(k)|^2

i.e. H(k) is fixed to 1 and the residual is averaged over all modes
By Parseval (ortho norm), this equals the spatial-domain MSE per element

Per-channel values are also reported. The DPS_NO_iso method uses the
global scalar; the per-channel values are diagnostic

Usage:
    python -m scripts.estimate_sigma2_iso \
        --h5 ./data/mifno_outputs_val.h5 \
        --stats ./checkpoints/vel_zscore_stats_train.npz \
        --out ./checkpoints/sigma2_iso
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from utils.h5_streaming import make_lowpass_sos, load_zscore_stats, iter_h5_chunks


DEFAULT_H5    = "./data/mifno_outputs_val.h5"
DEFAULT_STATS = "./checkpoints/vel_zscore_stats_train.npz"
DEFAULT_OUT   = "./checkpoints/sigma2_iso"

CHANNEL_NAMES = ["E", "N", "Z"]


def estimate_sigma2_iso(
    h5_path, stats_path, device,
    chunk_size=8, lowpass_hz=5.0, dt=0.02, butter_order=4,
):
    """Single-pass streaming estimation of the isotropic NO residual variance"""
    mean, std = load_zscore_stats(stats_path)
    sos = make_lowpass_sos(lowpass_hz, dt, butter_order)

    with h5py.File(h5_path, "r") as f:
        N_total    = f["uE"].shape[0]
        Nx, Ny, Nt = f["uE"].shape[1:]
    C  = 3
    Nf = Nt // 2 + 1
    n_modes = Nx * Ny * Nf

    print(f"[info] N={N_total}, shape=(N,{C},{Nx},{Ny},{Nt}), Nf={Nf}, modes/ch={n_modes}")
    print("[pass] streaming residual |F(u_NO) - F(u)|^2 with H=1")

    resid_sq_per_ch = torch.zeros(C, dtype=torch.float64)
    resid_sq_full   = torch.zeros(C, Nx, Ny, Nf, dtype=torch.float64)

    for start, end, N, u, u_no in iter_h5_chunks(h5_path, chunk_size, mean, std, sos):
        diff = (u_no - u).to(device)
        F_diff = torch.fft.rfftn(diff, dim=(-3, -2, -1), norm="ortho")
        sq = (torch.abs(F_diff) ** 2).cpu().double()

        resid_sq_per_ch += sq.sum(dim=(0, 2, 3, 4))
        resid_sq_full   += sq.sum(dim=0)
        del diff, F_diff, sq
        print(f"      {end}/{N}", end="\r")
    print()

    sigma2_iso_per_ch = resid_sq_per_ch / (N_total * n_modes)
    sigma2_iso_global = sigma2_iso_per_ch.mean()
    sigma2_per_mode   = (resid_sq_full / N_total).float()

    print("\n  Results:")
    for c, name in enumerate(CHANNEL_NAMES):
        print(f"    ch={name}:  sigma2_iso = {sigma2_iso_per_ch[c]:.6e}")
    print(f"    global:   sigma2_iso = {sigma2_iso_global:.6e}")

    return {
        "sigma2_iso_per_channel": sigma2_iso_per_ch.float(),
        "sigma2_iso_global":      sigma2_iso_global.float(),
        "sigma2_per_mode_H1":     sigma2_per_mode,
    }


def main():
    p = argparse.ArgumentParser(
        description="Estimate the isotropic NO residual variance (H=1 ablation)",
    )
    p.add_argument("--h5",           default=DEFAULT_H5)
    p.add_argument("--stats",        default=DEFAULT_STATS)
    p.add_argument("--out",          default=DEFAULT_OUT)
    p.add_argument("--chunk",        type=int,   default=8)
    p.add_argument("--dt",           type=float, default=0.02)
    p.add_argument("--lowpass_hz",   type=float, default=5.0)
    p.add_argument("--butter_order", type=int,   default=4)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\n")

    result = estimate_sigma2_iso(
        args.h5, args.stats, device,
        chunk_size=args.chunk,
        lowpass_hz=args.lowpass_hz, dt=args.dt, butter_order=args.butter_order,
    )

    pt_path = args.out + ".pt"
    torch.save({
        "sigma2_iso_per_channel": result["sigma2_iso_per_channel"],
        "sigma2_iso_global":      result["sigma2_iso_global"],
        "sigma2_per_mode_H1":     result["sigma2_per_mode_H1"],
    }, pt_path)
    print(f"\nSaved -> {pt_path}")

    npz_path = args.out + ".npz"
    np.savez(
        npz_path,
        sigma2_iso_per_channel=result["sigma2_iso_per_channel"].numpy(),
        sigma2_iso_global=result["sigma2_iso_global"].item(),
        sigma2_per_mode_H1=result["sigma2_per_mode_H1"].numpy(),
    )
    print(f"Saved -> {npz_path}")


if __name__ == "__main__":
    main()