"""Estimate the per-mode spectral residual model (H, sigma2_NO, P_u)

Computes the spectral transfer function H(k), per-mode NO residual variance
sigma2_NO(k), and signal power spectrum P_u(k) from MIFNO validation outputs

Two-pass streaming over HDF5:
    Pass 1: accumulate cross-spectrum and auto-spectrum -> H, P_u
    Pass 2: compute residual variance given H -> sigma2_NO

Both H_real (real-valued, Re part of the LS estimator) and H_complex (full
complex LS estimator) are saved for downstream diagnostics

Usage:
    python -m scripts.estimate_spectral_model \
        --h5 ./data/mifno_outputs_val.h5 \
        --stats ./checkpoints/vel_zscore_stats_train.npz \
        --out ./checkpoints/spectral_model_final
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
DEFAULT_OUT   = "./checkpoints/spectral_model"

CHANNEL_NAMES = ["E", "N", "Z"]


def estimate_H_and_sigma2(
    h5_path, stats_path, device,
    chunk_size=8, lowpass_hz=5.0, dt=0.02, butter_order=4,
):
    """Two-pass streaming estimation of H, sigma2_NO, and P_u"""
    mean, std = load_zscore_stats(stats_path)
    sos = make_lowpass_sos(lowpass_hz, dt, butter_order)

    with h5py.File(h5_path, "r") as f:
        N_total    = f["uE"].shape[0]
        Nx, Ny, Nt = f["uE"].shape[1:]
    C  = 3
    Nf = Nt // 2 + 1
    shape = (C, Nx, Ny, Nf)
    print(f"[info] N={N_total}, shape=(N,{C},{Nx},{Ny},{Nt}), Nf={Nf}")

    # Pass 1: cross-spectrum and auto-spectrum
    cross = torch.zeros(shape, dtype=torch.complex128)
    auto  = torch.zeros(shape, dtype=torch.float64)
    print("[pass 1] cross & auto spectra")
    for start, end, N, u, u_no in iter_h5_chunks(h5_path, chunk_size, mean, std, sos):
        Fu    = torch.fft.rfftn(u.to(device),    dim=(-3, -2, -1), norm="ortho")
        Fu_no = torch.fft.rfftn(u_no.to(device), dim=(-3, -2, -1), norm="ortho")
        cross += (Fu_no * torch.conj(Fu)).sum(0).cpu().to(torch.complex128)
        auto  += (torch.abs(Fu) ** 2).sum(0).cpu().double()
        del Fu, Fu_no, u, u_no
        print(f"      {end}/{N}", end="\r")
    print()

    P_u = (auto / N_total).to(torch.float32)

    H_complex = torch.zeros(shape, dtype=torch.complex128)
    valid = auto > 1e-12
    H_complex[valid] = cross[valid] / auto[valid].to(torch.complex128)
    H_complex = H_complex.to(torch.complex64)

    H_real = torch.zeros(shape, dtype=torch.float64)
    H_real[valid] = cross[valid].real / auto[valid]
    H_real = H_real.to(torch.float32)

    # Pass 2: residuals given H
    resid_sq_complex = torch.zeros(shape, dtype=torch.float64)
    resid_sq_real    = torch.zeros(shape, dtype=torch.float64)
    H_complex_dev = H_complex.to(device)
    H_real_dev    = H_real.to(device)

    print("[pass 2] residuals for complex & real H")
    for start, end, N, u, u_no in iter_h5_chunks(h5_path, chunk_size, mean, std, sos):
        Fu    = torch.fft.rfftn(u.to(device),    dim=(-3, -2, -1), norm="ortho")
        Fu_no = torch.fft.rfftn(u_no.to(device), dim=(-3, -2, -1), norm="ortho")
        r_complex = Fu_no - H_complex_dev.unsqueeze(0) * Fu
        r_real    = Fu_no - H_real_dev.unsqueeze(0) * Fu
        resid_sq_complex += (torch.abs(r_complex) ** 2).sum(0).cpu().double()
        resid_sq_real    += (torch.abs(r_real)    ** 2).sum(0).cpu().double()
        del Fu, Fu_no, r_complex, r_real, u, u_no
        print(f"      {end}/{N}", end="\r")
    print()

    sigma2_complex = (resid_sq_complex / N_total).to(torch.float32)
    sigma2_real    = (resid_sq_real    / N_total).to(torch.float32)

    for c, name in enumerate(CHANNEL_NAMES):
        Hc = torch.abs(H_complex[c])
        Hr = H_real[c]
        print(f"ch={name}:  |H_cpx| in [{Hc.min():.4f}, {Hc.max():.4f}]  "
              f"H_real in [{Hr.min():.4f}, {Hr.max():.4f}]")
        print(f"        sigma2_cpx in [{sigma2_complex[c].min():.3e}, {sigma2_complex[c].max():.3e}]  "
              f"sigma2_real in [{sigma2_real[c].min():.3e}, {sigma2_real[c].max():.3e}]")

    return {
        "H_complex":      H_complex,
        "H_real":         H_real,
        "sigma2_complex": sigma2_complex,
        "sigma2_real":    sigma2_real,
        "P_u":            P_u,
    }


def save_model(model: dict, out_stem: str):
    """Save spectral model to .npz and .pt"""
    H_cpx = model["H_complex"]
    payload = {
        "H_complex_real": H_cpx.real.cpu().float(),
        "H_complex_imag": H_cpx.imag.cpu().float(),
        "H_real":         model["H_real"].cpu().float(),
        "sigma2_complex": model["sigma2_complex"].cpu().float(),
        "sigma2_real":    model["sigma2_real"].cpu().float(),
        "P_u":            model["P_u"].cpu().float(),
    }

    pt_path = out_stem + ".pt"
    torch.save(payload, pt_path)
    print(f".pt   -> {pt_path}")

    npz_path = out_stem + ".npz"
    np.savez(npz_path, **{k: v.numpy() for k, v in payload.items()})
    print(f".npz  -> {npz_path}")


def main():
    p = argparse.ArgumentParser(
        description="Estimate the per-mode spectral residual model (H, sigma2_NO, P_u)",
    )
    p.add_argument("--h5",           default=DEFAULT_H5)
    p.add_argument("--stats",        default=DEFAULT_STATS)
    p.add_argument("--out",          default=DEFAULT_OUT)
    p.add_argument("--chunk",        type=int,   default=64)
    p.add_argument("--dt",           type=float, default=0.02)
    p.add_argument("--lowpass_hz",   type=float, default=5.0,
                   help="Butterworth lowpass cutoff in Hz; set to 0 to disable")
    p.add_argument("--butter_order", type=int,   default=4)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Output: {args.out}.[pt|npz]\n")

    model = estimate_H_and_sigma2(
        args.h5, args.stats, device,
        chunk_size=args.chunk,
        lowpass_hz=args.lowpass_hz, dt=args.dt, butter_order=args.butter_order,
    )

    save_model(model, args.out)


if __name__ == "__main__":
    main()