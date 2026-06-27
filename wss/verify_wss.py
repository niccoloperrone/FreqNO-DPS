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

"""Verify the wide-sense stationarity (WSS) assumption on the NO residual

WSS implies that distinct Fourier modes of the residual are uncorrelated:

    E[ eta_hat(k) conj(eta_hat(k')) ] = sigma2_NO(k) delta_{kk'}

This script estimates the off-diagonal cross-spectral coherence

    coherence(k, k') = |C(k,k')| / sqrt(sigma2_NO(k) sigma2_NO(k'))

with C(k,k') = (1/N) sum_n eta_hat^(n)(k) conj(eta_hat^(n)(k')) for a
large number of randomly sampled mode pairs (k != k'), bins by ||k - k'||,
and checks whether coherence sits near the finite-sample noise floor
1/sqrt(N)

Optionally stratifies by spectral regime (low-low, high-high, mixed)
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


DEFAULT_H5     = "./data/MIFNO_outputs_val.h5"
DEFAULT_STATS  = "./checkpoints/vel_zscore_stats_train.npz"
DEFAULT_MODEL  = "./checkpoints/spectral_model.pt"
DEFAULT_OUT    = "./results/wss_verification"

CHANNEL_NAMES  = ["E", "N", "Z"]
NUM_CHANNELS   = 3


def _make_sos(lowpass_hz, dt, order):
    if lowpass_hz <= 0:
        return None
    fs = 1.0 / dt
    Wn = lowpass_hz / (fs / 2.0)
    if Wn >= 1.0:
        print(f"  [lowpass] cutoff {lowpass_hz} Hz >= Nyquist, disabled")
        return None
    print(f"  [lowpass] Butterworth order={order}, cutoff={lowpass_hz} Hz, Wn={Wn:.4f}")
    return butter(order, Wn, btype="low", output="sos")


def _load_stats(stats_path):
    print(f"[setup] z-score stats from {stats_path}")
    s = np.load(stats_path)
    mean = torch.from_numpy(s["mean"].astype(np.float32)).view(1, -1, 1, 1, 1)
    std  = torch.from_numpy(s["std"].astype(np.float32)).view(1, -1, 1, 1, 1)
    return mean, std


def _iter_chunks(h5_path, chunk_size, mean, std, sos):
    """Yield z-scored, optionally lowpassed (u, u_NO) chunks on CPU"""
    with h5py.File(h5_path, "r") as f:
        N = f["uE"].shape[0]
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            u_np    = np.stack([f["uE"][start:end],  f["uN"][start:end],  f["uZ"][start:end]],
                               axis=1).astype(np.float32, copy=False)
            u_no_np = np.stack([f["outE"][start:end], f["outN"][start:end], f["outZ"][start:end]],
                               axis=1).astype(np.float32, copy=False)
            if sos is not None:
                u_np    = sosfiltfilt(sos, u_np,    axis=-1).astype(np.float32, copy=False)
                u_no_np = sosfiltfilt(sos, u_no_np, axis=-1).astype(np.float32, copy=False)
            u    = (torch.from_numpy(u_np)    - mean) / std
            u_no = (torch.from_numpy(u_no_np) - mean) / std
            yield start, end, N, u, u_no


def load_spectral_model(model_path):
    print(f"[step 1] Loading spectral model from {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    H_real = ckpt["H_real"]
    sigma2 = ckpt["sigma2_real"]
    print(f"         H_real  shape={H_real.shape}")
    print(f"         sigma2  shape={sigma2.shape}")
    return H_real, sigma2


def sample_mode_pairs(Nx, Ny, Nf, num_pairs, rng):
    """Draw `num_pairs` distinct random pairs from the rFFT grid (Nx, Ny, Nf)

    Returns
    -------
    idx_a, idx_b : (P,) int64    flat indices
    sep          : (P,) float32  ||k_a - k_b|| in normalized frequency units
    kmag_a, kmag_b : (P,) float32   ||k|| for stratification
    """
    print(f"[step 2] Sampling {num_pairs} mode pairs from ({Nx}, {Ny}, {Nf}) grid")
    total_modes = Nx * Ny * Nf

    idx_a = rng.integers(0, total_modes, size=num_pairs)
    idx_b = rng.integers(0, total_modes, size=num_pairs)
    same  = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.integers(0, total_modes, size=same.sum())
        same = idx_a == idx_b

    # Nf = Nt//2 + 1, so Nt = (Nf-1)*2 and rfftfreq(Nt) has shape (Nf,)
    kx = np.fft.fftfreq(Nx).astype(np.float32)
    ky = np.fft.fftfreq(Ny).astype(np.float32)
    kt = np.fft.rfftfreq((Nf - 1) * 2).astype(np.float32)

    ix_a, iy_a, it_a = np.unravel_index(idx_a, (Nx, Ny, Nf))
    ix_b, iy_b, it_b = np.unravel_index(idx_b, (Nx, Ny, Nf))

    sep    = np.sqrt((kx[ix_a] - kx[ix_b])**2
                   + (ky[iy_a] - ky[iy_b])**2
                   + (kt[it_a] - kt[it_b])**2)
    kmag_a = np.sqrt(kx[ix_a]**2 + ky[iy_a]**2 + kt[it_a]**2)
    kmag_b = np.sqrt(kx[ix_b]**2 + ky[iy_b]**2 + kt[it_b]**2)

    print(f"         ||k_a - k_b|| in [{sep.min():.4f}, {sep.max():.4f}]")
    return idx_a, idx_b, sep, kmag_a, kmag_b


def accumulate_cross_spectra(
    h5_path, stats_path, H_real, idx_a, idx_b,
    device, chunk_size=8, lowpass_hz=5.0, dt=0.02, butter_order=4,
):
    """Stream through the dataset, accumulate the cross-spectrum sum
    sum_n eta(k_a) conj(eta(k_b)) for each sampled pair

    Returns
    -------
    cross   : (C, P) complex128
    N_total : int
    """
    mean, std = _load_stats(stats_path)
    sos = _make_sos(lowpass_hz, dt, butter_order)

    C = H_real.shape[0]
    P = len(idx_a)
    cross = np.zeros((C, P), dtype=np.complex128)
    H_dev = H_real.to(device)
    idx_a_np = idx_a.astype(np.int64)
    idx_b_np = idx_b.astype(np.int64)

    print(f"[step 3] Streaming cross-spectrum accumulation")
    N_total = 0
    for start, end, N, u, u_no in _iter_chunks(h5_path, chunk_size, mean, std, sos):
        N_total = N
        B = u.shape[0]

        Fu    = torch.fft.rfftn(u.to(device),    dim=(-3, -2, -1), norm="ortho")
        Fu_no = torch.fft.rfftn(u_no.to(device), dim=(-3, -2, -1), norm="ortho")
        eta   = Fu_no - H_dev.unsqueeze(0) * Fu
        del Fu, Fu_no, u, u_no

        eta_flat = eta.reshape(B, C, -1).cpu().numpy()
        del eta

        eta_a = eta_flat[:, :, idx_a_np]
        eta_b = eta_flat[:, :, idx_b_np]
        del eta_flat

        cross += (eta_a * np.conj(eta_b)).sum(axis=0)
        del eta_a, eta_b

        print(f"      {end}/{N}", end="\r")
    print()
    return cross, N_total


def compute_coherence(cross, N_total, sigma2, idx_a, idx_b):
    """Normalized coherence |C(k,k')| / sqrt(sigma2(k) sigma2(k'))

    Returns
    -------
    coherence    : (C, P) float64
    noise_floor  : float, 1/sqrt(N_total)
    """
    print(f"[step 4] Computing normalized coherence (N={N_total})")
    C_hat = cross / N_total

    sigma2_np   = sigma2.cpu().numpy()
    C_ch        = sigma2_np.shape[0]
    sigma2_flat = sigma2_np.reshape(C_ch, -1)
    s2_a = sigma2_flat[:, idx_a]
    s2_b = sigma2_flat[:, idx_b]
    denom = np.maximum(np.sqrt(s2_a * s2_b), 1e-30)
    coherence = np.abs(C_hat) / denom

    noise_floor = 1.0 / np.sqrt(N_total)
    print(f"         Finite-sample noise floor: 1/sqrt({N_total}) = {noise_floor:.4f}")
    for c, name in enumerate(CHANNEL_NAMES):
        med = np.median(coherence[c])
        p95 = np.percentile(coherence[c], 95)
        print(f"         ch={name}: median={med:.4f}, 95th pctl={p95:.4f}")
    return coherence, noise_floor


def bin_and_plot(coherence, sep, noise_floor, num_bins, out_dir,
                 suffix="", title_extra=""):
    """Bin coherence by ||k-k'||, plot mean and 95th percentile per bin"""
    fig, axes = plt.subplots(1, NUM_CHANNELS, figsize=(6 * NUM_CHANNELS, 5))
    fig.suptitle(
        f"Off-diagonal cross-spectral coherence vs mode separation{title_extra}",
        fontsize=13, fontweight="bold",
    )

    bins    = np.linspace(0, sep.max(), num_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.clip(np.digitize(sep, bins) - 1, 0, num_bins - 1)

    for c, name in enumerate(CHANNEL_NAMES):
        ax = axes[c]
        mean_vals = np.full(num_bins, np.nan)
        p95_vals  = np.full(num_bins, np.nan)
        counts    = np.zeros(num_bins, dtype=int)

        for i in range(num_bins):
            mask = bin_idx == i
            counts[i] = mask.sum()
            if counts[i] > 0:
                vals = coherence[c, mask]
                mean_vals[i] = vals.mean()
                p95_vals[i]  = np.percentile(vals, 95)

        valid = counts > 0
        ax.plot(centers[valid], mean_vals[valid],
                color="steelblue", lw=1.5, label="mean coherence")
        ax.plot(centers[valid], p95_vals[valid],
                color="coral",     lw=1.2, ls="--", label="95th percentile")
        ax.axhline(noise_floor, color="grey", ls=":", lw=1.0,
                   label=rf"$1/\sqrt{{N}}={noise_floor:.3f}$")
        ax.set_xlabel(r"Mode separation $\|k - k'\|$")
        ax.set_ylabel("Coherence")
        ax.set_title(f"Channel {name}")
        ax.legend(fontsize=9)
        ax.grid(True, ls="--", alpha=0.5)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fname = f"{out_dir}/wss_coherence_vs_separation{suffix}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {fname}")


def stratified_plots(coherence, sep, kmag_a, kmag_b, noise_floor, num_bins,
                     out_dir, kmag_threshold=0.15):
    """Split pairs by spectral regime and plot each separately"""
    print(f"\n[step 6] Stratified plots (threshold ||k||={kmag_threshold})")
    low_a = kmag_a < kmag_threshold
    low_b = kmag_b < kmag_threshold

    strata = {
        "low_low":   low_a & low_b,
        "high_high": (~low_a) & (~low_b),
        "mixed":     low_a ^ low_b,
    }
    for label, mask in strata.items():
        n = mask.sum()
        print(f"    {label}: {n} pairs")
        if n < 100:
            print(f"    Skipping {label}, too few pairs")
            continue
        bin_and_plot(
            coherence[:, mask], sep[mask], noise_floor, num_bins,
            out_dir, suffix=f"_{label}",
            title_extra=f" [{label.replace('_', '-')}]",
        )


def figure_wss_coherence(coherence, sep, kmag_a, kmag_b, noise_floor,
                         num_bins, out_path, component=0, kmag_threshold=0.15):
    """Two-panel paper figure for a single component: all pairs vs mixed"""
    ch_name = CHANNEL_NAMES[component]
    coh = coherence[component]

    low_a = kmag_a < kmag_threshold
    low_b = kmag_b < kmag_threshold
    mixed = low_a ^ low_b

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    for ax, mask, label in [
        (axes[0], np.ones(len(sep), dtype=bool), "All pairs"),
        (axes[1], mixed, f"Mixed pairs ($\\|k\\| \\lessgtr {kmag_threshold}$)"),
    ]:
        s = sep[mask]
        c = coh[mask]
        bins    = np.linspace(0, s.max(), num_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        bin_idx = np.clip(np.digitize(s, bins) - 1, 0, num_bins - 1)

        mean_vals = np.full(num_bins, np.nan)
        p95_vals  = np.full(num_bins, np.nan)
        for i in range(num_bins):
            m = bin_idx == i
            if m.sum() > 0:
                vals = c[m]
                mean_vals[i] = vals.mean()
                p95_vals[i]  = np.percentile(vals, 95)

        valid = ~np.isnan(mean_vals)
        ax.plot(centers[valid], mean_vals[valid],
                color="steelblue", lw=2.0, label="Mean coherence")
        ax.plot(centers[valid], p95_vals[valid],
                color="coral",     lw=1.5, ls="--", label="95th percentile")
        ax.axhline(noise_floor, color="grey", ls=":", lw=1.2,
                   label=rf"$1/\sqrt{{N}} = {noise_floor:.3f}$")
        ax.set_ylabel("Coherence", fontsize=14)
        ax.set_title(f"{label} ({ch_name} component)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=12)

    axes[1].set_xlabel(r"Mode separation $\|k - k'\|$", fontsize=14)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Verify WSS assumption on NO residual")
    p.add_argument("--h5",             default=DEFAULT_H5)
    p.add_argument("--stats",          default=DEFAULT_STATS)
    p.add_argument("--model",          default=DEFAULT_MODEL)
    p.add_argument("--out",            default=DEFAULT_OUT)
    p.add_argument("--num_pairs",      type=int,   default=80_000)
    p.add_argument("--num_bins",       type=int,   default=100)
    p.add_argument("--chunk",          type=int,   default=8)
    p.add_argument("--dt",             type=float, default=0.02)
    p.add_argument("--lowpass_hz",     type=float, default=5.0)
    p.add_argument("--butter_order",   type=int,   default=4)
    p.add_argument("--kmag_threshold", type=float, default=0.15)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--component",      type=int,   default=0,
                   help="Component for paper figure (0=E, 1=N, 2=Z)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device      : {device}")
    print(f"Output dir  : {out_dir}/")
    print(f"Num pairs   : {args.num_pairs}")
    print(f"Seed        : {args.seed}\n")

    rng = np.random.default_rng(args.seed)

    H_real, sigma2 = load_spectral_model(args.model)
    _, Nx, Ny, Nf = H_real.shape

    idx_a, idx_b, sep, kmag_a, kmag_b = sample_mode_pairs(Nx, Ny, Nf, args.num_pairs, rng)

    cross, N_total = accumulate_cross_spectra(
        args.h5, args.stats, H_real, idx_a, idx_b,
        device=device, chunk_size=args.chunk,
        lowpass_hz=args.lowpass_hz, dt=args.dt, butter_order=args.butter_order,
    )

    coherence, noise_floor = compute_coherence(cross, N_total, sigma2, idx_a, idx_b)

    print(f"\n[step 5] Plotting coherence vs mode separation")
    bin_and_plot(coherence, sep, noise_floor, args.num_bins, str(out_dir))

    stratified_plots(coherence, sep, kmag_a, kmag_b, noise_floor,
                     args.num_bins, str(out_dir),
                     kmag_threshold=args.kmag_threshold)

    np_path = out_dir / "wss_coherence_raw.npz"
    np.savez(np_path,
             coherence=coherence.astype(np.float32),
             sep=sep, kmag_a=kmag_a, kmag_b=kmag_b,
             noise_floor=noise_floor, N_total=N_total)
    print(f"\n    Raw data saved to {np_path}")

    print("\n[step 7] Paper-quality WSS figure")
    figure_wss_coherence(
        coherence, sep, kmag_a, kmag_b, noise_floor,
        num_bins=args.num_bins,
        out_path=str(out_dir / "fig_wss_coherence.pdf"),
        component=args.component,
        kmag_threshold=args.kmag_threshold,
    )
    print("Done")


if __name__ == "__main__":
    main()