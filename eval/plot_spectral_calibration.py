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

"""Spectral calibration figures for the appendix

Renders four figures from a trained spectral model:
- NO error profile     : sigma2_NO, P_u, gamma vs ||k||
- Transfer function    : |H(k)| and arg H(k) vs ||k||
- LMMSE vs DPS weights : spectral weight at multiple noise levels
- Expected guidance    : E[|score|^2] vs ||k||, ours vs isotropic DPS

The spectral model was estimated on data with a 5 Hz Butterworth lowpass
along the temporal axis

Usage:
    python -m scripts.plot_spectral_calibration \
        --spectral_pt ./checkpoints/spectral_model.pt \
        --output_dir ./figures/spectral
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


DT = 0.02
NX, NY, NT = 32, 32, 320

COMP_NAMES  = ["E-W", "N-S", "Z"]
COMP_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]
NUM_BINS    = 200

# Diffusion noise levels at which to compare spectral weights
SIGMA_TAUS = [0.002, 0.01, 0.1, 1.0, 10.0, 80.0]
TAU_CMAP   = plt.cm.viridis_r

# Isotropic sigma^2_NO used as the DPS baseline reference
SIGMA2_ISO_REF = 3.54


def build_kmag(Nx: int, Ny: int, Nt: int) -> np.ndarray:
    """Wavenumber magnitude grid for rFFT output: shape (Nx, Ny, Nt//2+1)"""
    kx = np.fft.fftfreq(Nx)
    ky = np.fft.fftfreq(Ny)
    kt = np.fft.rfftfreq(Nt)
    gx, gy, gt = np.meshgrid(kx, ky, kt, indexing="ij")
    return np.sqrt(gx ** 2 + gy ** 2 + gt ** 2)


def radial_bin(k_mag, values, num_bins=NUM_BINS):
    """Mean of values within radial bins of ||k||"""
    k = k_mag.ravel()
    v = values.ravel()
    valid = ~np.isnan(k) & ~np.isnan(v) & np.isfinite(v)
    k, v = k[valid], v[valid]

    if len(k) == 0:
        return {"k_centers": np.linspace(0, 1, num_bins),
                "mean": np.full(num_bins, np.nan)}

    bins = np.linspace(0.0, k.max(), num_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.clip(np.digitize(k, bins) - 1, 0, num_bins - 1)

    out = np.full(num_bins, np.nan)
    for i in range(num_bins):
        m = idx == i
        if m.any():
            out[i] = v[m].mean()
    return {"k_centers": centers, "mean": out}


def radial_bin_percentiles(k_mag, values, num_bins=NUM_BINS, percentiles=(5, 50, 95)):
    """Percentiles of values within radial bins of ||k||"""
    k = k_mag.ravel()
    v = values.ravel()
    valid = ~np.isnan(k) & ~np.isnan(v) & np.isfinite(v)
    k, v = k[valid], v[valid]

    bins = np.linspace(0.0, k.max(), num_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.clip(np.digitize(k, bins) - 1, 0, num_bins - 1)

    out = {"k_centers": centers, **{f"p{p}": np.full(num_bins, np.nan) for p in percentiles}}
    for i in range(num_bins):
        m = idx == i
        if m.any():
            vals = v[m]
            for p in percentiles:
                out[f"p{p}"][i] = np.percentile(vals, p)
    return out


def load_spectral_model(pt_path: str) -> dict[str, np.ndarray]:
    """Load spectral model and return numpy arrays

    Returns H (real), H_complex, sigma2_NO, P_u, each of shape (C, Nx, Ny, Nf)
    """
    spec = torch.load(pt_path, map_location="cpu", weights_only=True)
    print(f"Loaded: {pt_path}")
    return {
        "H":         spec["H_real"].numpy(),
        "H_complex": spec["H_complex_real"].numpy() + 1j * spec["H_complex_imag"].numpy(),
        "sigma2_NO": spec["sigma2_real"].numpy(),
        "P_u":       spec["P_u"].numpy(),
    }


def figure_spectral_error_profile(spec, k_mag, output_path):
    """sigma2_NO, P_u, gamma vs ||k||, one panel each, all components overlaid"""
    H, sigma2, P_u = spec["H"], spec["sigma2_NO"], spec["P_u"]
    gamma = sigma2 / (H ** 2 * P_u + 1e-12)

    quantities = [
        (sigma2, r"$\sigma^2_{\mathrm{NO}}(k)$", "NO residual variance"),
        (P_u,    r"$P_{\mathbf{u}}(k)$",         "Signal power spectrum"),
        (gamma,  r"$\gamma(k)$",                  "Relative NO error"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
    for ax, (vals, ylabel, title) in zip(axes, quantities):
        for c in range(H.shape[0]):
            b = radial_bin(k_mag, vals[c])
            v = ~np.isnan(b["mean"])
            ax.plot(b["k_centers"][v], b["mean"][v],
                    color=COMP_COLORS[c], lw=1.8, label=COMP_NAMES[c])
        ax.set_yscale("log")
        ax.set_xlabel(r"$\|k\|$", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(title, fontsize=18)
        ax.set_xlim(0, None)
        ax.legend(fontsize=15)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        if title == "Relative NO error":
            ax.axhline(1.0, color="grey", ls="--", lw=1.0, alpha=0.6)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def figure_transfer_function(spec, k_mag, output_path):
    """|H(k)| and arg H(k) vs ||k|| with percentile bands on phase"""
    H, H_complex = spec["H"], spec["H_complex"]
    H_mag = np.abs(H_complex)
    phase = np.angle(H_complex)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # |H(k)|
    ax = axes[0]
    for c in range(H.shape[0]):
        b = radial_bin(k_mag, np.abs(H[c]))
        v = ~np.isnan(b["mean"])
        ax.plot(b["k_centers"][v], b["mean"][v],
                color=COMP_COLORS[c], lw=1.8, label=COMP_NAMES[c])
    ax.axhline(1.0, color="grey", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel(r"$\|k\|$", fontsize=16)
    ax.set_ylabel(r"$|H(k)|$", fontsize=16)
    ax.set_title("Transfer function magnitude", fontsize=18)
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=15)
    ax.grid(True, ls="--", alpha=0.3)

    # arg H(k) with 5-95 band
    ax = axes[1]
    for c in range(H.shape[0]):
        phase_c = phase[c].copy()
        phase_c[H_mag[c] < 1e-2] = np.nan  # mask near-zero modes where phase is meaningless
        b = radial_bin_percentiles(k_mag, phase_c, percentiles=(5, 50, 95))
        v = ~np.isnan(b["p50"])
        ax.plot(b["k_centers"][v], b["p50"][v],
                color=COMP_COLORS[c], lw=1.8, label=COMP_NAMES[c])
        ax.fill_between(b["k_centers"][v], b["p5"][v], b["p95"][v],
                        color=COMP_COLORS[c], alpha=0.15)
    ax.axhline(0.0, color="grey", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel(r"$\|k\|$", fontsize=16)
    ax.set_ylabel(r"$\arg\, \tilde{H}(k)$ [rad]", fontsize=16)
    ax.set_title("Transfer function phase", fontsize=18)
    ax.set_xlim(0, None)
    ax.legend(fontsize=15)
    ax.grid(True, ls="--", alpha=0.3)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def figure_spectral_weights(spec, k_mag, output_path, component=0):
    """LMMSE-corrected (left) vs standard DPS (right) spectral weight vs ||k||

    Color encodes sigma_tau. Corrected weight decreases at high ||k||,
    DPS weight increases
    """
    H      = spec["H"][component]
    sigma2 = spec["sigma2_NO"][component]
    P_u    = spec["P_u"][component]
    H2     = H ** 2
    H_abs  = np.abs(H)
    eps    = 1e-12
    n_taus = len(SIGMA_TAUS)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for i, st in enumerate(SIGMA_TAUS):
        col = TAU_CMAP(i / max(n_taus - 1, 1))
        s2t = st ** 2

        alpha       = P_u / (s2t + P_u + eps)
        Lambda_corr = sigma2 + H2 * s2t * alpha + eps
        weight_corr = 2.0 * H_abs * alpha / Lambda_corr

        Lambda_dps  = sigma2 + H2 * s2t + eps
        weight_dps  = 2.0 * H_abs / Lambda_dps

        label = rf"$\sigma_\tau = {st}$"
        for ax, w in zip(axes, (weight_corr, weight_dps)):
            b = radial_bin(k_mag, w)
            v = ~np.isnan(b["mean"])
            ax.semilogy(b["k_centers"][v], b["mean"][v],
                        color=col, lw=1.5, label=label)

    axes[0].set_title(rf"LMMSE-corrected $\tilde{{\Lambda}}_\tau$ ({COMP_NAMES[component]})",
                      fontsize=11)
    axes[1].set_title(rf"Standard DPS $\tilde{{\Lambda}}^{{\mathrm{{DPS}}}}_\tau$ ({COMP_NAMES[component]})",
                      fontsize=11)
    for ax in axes:
        ax.set_xlabel(r"$\|k\|$")
        ax.set_ylabel("Spectral weight")
        ax.set_xlim(0, None)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", ls="--", alpha=0.3)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def figure_expected_guidance(spec, k_mag, output_path, component=0):
    """Expected NO guidance magnitude vs ||k||, ours vs isotropic DPS

    Ours: spectrally shaped, varies with ||k||
    Isotropic DPS: horizontal reference at 4 / (sigma2_iso + sigma_tau^2)
    """
    H      = spec["H"][component]
    sigma2 = spec["sigma2_NO"][component]
    P_u    = spec["P_u"][component]
    H2     = H ** 2
    eps    = 1e-12
    n_taus = len(SIGMA_TAUS)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    for i, st in enumerate(SIGMA_TAUS):
        col = TAU_CMAP(i / max(n_taus - 1, 1))
        s2t = st ** 2

        alpha       = P_u / (s2t + P_u + eps)
        Lambda_corr = sigma2 + H2 * s2t * alpha + eps
        guide_corr  = 4.0 * H2 * alpha ** 2 / Lambda_corr

        b = radial_bin(k_mag, guide_corr)
        v = ~np.isnan(b["mean"])
        ax.semilogy(b["k_centers"][v], b["mean"][v],
                    color=col, lw=2.0, ls="-", label=rf"$\sigma_\tau = {st}$")

        # Isotropic DPS reference at this sigma_tau
        ax.axhline(4.0 / (SIGMA2_ISO_REF + s2t), color=col, lw=1.5, ls="--", alpha=0.6)

    style_handles = [
        Line2D([0], [0], color="k", lw=2.0, ls="-", label="Spectrally shaped (ours)"),
        Line2D([0], [0], color="k", lw=1.5, ls="--", alpha=0.6, label="Isotropic DPS"),
    ]
    tau_handles = [
        Line2D([0], [0], color=TAU_CMAP(i / max(n_taus - 1, 1)),
               lw=2.5, label=rf"$\sigma_\tau = {st}$")
        for i, st in enumerate(SIGMA_TAUS)
    ]
    ax.legend(handles=style_handles + tau_handles, fontsize=15,
              loc="lower left", ncol=2)
    ax.set_xlabel(r"$\|k\|$", fontsize=18)
    ax.set_ylabel(r"$\mathbb{E}\left[|\mathrm{score}|^2\right]$", fontsize=18)
    ax.set_title(f"Expected NO guidance magnitude ({COMP_NAMES[component]})", fontsize=20)
    ax.set_xlim(0, None)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot spectral calibration figures")
    parser.add_argument("--spectral_pt", type=str, required=True,
                        help="Path to the spectral model .pt file")
    parser.add_argument("--output_dir", type=str, default="./figures/spectral")
    parser.add_argument("--component", type=int, default=0,
                        help="Component for weight and guidance plots: 0=E, 1=N, 2=Z")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_mag = build_kmag(NX, NY, NT)
    print(f"k_mag shape: {k_mag.shape}, range [{k_mag.min():.4f}, {k_mag.max():.4f}]")

    spec = load_spectral_model(args.spectral_pt)

    figure_spectral_error_profile(spec, k_mag, out_dir / "fig_spectral_error_profile.pdf")
    figure_transfer_function    (spec, k_mag, out_dir / "fig_transfer_function.pdf")
    figure_spectral_weights     (spec, k_mag, out_dir / "fig_spectral_weights.pdf",
                                 component=args.component)
    figure_expected_guidance    (spec, k_mag, out_dir / "fig_expected_guidance.pdf",
                                 component=args.component)


if __name__ == "__main__":
    main()