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

"""Regenerate paper-quality WSS figure from saved raw data."""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CHANNEL_NAMES = ["E–W", "N–S", "Z"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="wss_verification/wss_coherence_raw.npz")
    p.add_argument("--output", type=str, default="figures/fig_wss_coherence.pdf")
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--kmag_threshold", type=float, default=0.15)
    p.add_argument("--num_bins", type=int, default=100)
    args = p.parse_args()

    raw = np.load(args.input)
    coh = raw["coherence"][args.component]
    sep = raw["sep"]
    kmag_a = raw["kmag_a"]
    kmag_b = raw["kmag_b"]
    noise_floor = float(raw["noise_floor"])
    ch_name = CHANNEL_NAMES[args.component]

    low_a = kmag_a < args.kmag_threshold
    low_b = kmag_b < args.kmag_threshold
    mixed = low_a ^ low_b

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, mask, label in [
        (axes[0], np.ones(len(sep), dtype=bool), "All pairs"),
        (axes[1], mixed,
         f"Mixed pairs ($\\|k\\| \\lessgtr {args.kmag_threshold}$)"),
    ]:
        s = sep[mask]
        c = coh[mask]

        bins = np.linspace(0, s.max(), args.num_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        bin_idx = np.clip(np.digitize(s, bins) - 1, 0, args.num_bins - 1)

        mean_vals = np.full(args.num_bins, np.nan)
        p95_vals = np.full(args.num_bins, np.nan)

        for i in range(args.num_bins):
            m = bin_idx == i
            if m.sum() > 0:
                vals = c[m]
                mean_vals[i] = vals.mean()
                p95_vals[i] = np.percentile(vals, 95)

        valid = ~np.isnan(mean_vals)
        ax.plot(centers[valid], mean_vals[valid],
                color="steelblue", lw=2.0, label="Mean coherence")
        ax.plot(centers[valid], p95_vals[valid],
                color="coral", lw=1.5, ls="--", label="95th percentile")
        ax.axhline(noise_floor, color="grey", ls=":", lw=1.2,
                   label=rf"$1/\sqrt{{N}} = {noise_floor:.3f}$")

        ax.set_ylabel("Coherence", fontsize=18)
        ax.set_title(f"{label} ({ch_name} component)", fontsize=20)
        ax.legend(fontsize=15)
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=12)
        ax.set_xlabel(r"Mode separation $\|k - k'\|$", fontsize=18)


    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()