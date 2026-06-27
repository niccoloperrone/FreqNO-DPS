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

"""MIFNO model construction and inference helpers.

Public API
----------
- ``build_mifno_model``       – build MIFNO_3D and load checkpoint.
- ``mifno_prior_from_cond``   – run MIFNO forward to get (B,3,X,Y,T) prior.
"""

from __future__ import annotations

import os
import torch

from MIFNO.models.mifno_model import MIFNO_3D

# Default checkpoint path (override via argument if needed).
MIFNO_CKPT = ("./checkpoints/mifno_3d_S32_T320.pt")


def build_mifno_model(
    device: torch.device,
    dtype: torch.dtype,
    ckpt_path: str = MIFNO_CKPT,
) -> MIFNO_3D:
    """Build the MIFNO_3D architecture and load pre-trained weights.

    Architecture matches the HEMEWS3D JSON config:
    ``input_dim=4, output_dim=1, source_dim=9, n_layers=16, dv=16``.
    """
    n_layers = 16

    list_D3 = [32] * 12 + [64, 128, 256, 320]
    list_M3 = [16] * 12 + [16, 32, 32, 32]

    mifno = MIFNO_3D(
        list_D1=[32] * n_layers,
        list_D2=[32] * n_layers,
        list_D3=list_D3,
        list_M1=[16] * n_layers,
        list_M2=[16] * n_layers,
        list_M3=list_M3,
        width=16,
        input_dim=4,
        output_dim=1,
        source_dim=9,
        branching_index=4,
        n_layers=n_layers,
        padding=0,
        original=False,
    ).to(device=device, dtype=dtype)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"MIFNO checkpoint not found at {ckpt_path}. "
            f"Download the pretrained weights and place them at this path, "
            f"or pass --mifno_ckpt to override."
        )

    print(f"[MIFNO] Loading checkpoint from {ckpt_path}", flush=True)
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        mifno.load_state_dict(state["state_dict"])
    elif isinstance(state, dict) and all(
        not k.startswith("module.") for k in state.keys()
    ):
        mifno.load_state_dict(state)
    else:
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        mifno.load_state_dict(new_state, strict=False)

    return mifno


@torch.no_grad()
def mifno_prior_from_cond(
    mifno: MIFNO_3D,
    cond: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run MIFNO to produce a velocity prior from conditioning fields.

    Parameters
    ----------
    cond : dict
        Must contain ``"a"`` (geology, B×1×X×Y×Z), ``"s_mifno"`` (source
        coords, B×6), and ``"angle"`` (angles, B×3).

    Returns
    -------
    Tensor (B, 3, X, Y, T)
    """
    for key in ("a", "s_mifno", "angle"):
        if key not in cond:
            raise ValueError(f"cond must contain key '{key}' for MIFNO.")

    geo = cond["a"].to(device=device, dtype=dtype)          # (B, 1, X, Y, Z_geo)
    s = cond["s_mifno"].to(device=device, dtype=dtype)      # (B, 6)

    # MIFNO expects (B, X, Y, Z, C_in)
    x_mifno = geo.permute(0, 2, 3, 4, 1)

    uE, uN, uZ = mifno(x_mifno, s)

    # Squeeze output_dim==1 and stack components
    prior = torch.stack(
        [uE.squeeze(-1), uN.squeeze(-1), uZ.squeeze(-1)],
        dim=1,
    )  # (B, 3, X, Y, T)

    return prior