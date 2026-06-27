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

"""DPS inference and evaluation for sparse-sensor seismic reconstruction.

Single GPU:
    python -m eval.eval_dps_final --model_dir <path> --batch_size 2 ...

Multi-GPU:
    torchrun --nproc_per_node=4 -m eval.eval_dps_final ...

Set METHOD and WHAT_DPS below to choose the inversion configuration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
import torch
from numpy.fft import rfft, rfftfreq
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.parser_utils import inference_args
from utils.gencfd_utils import create_denoiser, load_json_file, replace_args
from utils.diffusion_utils import get_time_step_scheduler, get_diffusion_scheme
from utils.ddp_utils import ddp_setup, ddp_cleanup, is_main

from train.train_states import DenoisingModelTrainState
from train.trainers import DenoisingTrainer
from dataloader.dataset import MIFNOGenCFDDataset

from dps.likelihoods import (
    CombinedLikelihood,
    StandardDPSCombinedLikelihood,
    SparseSensorLikelihood,
    load_spectral_likelihood,
    load_pixel_space_NO_likelihood,
)
from dps.measurements import SparseSensorOperator
from diffusion.samplers import DPSOdeSampler, DPSSampler
from solvers.ode import ExplicitEuler
from solvers.sde import EulerMaruyama

from eval.normalization import load_vel_stats, denorm_vel
from eval.sensor_utils import random_sensor_mask
from eval.metrics import (
    myL1, myL2,
    significant_duration_maps_all_sensors
)
from eval.mifno_helpers import build_mifno_model, mifno_prior_from_cond

import torch._dynamo
torch._dynamo.config.suppress_errors = True


# Checkpoints and data paths (override for your environment)
PRIOR_CKPT  = "./checkpoints/gencfd_unconditional/checkpoint_430000.pth"
SPECTRAL_PT = "./checkpoints/spectral_model.pt"
SIGMA2_PT   = "./checkpoints/sigma2_iso.pt"
VAL_DIRS    = "./data/HEMEWS3D_S32_Z32_T320_fmax5_rot0_test"
SAVE_ROOT   = "./results"

# Method selection
METHOD   = "FreqNO_DPS"   # FreqNO_DPS | DPS_sensor_only | DPS_NO_iso
WHAT_DPS = "ODE"          # ODE | SDE

# Experiment knobs.
KEEP_FRAC   = 0.05
MAX_BATCHES = None        # None processes the entire loader
DT          = 0.02
SEED        = 100

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def build_likelihood(method, sensor_op, device):
    """Construct the DPS likelihood and sampler hparams for a given method.

    SDE hparams are ODE hparams halved: the reverse SDE includes a Langevin
    correction that doubles the effective score coefficient relative to the
    probability-flow ODE, so DPS guidance terms also pick up a 2x factor and
    lambdas are halved to compensate
    """
    if method == "DPS_NO_iso":
        lk_NO = load_pixel_space_NO_likelihood(SIGMA2_PT, device=device)
        lk_obs = SparseSensorLikelihood(
            operator=sensor_op, sigma_obs=0.01,
            P_u=None, use_spectral_denom=False,
        )
        if WHAT_DPS == "SDE":
            lk_hp = {"lambda_NO": 5000.0,  "lambda_obs": 11500.0}
        else:
            lk_hp = {"lambda_NO": 10000.0,  "lambda_obs": 23000.0}
        likelihood = StandardDPSCombinedLikelihood(
            likelihood_NO=lk_NO, likelihood_obs=lk_obs, **lk_hp,
        )
        return likelihood, {"lambda_sensor": 1.0, "lambda_NO": 0.0}

    lk_NO = load_spectral_likelihood(SPECTRAL_PT, device=device)
    lk_obs = SparseSensorLikelihood(
        operator=sensor_op, sigma_obs=0.01,
    )
    likelihood = CombinedLikelihood(likelihood_NO=lk_NO, likelihood_obs=lk_obs)

    if method == "FreqNO_DPS":
        hp = ({"lambda_sensor": 11500.0, "lambda_NO": 0.17} if WHAT_DPS == "SDE"
              else {"lambda_sensor": 23000.0, "lambda_NO": 0.35})
    elif method == "DPS_sensor_only":
        hp = ({"lambda_sensor": 11500.0, "lambda_NO": 0.0} if WHAT_DPS == "SDE"
              else {"lambda_sensor": 23000.0, "lambda_NO": 0.0})
    else:
        raise ValueError(f"Unknown METHOD: {method}")

    return likelihood, hp


def build_sampler(likelihood, hp, shape, scheme, denoise_fn, tspan, device, dtype):
    """Build the DPS ODE or SDE sampler."""
    common = dict(
        input_shape=shape, scheme=scheme, denoise_fn=denoise_fn, tspan=tspan,
        likelihood=likelihood, apply_denoise_at_end=True,
        device=device, dtype=dtype,
        lambda_sensor=hp["lambda_sensor"], lambda_NO=hp["lambda_NO"],
    )
    if WHAT_DPS == "SDE":
        return DPSSampler(integrator=EulerMaruyama(terminal_only=True), **common)
    return DPSOdeSampler(integrator=ExplicitEuler(terminal_only=True), **common)


def _broadcast_norm(norm, ref):
    while norm.dim() < ref.dim():
        norm = norm.unsqueeze(-1)
    return norm.expand_as(ref)


def _fourier_band_spectra(values_4d, low_band, mid_band, high_band, dt=DT):
    """Mean |FFT| in three frequency bands. values_4d: (N, X, Y, T)."""
    freqs = rfftfreq(values_4d.shape[-1], d=dt)
    mag = np.abs(rfft(values_4d, axis=-1))
    lo  = mag[..., (freqs >= low_band[0])  & (freqs <= low_band[1])].mean(-1)
    mid = mag[..., (freqs >  mid_band[0])  & (freqs <= mid_band[1])].mean(-1)
    hi  = mag[..., (freqs >  high_band[0]) & (freqs <= high_band[1])].mean(-1)
    return lo, mid, hi


def _bands_over_components(x5d, dt=DT):
    """Three-band FFT spectra averaged over the 3 velocity components."""
    lo, mi, hi = zip(*[
        _fourier_band_spectra(x5d[:, c], (0., 1.), (1., 2.), (2., 5.), dt=dt)
        for c in range(3)
    ])
    return sum(lo) / 3, sum(mi) / 3, sum(hi) / 3


def main():
    args = inference_args()
    rank, world_size, local_rank, device = ddp_setup()
    print(f"[rank {rank}/{world_size}] device={device} | {METHOD} / {WHAT_DPS}")

    cfg_path = Path(args.model_dir).resolve() / "training_config.json"
    train_cfg = load_json_file(str(cfg_path))
    replace_args(args, train_cfg)
    args.task = "superresolver"

    # Dataset.
    val_dirs = train_cfg.get("val_dirs", VAL_DIRS)
    ds = MIFNOGenCFDDataset(
        dir_data=[val_dirs],
        T_out=320, S_in=32, S_in_z=32, S_out=32,
        transform_a="normal", transform_traces="distance_Vs", N=None,
    )
    print(f"Validation dataset: {len(ds)} samples")

    sampler = (DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                  shuffle=False, drop_last=False)
               if world_size > 1 else None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    sampler=sampler, num_workers=1, pin_memory=True)

    # Velocity z-score stats
    vel_mean, vel_std = load_vel_stats(device=device, dtype=args.dtype)
    vel_mean = vel_mean.view(1, 3, 1, 1, 1)
    vel_std  = vel_std.view(1, 3, 1, 1, 1)

    # Prior denoiser (unconditional GenCFD)
    args.compile = False
    prior = create_denoiser(
        args=args,
        input_channels=ds.input_channel, out_channels=ds.output_channel,
        spatial_resolution=ds.spatial_resolution,
        time_cond=False, device=device, dtype=args.dtype, buffer_dict=None,
    )
    trainer = DenoisingTrainer(
        model=prior, optimizer=None, device=device,
        ema_decay=args.ema_decay, store_ema=True, track_memory=False,
        use_mixed_precision=args.use_mixed_precision, is_compiled=False,
    )
    state = DenoisingModelTrainState.restore_from_checkpoint(
        PRIOR_CKPT, model=prior.denoiser, optimizer=None,
    )
    denoise_fn = trainer.inference_fn_from_state_dict(
        state, use_ema=True, denoiser=prior.denoiser,
        task="superresolver", lead_time=False,
    )

    # MIFNO surrogate
    mifno = build_mifno_model(device=device, dtype=args.dtype)
    mifno.eval()

    # Likelihood and sampler
    sampler_shape = (3, *ds.spatial_resolution)
    C, Nx, Ny, Nt = sampler_shape
    dummy_mask = torch.ones((1, C, Nx, Ny, 1), device=device, dtype=args.dtype)
    sensor_op = SparseSensorOperator(mask=dummy_mask).to(device)

    likelihood, hp = build_likelihood(METHOD, sensor_op, device)
    scheme = get_diffusion_scheme(args, device)
    tspan = get_time_step_scheduler(args=args, scheme=scheme, device=device, dtype=args.dtype)
    dps = build_sampler(likelihood, hp, sampler_shape, scheme, denoise_fn, tspan,
                        device, args.dtype)
    print(f"DPS {WHAT_DPS} ready | steps={len(tspan)} | hparams={hp}")

    # Output paths
    ckpt_name = Path(PRIOR_CKPT).stem
    save_dir = (Path(SAVE_ROOT) /
                f"{METHOD}_{WHAT_DPS}_keep{int(KEEP_FRAC*100)}_steps{args.sampling_steps}" /
                ckpt_name)
    partials_dir = save_dir / "partials"
    partials_dir.mkdir(parents=True, exist_ok=True)

    # Inference loop
    all_targ, all_mifno, all_gen, all_norms, all_source = [], [], [], [], []

    prior.denoiser.eval()
    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(dl, desc="DPS inference")):
            if MAX_BATCHES is not None and bidx >= MAX_BATCHES:
                break

            targ_norm = batch["target_cond"].to(device)
            cond = {k: v.to(device) for k, v in batch["cond"].items()}
            B = targ_norm.size(0)

            # MIFNO prediction in z-score space
            u_NO = mifno_prior_from_cond(mifno, cond, device, args.dtype)
            u_NO = (u_NO - vel_mean) / vel_std

            # One IID sparse mask per sample
            masks = torch.cat(
                [random_sensor_mask(targ=targ_norm[i:i+1], keep_frac=KEEP_FRAC)
                for i in range(B)], dim=0
            )
            sensor_op.set_mask(masks)
            y_obs = sensor_op(targ_norm)

            with torch.enable_grad():
                gen_norm = dps.generate(num_samples=B, y=(u_NO, y_obs), cond=None)
            gen_norm = gen_norm.detach()

            # De-normalize to physical units
            targ_phys  = denorm_vel(targ_norm, vel_mean, vel_std)
            mifno_phys = denorm_vel(u_NO,      vel_mean, vel_std)
            gen_phys   = denorm_vel(gen_norm,  vel_mean, vel_std)

            all_targ.append(targ_phys.cpu())
            all_mifno.append(mifno_phys.cpu())
            all_gen.append(gen_phys.cpu())
            all_norms.append(batch["norm_traces"])
            all_source.append(cond["s"].cpu())

            del gen_norm, u_NO, y_obs, masks, gen_phys, mifno_phys, targ_phys
            torch.cuda.empty_cache()

    # Save per-rank partials
    part_path = partials_dir / f"part_rank{rank:03d}.pt"
    torch.save({
        "targ":   torch.cat(all_targ),
        "mifno":  torch.cat(all_mifno),
        "gen":    torch.cat(all_gen),
        "norm":   torch.cat(all_norms),
        "source": torch.cat(all_source),
    }, part_path)

    # Force durability before the barrier so main rank sees a complete file
    fd = os.open(str(part_path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    if not is_main():
        ddp_cleanup()
        return

    # Belt-and-braces against NFS visibility lag after the barrier
    for r in range(world_size):
        p = partials_dir / f"part_rank{r:03d}.pt"
        for _ in range(60):
            if p.exists() and p.stat().st_size > 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Partial {p} not visible 60s after barrier")

    # Gather all ranks
    parts = [torch.load(partials_dir / f"part_rank{r:03d}.pt", map_location="cpu")
            for r in range(world_size)]
    targ_all  = torch.cat([p["targ"]  for p in parts])
    mifno_all = torch.cat([p["mifno"] for p in parts])
    gen_all   = torch.cat([p["gen"]   for p in parts])
    norm_all  = torch.cat([p["norm"]  for p in parts])
    N = targ_all.shape[0]

    targ_np = targ_all.view(N, 3, -1).numpy()
    gen_np  = gen_all.view(N, 3, -1).numpy()
    init_np = mifno_all.view(N, 3, -1).numpy()

    # rMAE / rRMSE
    eps = 1e-2
    base = np.abs(targ_np) + eps
    rel_gen  = (gen_np  - targ_np) / base
    rel_init = (init_np - targ_np) / base

    df = pd.DataFrame(index=np.arange(N), dtype=np.float32)
    df["rMAE_gen"]   = myL1(rel_gen,  axis=2).mean(axis=1)
    df["rRMSE_gen"]  = myL2(rel_gen,  axis=2).mean(axis=1)
    df["rMAE_init"]  = myL1(rel_init, axis=2).mean(axis=1)
    df["rRMSE_init"] = myL2(rel_init, axis=2).mean(axis=1)

    # FFT band ratios in trace-normalized space
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

    # Significant duration D5-95
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
    csv_path = save_dir / f"metrics_{METHOD}_{WHAT_DPS}_{args.sampling_steps}steps.csv"
    df_final.to_csv(csv_path, index=True)
    print(f"\nMetrics saved -> {csv_path}")
    print(df_final.tail(3).to_string())

    ddp_cleanup()

if __name__ == "__main__":
    main()