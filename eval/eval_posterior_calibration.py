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

"""K-realization posterior sampling for calibration analysis

For each of N_EVAL test samples, generates K posterior realizations with
shared sensor mask and MIFNO prediction but different initial noise
Reports 1-sigma and 2-sigma coverage, CI width, and per-realization rMAE

Single GPU:
    python -m eval.eval_posterior_calibration --model_dir <path> ...

Multi-GPU:
    torchrun --nproc_per_node=4 -m eval.eval_posterior_calibration ...

Set METHOD and WHAT_DPS below to choose the configuration
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Subset
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
from eval.mifno_helpers import build_mifno_model, mifno_prior_from_cond

import torch._dynamo
torch._dynamo.config.suppress_errors = True


# Checkpoints and data paths
PRIOR_CKPT  = "./checkpoints/gencfd_unconditional/checkpoint_430000.pth"
SPECTRAL_PT = "./checkpoints/spectral_model.pt"
SIGMA2_PT   = "./checkpoints/sigma2_iso.pt"
VAL_DIRS    = "./data/HEMEWS3D_S32_Z32_T320_fmax5_rot0_test"
SAVE_ROOT   = "./results"

# Method selection
METHOD   = "FreqNO_DPS"   # FreqNO_DPS | DPS_sensor_only | DPS_NO_iso
WHAT_DPS = "ODE"          # ODE | SDE

# Experiment knobs
N_EVAL         = 100
K_REALIZATIONS = 20
KEEP_FRAC      = 0.05
BASE_SEED      = 42


def build_likelihood(method, sensor_op, device):
    """Construct the DPS likelihood and sampler hparams for a given method

    SDE hparams are ODE hparams halved: the reverse SDE includes a Langevin
    correction that doubles the effective score coefficient relative to the
    probability-flow ODE, so DPS guidance terms also pick up a 2x factor and
    lambdas are halved to compensate
    """
    if method == "DPS_NO_iso":
        lk_NO = load_pixel_space_NO_likelihood(SIGMA2_PT, device=device)
        lk_obs = SparseSensorLikelihood(operator=sensor_op, sigma_obs=0.01)
        if WHAT_DPS == "SDE":
            lk_hp = {"lambda_NO": 5000.0, "lambda_obs": 11500.0}
        else:
            lk_hp = {"lambda_NO": 10000.0, "lambda_obs": 23000.0}
        # Lambdas are pre-applied inside the likelihood, so the sampler sees
        # lambda_sensor=1.0 and no separate NO term
        likelihood = StandardDPSCombinedLikelihood(
            likelihood_NO=lk_NO, likelihood_obs=lk_obs, **lk_hp,
        )
        return likelihood, {"lambda_sensor": 1.0, "lambda_NO": 0.0}

    lk_NO = load_spectral_likelihood(SPECTRAL_PT, device=device)
    lk_obs = SparseSensorLikelihood(operator=sensor_op, sigma_obs=0.01)
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
    """Build the DPS ODE or SDE sampler"""
    common = dict(
        input_shape=shape, scheme=scheme, denoise_fn=denoise_fn, tspan=tspan,
        likelihood=likelihood, apply_denoise_at_end=True,
        device=device, dtype=dtype,
        lambda_sensor=hp["lambda_sensor"], lambda_NO=hp["lambda_NO"],
    )
    if WHAT_DPS == "SDE":
        return DPSSampler(integrator=EulerMaruyama(terminal_only=True), **common)
    return DPSOdeSampler(integrator=ExplicitEuler(terminal_only=True), **common)


def compute_summary(targ_all, gen_all, norm_all, N, K):
    """Coverage at 1-sigma and 2-sigma, CI width, per-realization rMAE"""
    gen_mean = gen_all.float().mean(dim=1)
    gen_std  = gen_all.float().std(dim=1)
    targ_f   = targ_all.float()

    cov_1 = ((targ_f >= gen_mean -       gen_std) & (targ_f <= gen_mean +       gen_std)).float().mean().item()
    cov_2 = ((targ_f >= gen_mean - 2.0 * gen_std) & (targ_f <= gen_mean + 2.0 * gen_std)).float().mean().item()

    while norm_all.dim() < gen_std.dim():
        norm_all = norm_all.unsqueeze(-1)
    norm_b = norm_all.expand_as(gen_std).float()
    mean_width = (4.0 * gen_std / (norm_b + 1e-8)).mean().item()
    mean_std   = gen_std.mean().item()

    eps = 1e-2
    targ_flat = targ_all.float().view(N, 1, 3, -1).expand(N, K, 3, -1)
    gen_flat  = gen_all.float().view(N, K, 3, -1)
    rMAE_per_k = ((gen_flat - targ_flat) / (targ_flat.abs() + eps)).abs().mean(dim=(-1, -2))
    rMAE_mean = rMAE_per_k.mean(dim=1)
    rMAE_std  = rMAE_per_k.std(dim=1)
    cov_var   = (rMAE_std / (rMAE_mean + 1e-8)).mean().item()

    print(f"\n{'='*60}")
    print(f"Posterior calibration: {METHOD} / {WHAT_DPS}  |  N={N} x K={K}")
    print(f"{'='*60}")
    print(f"  1-sigma coverage:   {cov_1:.4f}  (nominal: 0.6827)")
    print(f"  2-sigma coverage:   {cov_2:.4f}  (nominal: 0.9545)")
    print(f"  Mean CI width (2s): {mean_width:.4f}  (normalized)")
    print(f"  Mean posterior std: {mean_std:.4e}")
    print(f"  rMAE mean of means: {rMAE_mean.mean():.4f}")
    print(f"  rMAE mean of stds:  {rMAE_std.mean():.4f}")
    print(f"  rMAE coeff of var:  {cov_var:.4f}")
    print(f"{'='*60}")

    return {
        "cov_1sigma": cov_1, "cov_2sigma": cov_2,
        "mean_ci_width": mean_width, "mean_posterior_std": mean_std,
        "rMAE_mean_of_means": rMAE_mean.mean().item(),
        "rMAE_mean_of_stds":  rMAE_std.mean().item(),
        "rMAE_coeff_of_var":  cov_var,
    }


def main():
    args = inference_args()
    rank, world_size, local_rank, device = ddp_setup()
    print(f"[rank {rank}/{world_size}] device={device} | {METHOD} / {WHAT_DPS}")

    cfg_path = Path(args.model_dir).resolve() / "training_config.json"
    train_cfg = load_json_file(str(cfg_path))
    replace_args(args, train_cfg)
    args.task = "superresolver"

    # Dataset (first N_EVAL samples)
    val_dirs = train_cfg.get("val_dirs", VAL_DIRS)
    ds_full = MIFNOGenCFDDataset(
        dir_data=[val_dirs],
        T_out=320, S_in=32, S_in_z=32, S_out=32,
        transform_a="normal", transform_traces="distance_Vs", N=None,
    )
    ds = Subset(ds_full, list(range(N_EVAL)))
    print(f"Posterior calibration: {len(ds)} samples x {K_REALIZATIONS} realizations")

    sampler = (DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                  shuffle=False, drop_last=False)
               if world_size > 1 else None)
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    sampler=sampler, num_workers=4, pin_memory=True)

    # Velocity z-score stats
    vel_mean, vel_std = load_vel_stats(device=device, dtype=args.dtype)
    vel_mean = vel_mean.view(1, 3, 1, 1, 1)
    vel_std  = vel_std.view(1, 3, 1, 1, 1)

    # Prior denoiser
    args.compile = False
    prior = create_denoiser(
        args=args,
        input_channels=ds_full.input_channel, out_channels=ds_full.output_channel,
        spatial_resolution=ds_full.spatial_resolution,
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
    sampler_shape = (3, *ds_full.spatial_resolution)
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
                f"posterior_calibration_{METHOD}_{WHAT_DPS}_keep{int(KEEP_FRAC*100)}_steps{args.sampling_steps}_K{K_REALIZATIONS}" /
                ckpt_name)
    partials_dir = save_dir / "partials"
    partials_dir.mkdir(parents=True, exist_ok=True)

    # Inference loop
    prior.denoiser.eval()
    results = []

    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(dl, desc="Posterior calibration")):
            targ_norm   = batch["target_cond"].to(device)
            norm_traces = batch["norm_traces"].to(device)
            cond = {k: v.to(device) for k, v in batch["cond"].items()}

            # Exact global index under DistributedSampler(shuffle=False)
            global_idx = bidx * world_size + rank

            # MIFNO prediction (once per sample)
            u_NO = mifno_prior_from_cond(mifno, cond, device, args.dtype)
            u_NO = (u_NO - vel_mean) / vel_std

            # Sensor mask (deterministic per-sample, shared across K realizations)
            torch.manual_seed(BASE_SEED + global_idx)
            torch.cuda.manual_seed_all(BASE_SEED + global_idx)
            mask = random_sensor_mask(targ=targ_norm, keep_frac=KEEP_FRAC)
            sensor_op.set_mask(mask)
            y_obs = sensor_op(targ_norm)

            # K realizations with different initial noise
            gen_samples = []
            for k in range(K_REALIZATIONS):
                seed = BASE_SEED + global_idx * K_REALIZATIONS + k
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                with torch.enable_grad():
                    gen_norm = dps.generate(num_samples=1, y=(u_NO, y_obs), cond=None)
                gen_samples.append(denorm_vel(gen_norm.detach(), vel_mean, vel_std).cpu())
                del gen_norm

            results.append({
                "targ":  denorm_vel(targ_norm, vel_mean, vel_std).cpu(),
                "mifno": denorm_vel(u_NO,      vel_mean, vel_std).cpu(),
                "gen":   torch.cat(gen_samples, dim=0),
                "norm":  norm_traces.cpu(),
                "mask":  mask.cpu(),
            })

            del u_NO, y_obs, gen_samples
            torch.cuda.empty_cache()

    # Save per-rank partials with fsync
    part_path = partials_dir / f"part_rank{rank:03d}.pt"
    torch.save(results, part_path)
    fd = os.open(str(part_path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(f"[rank {rank}] saved {len(results)} samples -> {part_path}")

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    if not is_main():
        ddp_cleanup()
        return

    # Belt-and-braces against NFS visibility lag
    for r in range(world_size):
        p = partials_dir / f"part_rank{r:03d}.pt"
        for _ in range(60):
            if p.exists() and p.stat().st_size > 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Partial {p} not visible 60s after barrier")

    # Gather all ranks
    all_results = []
    for r in range(world_size):
        all_results.extend(torch.load(partials_dir / f"part_rank{r:03d}.pt", map_location="cpu"))

    N = len(all_results)
    print(f"\nGathered {N} samples x {K_REALIZATIONS} realizations")

    targ_all  = torch.cat([r["targ"]  for r in all_results], dim=0)
    mifno_all = torch.cat([r["mifno"] for r in all_results], dim=0)
    gen_all   = torch.stack([r["gen"] for r in all_results], dim=0)
    norm_all  = torch.cat([r["norm"]  for r in all_results], dim=0)
    mask_all  = torch.cat([r["mask"]  for r in all_results], dim=0)

    consolidated_path = save_dir / "posterior_samples.pt"
    torch.save({
        "targ": targ_all, "mifno": mifno_all, "gen": gen_all,
        "norm": norm_all, "mask": mask_all,
        "K": K_REALIZATIONS, "N": N, "keep_frac": KEEP_FRAC,
        "method": METHOD, "sampler": WHAT_DPS,
    }, consolidated_path)
    print(f"Saved consolidated data -> {consolidated_path}")

    summary = compute_summary(targ_all, gen_all, norm_all, N=N, K=K_REALIZATIONS)

    summary_path = save_dir / "calibration_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"method={METHOD}, sampler={WHAT_DPS}\n")
        f.write(f"N={N}, K={K_REALIZATIONS}, keep_frac={KEEP_FRAC}\n")
        f.write(f"sampler_hparams={hp}\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print(f"Summary saved -> {summary_path}")

    ddp_cleanup()


if __name__ == "__main__":
    main()