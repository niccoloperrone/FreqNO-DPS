# Copyright 2024 The CAM Lab at ETH Zurich.
# Modifications made by Niccolò Perrone, Politecnico di Milano /
# CentraleSupélec LMPS, 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GenCFD assembly: dataset, model, sampler, callbacks, and config I/O"""

import json
import os
import re
from argparse import ArgumentParser
from typing import Callable, Dict, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

import diffusion.diffusion as dfn_lib
from dataloader.dataset import MIFNOGenCFDDataset
from diffusion.samplers import OdeSampler, Sampler, SdeSampler
from model.building_blocks.unets.unets3d import PreconditionedDenoiser3DGeoUncond
from model.probabilistic_diffusion.denoising_model import DenoisingModel
from solvers.ode import ExplicitEuler, HeunsMethod
from solvers.sde import EulerMaruyama
from utils import callbacks
from utils.callbacks import TrainStateCheckpoint
from utils.diffusion_utils import (
    get_diffusion_scheme,
    get_noise_sampling,
    get_noise_weighting,
    get_sampler_args,
    get_time_step_scheduler,
)
from utils.model_utils import get_denoiser_args, get_model_args


Tensor = torch.Tensor
TensorMapping = Dict[str, Tensor]
DenoiseFn = Callable[[Tensor, Tensor, TensorMapping | None], Tensor]


DEFAULT_TRAIN_DIR = "./data/HEMEWS3D_S32_Z32_T320_fmax5_rot0_train"


def get_dataset(name: str, is_time_dependent: bool = False) -> Dataset:
    if name != "HEMEW^S-3D_MIFNO_ONLINE":
        raise ValueError(f"Unknown dataset: {name}")

    dataset = MIFNOGenCFDDataset(
        dir_data=[DEFAULT_TRAIN_DIR],
        T_out=320, S_in=32, S_in_z=32, S_out=32,
        transform_a="normal", transform_traces="distance_Vs", N=None,
    )
    time_cond = False

    return (dataset, time_cond) if is_time_dependent else dataset


def get_dataset_loader(
    name: str,
    batch_size: int = 5,
    num_worker: int = 0,
    prefetch_factor: int = 2,
    split: bool = True,
    split_ratio: float = 0.9995,
):
    """Return train/eval dataloaders, or a single dataloader if split=False"""
    dataset, time_cond = get_dataset(name=name, is_time_dependent=True)

    if split:
        train_size = int(split_ratio * len(dataset))
        eval_size  = len(dataset) - train_size
        train_ds, eval_ds = random_split(dataset, [train_size, eval_size])
        train_dl = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
            num_workers=num_worker, prefetch_factor=prefetch_factor, drop_last=True,
        )
        eval_dl = DataLoader(
            eval_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
            num_workers=num_worker, prefetch_factor=prefetch_factor,
        )
        return train_dl, eval_dl, dataset, time_cond

    dl = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=num_worker, prefetch_factor=prefetch_factor,
    )
    return dl, dataset, time_cond


def get_buffer_dict(dataset: Dataset, device: torch.device = None) -> dict:
    """Identity normalization buffers, since the dataset z-scores its targets"""
    return {
        "mean_training_input":  torch.zeros((dataset.input_channel,)),
        "mean_training_output": torch.zeros((dataset.output_channel,)),
        "std_training_input":   torch.ones((dataset.input_channel,)),
        "std_training_output":  torch.ones((dataset.output_channel,)),
    }


def get_model(
    args: ArgumentParser,
    in_channels: int,
    out_channels: int,
    spatial_resolution: tuple,
    time_cond: bool,
    device: torch.device = None,
    buffer_dict: dict = None,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Build the unconditional GenCFD denoiser"""
    if args.model_type != "PreconditionedDenoiser3DGeoUncond":
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    model_args = get_model_args(
        args=args, in_channels=in_channels, out_channels=out_channels,
        spatial_resolution=spatial_resolution, time_cond=time_cond,
        device=device, buffer_dict=buffer_dict, dtype=dtype,
    )
    return PreconditionedDenoiser3DGeoUncond(**model_args)


def get_denoising_model(
    args, input_channels, spatial_resolution, time_cond,
    denoiser, noise_sampling, noise_weighting,
    device=None, dtype=torch.float32,
) -> DenoisingModel:
    return DenoisingModel(**get_denoiser_args(
        args=args, input_channels=input_channels,
        spatial_resolution=spatial_resolution, time_cond=time_cond,
        denoiser=denoiser,
        noise_sampling=noise_sampling, noise_weighting=noise_weighting,
        device=device, dtype=dtype,
    ))


def create_denoiser(
    args: ArgumentParser,
    input_channels: int,
    out_channels: int,
    spatial_resolution: Sequence[int],
    time_cond: bool,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    buffer_dict: dict = None,
):
    """Build the unconditional GenCFD denoising model"""
    model = get_model(
        args=args,
        in_channels=input_channels,
        out_channels=out_channels,
        spatial_resolution=spatial_resolution,
        time_cond=time_cond,
        device=device, buffer_dict=buffer_dict, dtype=dtype,
    )
    return get_denoising_model(
        args=args, input_channels=input_channels,
        spatial_resolution=spatial_resolution, time_cond=time_cond,
        denoiser=model,
        noise_sampling=get_noise_sampling(args, device),
        noise_weighting=get_noise_weighting(args, device),
        device=device, dtype=dtype,
    )


def create_callbacks(args: ArgumentParser, save_dir: str) -> Sequence[callbacks.Callback]:
    out = []
    if args.checkpoints:
        out.append(TrainStateCheckpoint(
            base_dir=save_dir, save_every_n_step=args.save_every_n_steps,
        ))
    return tuple(out)


def save_json_file(
    args: ArgumentParser,
    time_cond: bool,
    split_ratio: float,
    out_shape: Sequence[int],
    input_channel: int,
    output_channel: int,
    spatial_resolution: Sequence[int],
    device: torch.device = None,
    seed: int = None,
):
    config = {
        "save_dir":             args.save_dir,
        "dataset":              args.dataset,
        "batch_size":           args.batch_size,
        "split_ratio":          split_ratio,
        "worker":               args.worker,
        "time_cond":            time_cond,
        "out_shape":            out_shape,
        "input_channel":        input_channel,
        "output_channel":       output_channel,
        "spatial_resolution":   spatial_resolution,
        "model_type":           args.model_type,
        "compile":              args.compile,
        "num_heads":            args.num_heads,
        "use_mixed_precision":  args.use_mixed_precision,
        "num_train_steps":      args.num_train_steps,
        "task":                 args.task,
        "device":               str(device) if device is not None else None,
        "seed":                 seed,
    }
    config_path = os.path.join(args.save_dir, "training_config.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Training configuration saved to {config_path}")


def load_json_file(config_path: str):
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found at {config_path}, using passed arguments")
        return None


def replace_args(args: ArgumentParser, train_args: dict):
    """Apply training-time argument values to inference-time args

    Skip-listed arguments are kept from the inference-time command line
    """
    skip = {"dataset", "save_dir", "batch_size", "compile"}
    for key, value in train_args.items():
        if key in skip:
            continue
        if hasattr(args, key):
            setattr(args, key, value)


def create_sampler(
    args: ArgumentParser,
    input_shape: tuple,
    denoise_fn: DenoiseFn,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Sampler:
    """Build a Sampler for unconditional generation, EulerMaruyama or Heun"""
    scheme = get_diffusion_scheme(args, device)

    if args.integrator == "EulerMaruyama":
        integrator = EulerMaruyama(
            time_axis_pos=args.time_axis_pos, terminal_only=args.terminal_only,
        )
    elif args.integrator == "Heun":
        integrator = HeunsMethod(
            time_axis_pos=args.time_axis_pos, terminal_only=args.terminal_only,
        )
    else:
        raise ValueError(f"Unknown integrator: {args.integrator}")

    tspan = get_time_step_scheduler(args=args, scheme=scheme, device=device, dtype=dtype)
    sampler_args = get_sampler_args(
        args=args, input_shape=input_shape, scheme=scheme,
        denoise_fn=denoise_fn, tspan=tspan, integrator=integrator,
        device=device, dtype=dtype,
    )
    return OdeSampler(**sampler_args) if args.integrator == "Heun" else SdeSampler(**sampler_args)