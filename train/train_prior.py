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

"""Unconditional GenCFD diffusion training with DDP

Trains the unconditional 3D denoiser used as the diffusion prior for
FreqNO-DPS.

Launch (8 GPUs, ~430k steps to match the paper):

    torchrun --nproc_per_node=8 -m scripts.train \\
        --batch_size 4 --num_train_steps 430000 \\
        --save_dir runs/gencfd_unconditional
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from utils.parser_utils import train_args
from utils.gencfd_utils import (
    get_dataset_loader,
    get_buffer_dict,
    create_denoiser,
    create_callbacks,
    save_json_file,
)
from train import training_loop
from train.trainers import DenoisingTrainer


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def main():
    # DDP init
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = (get_rank() == 0)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    args = train_args()

    if not hasattr(args, "dataset") or args.dataset is None:
        args.dataset = "HEMEW^S-3D_MIFNO_ONLINE"
    if not hasattr(args, "model_type") or args.model_type is None:
        args.model_type = "PreconditionedDenoiser3DGeoUncond"
    args.task = "solver"

    save_dir = os.path.join(os.getcwd(), args.save_dir)
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        print(f"[rank {get_rank()}] save_dir: {save_dir}", flush=True)

    # Dataset and dataloader
    train_dataloader, dataset, time_cond = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_worker=args.worker,
        prefetch_factor=2,
        split=False,
        split_ratio=1.0,
    )

    if get_world_size() > 1:
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=False,
        )
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=args.worker,
            pin_memory=True,
            prefetch_factor=2,
        )

    buffer_dict = get_buffer_dict(dataset=dataset, device=device)

    save_json_file(
        args=args,
        time_cond=time_cond,
        split_ratio=1.0,
        out_shape=dataset.output_shape,
        input_channel=dataset.input_channel,
        output_channel=dataset.output_channel,
        spatial_resolution=dataset.spatial_resolution,
        device=device,
        seed=0,
    )

    # Denoiser
    dtype = getattr(args, "dtype", torch.float32)
    denoising_model = create_denoiser(
        args=args,
        input_channels=dataset.input_channel,
        out_channels=dataset.output_channel,
        spatial_resolution=dataset.spatial_resolution,
        time_cond=time_cond,
        device=device,
        dtype=dtype,
        buffer_dict=buffer_dict,
    )
    denoiser = denoising_model.denoiser
    denoiser.to(device=device, dtype=dtype)

    if args.compile:
        if is_main:
            print("[compile] torch.compile(denoiser)", flush=True)
        denoiser = torch.compile(denoiser, mode="max-autotune")

    if get_world_size() > 1:
        denoiser = torch.nn.parallel.DistributedDataParallel(
            denoiser,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # Reassign back into the frozen DenoisingModel dataclass
    object.__setattr__(denoising_model, "denoiser", denoiser)

    if is_main:
        n_params = sum(p.numel() for p in denoising_model.denoiser.parameters()
                       if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}", flush=True)

    # Optimizer and trainer
    optim_params = [p for p in denoising_model.denoiser.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params, lr=args.peak_lr, weight_decay=args.weight_decay,
    )

    trainer = DenoisingTrainer(
        model=denoising_model,
        optimizer=optimizer,
        device=device,
        ema_decay=args.ema_decay,
        store_ema=True,
        track_memory=args.track_memory,
        use_mixed_precision=args.use_mixed_precision,
        is_compiled=args.compile,
        world_size=get_world_size(),
        local_rank=local_rank,
    )

    # Training loop
    start = time.time()
    writer = training_loop.SummaryWriter(log_dir=save_dir) if is_main else None

    training_loop.run(
        train_dataloader=train_dataloader,
        trainer=trainer,
        workdir=save_dir,
        total_train_steps=args.num_train_steps,
        metric_writer=writer,
        metric_aggregation_steps=args.metric_aggregation_steps,
        eval_every_steps=args.eval_every_steps,
        num_batches_per_eval=args.num_batches_per_eval,
        callbacks=create_callbacks(args, save_dir),
        compile_model=args.compile,
    )

    elapsed_h = (time.time() - start) / 3600.0
    if is_main:
        print(f"[train] done, elapsed {elapsed_h:.2f} h", flush=True)

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()