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

"""Distributed Data Parallel (DDP) helpers."""

import os
import torch
import torch.distributed as dist


def ddp_setup():
    """Initialise DDP if launched with torchrun, otherwise fall back to single-GPU.

    Returns
    -------
    rank, world_size, local_rank, device
    """
    if "RANK" not in os.environ:  # not launched with torchrun
        return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def ddp_cleanup():
    """Tear down the process group (no-op when DDP is not active)."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main() -> bool:
    """Return True on rank-0 (or when DDP is not active)."""
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0