# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich.
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

"""Denoising model wrapper for diffusion-based generation training"""

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

import diffusion as dfn_lib


Tensor = torch.Tensor
TensorDict = Mapping[str, Tensor]
BatchType = Mapping[str, Union[np.ndarray, Tensor]]
ModelVariable = Union[dict, tuple[dict, ...], Mapping[str, dict]]
PyTree = Any
LossAndAux = tuple[Tensor, tuple[TensorDict, PyTree]]
Metrics = dict


class BaseModel(ABC):
    """Base class for models wrapping neural modules for trainer interaction"""

    @abstractmethod
    def initialize(self) -> ModelVariable:
        raise NotImplementedError

    @abstractmethod
    def loss_fn(self, params, batch, mutables, **kwargs) -> LossAndAux:
        raise NotImplementedError

    def eval_fn(self, variables, batch, **kwargs) -> TensorDict:
        raise NotImplementedError

    @staticmethod
    def inference_fn(variables: PyTree, **kwargs) -> Callable[..., Any]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True, kw_only=True)
class DenoisingModel(BaseModel):
    """Trains a denoiser to remove Gaussian noise from samples"""

    spatial_resolution: Sequence[int]
    denoiser: nn.Module
    noise_sampling: dfn_lib.NoiseLevelSampling
    noise_weighting: dfn_lib.NoiseLossWeighting
    num_eval_noise_levels: int = 5
    num_eval_cases_per_lvl: int = 1
    min_eval_noise_lvl: float = 1e-3
    max_eval_noise_lvl: float = 50.0

    consistent_weight: float = 0
    device: Any | None = None
    dtype: torch.dtype = torch.float32

    input_channel: int = 1
    task: str = "solver"
    time_cond: bool = False

    def initialize(self, batch_size: int, time_cond: bool = False):
        """Dummy initialization, retained to satisfy BaseModel ABC

        Not called in normal flow: training builds the denoiser via
        create_denoiser, inference loads from checkpoint
        """
        X, Y, T = self.spatial_resolution
        x = torch.ones((batch_size, 3, X, Y, T), dtype=self.dtype, device=self.device)
        sigma = torch.ones((batch_size,), dtype=self.dtype, device=self.device)
        return self.denoiser(x=x, sigma=sigma)

    def loss_fn(self, batch: dict, mutables: Optional[dict] = None):
        """Standard denoising loss: weighted L2 between denoised and target

        Returns
        -------
        loss    : scalar Tensor for backward
        metrics : dict of scalar floats for logging
        """
        x = batch["target_cond"]
        batch_size = x.shape[0]

        # Sample noise level and EDM weighting
        sigma = self.noise_sampling(shape=(batch_size,))
        weights = self.noise_weighting(sigma)
        if weights.ndim != x.ndim:
            weights = weights.view(-1, *([1] * (x.ndim - 1)))

        noise = torch.randn(x.shape, dtype=self.dtype, device=self.device)
        if sigma.ndim != x.ndim:
            noised = x + noise * sigma.view(-1, *([1] * (x.ndim - 1)))
        else:
            noised = x + noise * sigma

        denoised = self.denoiser(x=noised, sigma=sigma)
        loss = torch.mean(weights * torch.square(denoised - x))

        return loss, {"loss": loss.item()}

    @staticmethod
    def inference_fn(
        denoiser: nn.Module,
        task: str = "solver",
        lead_time: bool = False,
    ) -> Callable[..., Tensor]:
        """Build the per-step denoising function used by the DPS sampler

        The unconditional GenCFD prior is invoked as denoiser(x, sigma),
        ignoring any conditioning since the model was trained without it
        """
        del task, lead_time  # accepted for API stability, not used

        def _denoise(
            x: Tensor,
            sigma: float | Tensor,
            cond: Mapping[str, Tensor] | None = None,
        ) -> Tensor:
            if not torch.is_tensor(sigma):
                sigma = sigma * torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)
            return denoiser.forward(x=x, sigma=sigma)

        return _denoise