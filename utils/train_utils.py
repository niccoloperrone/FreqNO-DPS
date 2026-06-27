# Copyright 2024 The CAM Lab at ETH Zurich.
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

"""Training-time helpers: metrics and memory tracking"""

from functools import wraps
from typing import Union

import numpy as np
import torch
from torchmetrics import Metric


Tensor = torch.Tensor


class StdMetric(Metric):
    """Streaming standard deviation"""

    def __init__(self):
        super().__init__()
        self.add_state("total",          default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_of_squares", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count",          default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, values: Union[Tensor, float]):
        if isinstance(values, float):
            values = torch.as_tensor(values)
        self.total          += values.sum()
        self.sum_of_squares += (values ** 2).sum()
        self.count          += values.numel()

    def compute(self):
        mean     = self.total / self.count
        variance = (self.sum_of_squares / self.count) - mean ** 2
        return torch.sqrt(torch.clamp(variance, min=0.0))


def compute_memory(func):
    """Decorator: if trainer.track_memory is True, inject peak GPU mem into the metrics dict"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.device.type == "cuda" and self.track_memory:
            torch.cuda.reset_peak_memory_stats(self.device)

        metrics = func(self, *args, **kwargs)

        if self.device.type == "cuda" and self.track_memory:
            peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            if isinstance(metrics, dict):
                metrics["mem"] = peak_gb

        return metrics

    return wrapper


def is_scalar(value) -> bool:
    """Check if a value is a 0-d or 1-element scalar"""
    if isinstance(value, (int, float, np.number)):
        return True
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.ndim == 0 or value.numel() <= 1
    return False