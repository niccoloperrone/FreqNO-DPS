# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by The CAM Lab at ETH Zurich.
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

"""Training callback library."""

from collections.abc import Mapping, Sequence
import os
from typing import Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

from train import trainers
from train.train_states import _clean_ckpt_keys 

Tensor = torch.Tensor
ComputedMetrics = Mapping[str, Tensor | Mapping[str, Tensor]]
Trainer = trainers.BaseTrainer


class Callback:
    """Abstract base class for callbacks

    Hooks are executed at fixed positions inside train.run. The execution flow:

        callbacks.on_train_begin()
        while training:
            callbacks.on_train_batches_begin()
            run_train_steps()
            callbacks.on_train_batches_end()
            if should_run_evaluation:
                callbacks.on_eval_batches_begin()
                run_eval_steps()
                callbacks.on_eval_batches_end()
        callbacks.on_train_end()

    When multiple callbacks are used, on_{train,eval}_batches_end methods
    are called in reverse order so they resemble the __exit__/__enter__
    of Python contexts
    """

    def __init__(self, log_dir: Optional[str] = None):
        self._metric_writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    @property
    def metric_writer(self) -> SummaryWriter:
        return self._metric_writer

    @metric_writer.setter
    def metric_writer(self, writer: SummaryWriter) -> None:
        self._metric_writer = writer

    def on_train_begin(self, trainer: Trainer) -> None:
        pass

    def on_train_batches_begin(self, trainer: Trainer) -> None:
        pass

    def on_train_batches_end(self, trainer: Trainer, train_metrics: ComputedMetrics) -> None:
        pass

    def on_eval_batches_begin(self, trainer: Trainer) -> None:
        pass

    def on_eval_batches_end(self, trainer: Trainer, eval_metrics: ComputedMetrics) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass


# This callback does not seem to work with `utils.primary_process_only`.
class TrainStateCheckpoint(Callback):
    """Periodically saves train state checkpoints"""

    def __init__(
        self,
        base_dir: str,
        folder_prefix: str = "checkpoints",
        train_state_field: str = "default",
        save_every_n_step: int = 1000,
    ):
        self.save_dir = os.path.join(base_dir, folder_prefix)
        self.train_state_field = train_state_field
        self.save_every_n_steps = save_every_n_step
        self.last_eval_metric = {}
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_begin(self, trainer: Trainer) -> None:
        """Restore the most recent checkpoint if one exists"""
        ckpt_path = self._get_latest_checkpoint()
        if ckpt_path is None:
            return

        ckpt = torch.load(ckpt_path, weights_only=True)

        # Strip torch.compile prefix if the stored model was compiled
        if ckpt["is_compiled"]:
            ckpt["model_state_dict"] = {
                k.replace("_orig_mod.", "", 1): v
                for k, v in ckpt["model_state_dict"].items()
            }
        trainer.model.denoiser.load_state_dict(ckpt["model_state_dict"])

        # Optimizer state may not load cleanly if param groups changed
        opt_state = ckpt.get("optimizer_state_dict", None)
        if opt_state is not None:
            try:
                trainer.optimizer.load_state_dict(opt_state)
                print("[INFO] Loaded optimizer state from checkpoint")
            except ValueError as e:
                print(
                    "[WARN] Optimizer param groups changed since the checkpoint, "
                    f"starting with a fresh optimizer\n       {e}"
                )
        else:
            print("[INFO] No optimizer_state_dict in checkpoint, using fresh optimizer")

        # EMA
        if getattr(trainer, "store_ema", False) and ckpt.get("ema_param"):
            ema_clean = _clean_ckpt_keys(ckpt["ema_param"])
            trainer.train_state.ema_model.load_state_dict(ema_clean, strict=False)
            trainer.train_state.ema = trainer.train_state.ema_parameters

        trainer.train_state.step = ckpt["step"]
        print(f"Resumed training from {ckpt_path}")

    def on_train_batches_end(self, trainer: Trainer, train_metrics: ComputedMetrics) -> None:
        cur_step = trainer.train_state.step
        if cur_step % self.save_every_n_steps == 0:
            self._save_checkpoint(trainer, cur_step, train_metrics)

    def on_eval_batches_end(self, trainer: Trainer, eval_metrics: ComputedMetrics) -> None:
        self.last_eval_metric = eval_metrics

    def on_train_end(self, trainer: Trainer) -> None:
        self._save_checkpoint(trainer, trainer.train_state.step, self.last_eval_metric, force=True)

    def _save_checkpoint(self, trainer, step, metrics, force=False):
        ckpt = {
            "model_state_dict":     trainer.model.denoiser.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "ema_param":            trainer.train_state.ema if trainer.store_ema else None,
            "step":                 step,
            "metrics":              metrics,
            "is_compiled":          trainer.is_compiled,
        }
        torch.save(ckpt, os.path.join(self.save_dir, f"checkpoint_{step}.pth"))

    def _get_latest_checkpoint(self) -> Optional[str]:
        ckpts = [f for f in os.listdir(self.save_dir) if f.endswith(".pth")]
        if not ckpts:
            return None
        ckpts.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
        return os.path.join(self.save_dir, ckpts[-1])


class TqdmProgressBar(Callback):
  """Tqdm progress bar callback to monitor training progress in real time."""

  def __init__(
      self,
      total_train_steps: int | None,
      train_monitors: Sequence[str],
      eval_monitors: Sequence[str] = (),
  ):
    """ProgressBar constructor.

    Args:
      total_train_steps: the total number of training steps, which is displayed
        as the maximum progress on the bar.
      train_monitors: keys in the training metrics whose values are updated on
        the progress bar after every training metric aggregation.
      eval_monitors: same as `train_monitors` except applying to evaluation.
    """
    super().__init__()
    self.total_train_steps = total_train_steps
    self.train_monitors = train_monitors
    self.eval_monitors = eval_monitors
    self.current_step = 0
    self.eval_postfix = {}  # keeps record of the most recent eval monitor
    self.bar = None

  def on_train_begin(self, trainer: Trainer) -> None:
    del trainer
    self.bar = tqdm.tqdm(total=self.total_train_steps, unit="step")

  def on_train_batches_end(
      self, trainer: Trainer, train_metrics: ComputedMetrics
  ) -> None:
    assert self.bar is not None
    self.bar.update(trainer.train_state.step - self.current_step)
    self.current_step = trainer.train_state.step
    postfix = {
        monitor: train_metrics[monitor] for monitor in self.train_monitors
    }
    self.bar.set_postfix(**postfix, **self.eval_postfix)

  def on_eval_batches_end(
      self, trainer: Trainer, eval_metrics: ComputedMetrics
  ) -> None:
    del trainer
    self.eval_postfix = {
        monitor: eval_metrics[monitor].item() for monitor in self.eval_monitors
    }

  def on_train_end(self, trainer: Trainer) -> None:
    del trainer
    assert self.bar is not None
    self.bar.close()