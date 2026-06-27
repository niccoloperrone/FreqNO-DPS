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

"""Argument parsers for training and inference"""

from argparse import ArgumentParser

import torch


def parse_tuple(value):
    """Parse '(a,b,c)' or 'a,b,c' into a tuple of ints"""
    if value is None or value.lower() == "none":
        return None
    try:
        return tuple(map(int, value.strip("()").split(",")))
    except ValueError:
        raise ValueError(f"Invalid tuple format: {value}")


def str_to_bool(value):
    """Parse a boolean from a string"""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "f", "0", "no", "n"):
        return False
    raise ValueError(f"Invalid boolean string: {value}")


def add_base_options(parser: ArgumentParser):
    g = parser.add_argument_group("base")
    g.add_argument("--work_dir", default="datasets", type=str, help="If empty, will use defaults according to the specified dataset.")
    g.add_argument("--save_dir",  default=None, type=str,
                   help="Directory where training or evaluation results are saved")
    g.add_argument("--model_dir", default=None, type=str,
                   help="Path to a pretrained model directory for inference")
    g.add_argument("--dtype",     default=torch.float32, type=torch.dtype)
    g.add_argument("--use_mixed_precision", default=False, type=str_to_bool,
                   help="Enable mixed-precision training")


def add_data_options(parser: ArgumentParser):
    g = parser.add_argument_group("dataset")
    g.add_argument("--dataset", default="HEMEW^S-3D_MIFNO_ONLINE", type=str,
                   choices=["HEMEW^S-3D_MIFNO_ONLINE"])
    g.add_argument("--batch_size", default=2, type=int)
    g.add_argument("--worker",     default=0, type=int,
                   help="Number of dataloader workers")


def add_model_options(parser: ArgumentParser):
    g = parser.add_argument_group("model")
    g.add_argument("--model_type", default="PreconditionedDenoiser3DGeoUncond",
                   type=str, choices=["PreconditionedDenoiser3DGeoUncond"])
    g.add_argument("--num_channels",     default=(64, 128, 256), type=parse_tuple)
    g.add_argument("--downsample_ratio", default=(2, 2, 2),       type=parse_tuple)
    g.add_argument("--use_attention", default=True, type=str_to_bool, help="Choose if attention blocks should be used")
    g.add_argument("--num_blocks",       default=4,   type=int)
    g.add_argument("--num_heads",        default=8,   type=int)
    g.add_argument("--normalize_qk",     default=False, type=str_to_bool)
    g.add_argument("--noise_embed_dim",  default=128, type=int)
    g.add_argument("--use_position_encoding", default=True, type=str_to_bool)
    g.add_argument("--padding_method",   default="zeros", type=str,
                   choices=["circular", "constant", "lonlat", "latlon", "zeros"])
    g.add_argument("--dropout_rate",     default=0.0, type=float)
    g.add_argument("--sigma_data",       default=1.0, type=float)
    g.add_argument("--resize_to_shape", default=None, type=parse_tuple, help="Choose a shape to resize inside the UNet. Necessary if dataset resolution changes")
    g.add_argument("--compile", action="store_true", default=False,
                   help="Wrap the denoiser with torch.compile")


def add_denoiser_options(parser: ArgumentParser):
    g = parser.add_argument_group("denoiser")
    g.add_argument("--diffusion_scheme", default="create_variance_exploding",
                   choices=["create_variance_preserving", "create_variance_exploding"])
    g.add_argument("--sigma", default="exponential_noise_schedule",
                   choices=["exponential_noise_schedule", "power_noise_schedule",
                            "tangent_noise_schedule"])
    g.add_argument("--noise_sampling", default="log_uniform_sampling", type=str,
                   choices=["log_uniform_sampling", "time_uniform_sampling",
                            "normal_sampling"])
    g.add_argument("--noise_weighting", default="edm_weighting", type=str,
                   choices=["edm_weighting"])
    g.add_argument("--num_eval_noise_levels",  default=5,    type=int)
    g.add_argument("--num_eval_cases_per_lvl", default=1,    type=int)
    g.add_argument("--min_eval_noise_lvl",     default=1e-3, type=float)
    g.add_argument("--max_eval_noise_lvl",     default=50.0, type=float)
    g.add_argument("--consistent_weight",      default=0.0,  type=float)


def add_trainer_options(parser: ArgumentParser):
    g = parser.add_argument_group("trainer")
    g.add_argument("--ema_decay",    default=0.999, type=float)
    g.add_argument("--peak_lr",      default=3e-4,  type=float)
    g.add_argument("--weight_decay", default=0.01,  type=float)
    g.add_argument("--task", default="solver", type=str,
                   choices=["solver", "superresolver"])


def add_training_options(parser: ArgumentParser):
    g = parser.add_argument_group("training")
    g.add_argument("--num_train_steps",         default=100_000, type=int)
    g.add_argument("--metric_aggregation_steps", default=200,    type=int)
    g.add_argument("--eval_every_steps",         default=1000,   type=int)
    g.add_argument("--num_batches_per_eval",     default=2,      type=int)
    g.add_argument("--checkpoints",              default=True,   type=str_to_bool)
    g.add_argument("--save_every_n_steps",       default=2000,   type=int)
    g.add_argument("--track_memory", action="store_true", default=False)


def add_sampler_options(parser: ArgumentParser):
    g = parser.add_argument_group("sampler")
    g.add_argument("--time_step_scheduler", default="edm_noise_decay", type=str,
                   choices=["edm_noise_decay", "exponential_noise_decay", "uniform_time"])
    g.add_argument("--sampling_steps",       default=64,   type=int)
    g.add_argument("--apply_denoise_at_end", default=True, type=str_to_bool)
    g.add_argument("--return_full_paths",    default=False, type=str_to_bool)
    g.add_argument("--rho",                  default=7,    type=int)


def add_sde_options(parser: ArgumentParser):
    g = parser.add_argument_group("sde")
    g.add_argument("--integrator",    default="EulerMaruyama", type=str,
                   choices=["EulerMaruyama", "Heun"])
    g.add_argument("--time_axis_pos", default=-1,  type=int)
    g.add_argument("--terminal_only", default=True, type=str_to_bool)


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_denoiser_options(parser)
    add_trainer_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def inference_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_denoiser_options(parser)
    add_trainer_options(parser)
    add_sde_options(parser)
    add_sampler_options(parser)
    return parser.parse_args()