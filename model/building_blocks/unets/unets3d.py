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

"""3D U-Net denoiser models.

Intended for inputs with dimensions (batch, time, x, y, channels). The U-Net
stacks successively apply 2D downsampling/upsampling in space only. At each
resolution, an axial attention block (involving space and/or time) is applied.
"""
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import default_init
from model.building_blocks.stacks.dstack_3d import DStack
from model.building_blocks.stacks.ustack_3d import UStack
from model.building_blocks.embeddings.fourier_emb import FourierEmbedding
from model.building_blocks.layers.convolutions import ConvLayer
from MIFNO.models.mifno_model import MIFNO_3D


Tensor = torch.Tensor


def _maybe_broadcast_to_list(
    source: bool | Sequence[bool], reference: Sequence[Any]
) -> list[bool]:
  """Broadcasts to a list with the same length if applicable."""
  if isinstance(source, bool):
    return [source] * len(reference)
  else:
    if len(source) != len(reference):
      raise ValueError(f"{source} must have the same length as {reference}!")
    return list(source)

class UNet3D(nn.Module):
    """UNet model for 3D time-space input.

    This model processes 3D spatiotemporal data using a UNet architecture. It
    progressively downsamples the input for efficient feature extraction at
    multiple scales. Features are extracted using 2D spatial convolutions along
    with spatial and/or temporal axial attention blocks. Upsampling and
    combination of features across scales produce an output with the same shape as
    the input.

    Attributes:
      out_channels: Number of output channels (should match the input).
      kernel_dim: Dimension of spatial resolution. Adds info if it's a 2 or 3D dataset
      resize_to_shape: Optional input resizing shape. Facilitates greater
        downsampling flexibility. Output is resized to the original input shape.
      num_channels: Number of feature channels in intermediate convolutions.
      downsample_ratio: Spatial downsampling ratio per resolution (must evenly
        divide spatial dimensions).
      num_blocks: Number of residual convolution blocks per resolution.
      noise_embed_dim: Embedding dimensions for noise levels.
      input_proj_channels: Number of input projection channels.
      output_proj_channels: Number of output projection channels.
      padding: 2D padding type for spatial convolutions.
      dropout_rate: Dropout rate between convolution layers.
      use_spatial_attention: Whether to enable axial attention in spatial
        directions at each resolution.
      use_temporal_attention: Whether to enable axial attention in the temporal
        direction at each resolution.
      use_position_encoding: Whether to add position encoding before axial
        attention.
      num_heads: Number of attention heads.
      cond_resize_method: Resize method for channel-wise conditioning.
      cond_embed_dim: Embedding dimension for channel-wise conditioning.
    """

    def __init__(
          self,
          in_channels: int,
          out_channels: int,
          spatial_resolution: Sequence[int],
          time_cond: bool,
          num_channels: Sequence[int] = (128, 256, 256),
          downsample_ratio: Sequence[int] = (2, 2, 2),
          num_blocks: int = 4,
          noise_embed_dim: int = 128,
          input_proj_channels: int = 128,
          output_proj_channels: int = 128, 
          padding_method: str = 'circular',
          dropout_rate: float = 0.0,
          use_spatial_attention: bool | Sequence[bool] = (False, False, False),
          use_position_encoding: bool = True,
          num_heads: int = 8,
          normalize_qk: bool = False,
          dtype: torch.dtype = torch.float32,
          device: torch.device = None,
          buffer_dict: dict = None
      ):
      super(UNet3D, self).__init__()

      if buffer_dict:
        # Store normalization parameters as buffers for all datasets!
        for name, tensor in buffer_dict.items():
          self.register_buffer(name, tensor)

      self.in_channels = in_channels
      self.out_channels = out_channels
      self.num_channels = num_channels
      self.spatial_resolution = spatial_resolution
      self.time_cond = time_cond
      self.kernel_dim = len(spatial_resolution)
      self.downsample_ratio = downsample_ratio
      self.num_blocks = num_blocks
      self.noise_embed_dim = noise_embed_dim 
      self.input_proj_channels = input_proj_channels
      self.output_proj_channels = output_proj_channels
      self.padding_method = padding_method
      self.dropout_rate = dropout_rate
      self.use_spatial_attention = use_spatial_attention
      self.use_position_encoding = use_position_encoding
      self.num_heads = num_heads
      self.normalize_qk = normalize_qk
      self.device = device
      self.dtype = dtype

      self.use_spatial_attention = _maybe_broadcast_to_list(
            source=self.use_spatial_attention, reference=self.num_channels
      )

      if self.time_cond:
        self.time_embedding = FourierEmbedding(
            dims=self.noise_embed_dim,
            dtype=self.dtype,
            device=self.device
        )

      self.sigma_embedding = FourierEmbedding(
          dims=self.noise_embed_dim,
          dtype=self.dtype,
          device=self.device
      )

      self.emb_channels = self.noise_embed_dim * 2 if self.time_cond else self.noise_embed_dim

      self.DStack = DStack(
          in_channels=self.in_channels,
          spatial_resolution=self.spatial_resolution,
          emb_channels = self.emb_channels,
          num_channels=self.num_channels,
          num_res_blocks=len(self.num_channels) * (self.num_blocks,),
          downsample_ratio=self.downsample_ratio,
          use_spatial_attention=self.use_spatial_attention,
          num_input_proj_channels=self.input_proj_channels,
          padding_method=self.padding_method,
          dropout_rate=self.dropout_rate,
          num_heads=self.num_heads,
          use_position_encoding=self.use_position_encoding,
          normalize_qk=self.normalize_qk,
          dtype=self.dtype,
          device=self.device
      )

      self.UStack = UStack(
         spatial_resolution=self.spatial_resolution,
         emb_channels = self.emb_channels,
         num_channels=self.num_channels[::-1],
         num_res_blocks=len(self.num_channels) * (self.num_blocks,),
         upsample_ratio=self.downsample_ratio[::-1],
         use_spatial_attention=self.use_spatial_attention[::-1],
         num_input_proj_channels=self.input_proj_channels,
         num_output_proj_channels=self.output_proj_channels,
         padding_method=self.padding_method,
         dropout_rate=self.dropout_rate,
         num_heads=self.num_heads,
         normalize_qk=self.normalize_qk,
         use_position_encoding=self.use_position_encoding,
         dtype=self.dtype,
         device=self.device
      )

      self.norm = nn.GroupNorm(
        min(max(self.output_proj_channels // 4, 1), 32),
        self.output_proj_channels,
        device=self.device,
        dtype=self.dtype
      )

      self.conv_layer = ConvLayer(
        in_channels=self.output_proj_channels,
        out_channels=self.out_channels,
        kernel_size=self.kernel_dim * (3,),
        padding_mode=self.padding_method,
        padding=1,
        case=self.kernel_dim,
        kernel_init=default_init(),
        dtype=self.dtype,
        device=self.device
      )

    def forward(
            self,
            x: Tensor,
            sigma: Tensor,
            time: Tensor = None,
            cond: dict[str, Tensor] | None = None,
    ) -> Tensor:
      """Predicts denoised given noised input and noise level.

      Args:
        x: The model input (i.e. noised sample) with shape `(batch,
          **spatial_dims, channels)`.
        sigma: The noise level, which either shares the same batch dimension as
          `x` or is a scalar (will be broadcasted accordingly).
        cond: The conditional inputs as a dictionary. Currently, only channelwise
          conditioning is supported. Can be used for additonal conditioning
        is_training: A boolean flag that indicates whether the module runs in
          training mode.

      Returns:
        An output array with the same dimension as `x`.
      """
      if sigma.ndim < 1:
          sigma = sigma.expand(x.size(0))

      if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
          raise ValueError(
              "`sigma` must be 1D and have the same leading (batch) dimension as x"
              f" ({x.shape[0]})!"
          )

      if self.time_cond:
        if time.ndim < 1:
            time = time.expand(x.size(0))

        if time.ndim != 1 or x.shape[0] != time.shape[0]:
          raise ValueError(
              "`time` must be 1D and have the same leading (batch) dimension as x"
              f" ({x.shape[0]})!"
          )

      if not x.ndim == 5:
        raise ValueError(
            "5D inputs (batch, x,y,z, features)! x.shape:"
            f" {x.shape}"
        )

      if len(self.num_channels) != len(self.downsample_ratio):
        raise ValueError(
            f"`num_channels` {self.num_channels} and `downsample_ratio`"
            f" {self.downsample_ratio} must have the same lengths!"
        )

      # Embedding
      emb_sigma = self.sigma_embedding(sigma)
      if self.time_cond:
        emb_time = self.time_embedding(time)
        emb = torch.cat((emb_sigma, emb_time), dim=-1)
      else:
        emb = emb_sigma

      # Downsampling
      skips = self.DStack(x, emb)

      # Upsampling
      h = self.UStack(skips[-1], emb, skips)

      h = F.silu(self.norm(h))
      h = self.conv_layer(h)

      return h

class PreconditionedDenoiser3DGeoUncond(UNet3D, nn.Module):
    """Preconditioned 3-dimensional UNet denoising model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        time_cond: bool,
        num_channels: Sequence[int] = (128, 256, 256),
        downsample_ratio: Sequence[int] = (2, 2, 2),
        num_blocks: int = 4,
        noise_embed_dim: int = 128,
        input_proj_channels: int = 128,
        output_proj_channels: int = 128, 
        padding_method: str = 'circular',
        dropout_rate: float = 0.0,
        cond_dropout_prob: float = 0.0,
        use_spatial_attention: bool | Sequence[bool] = (False, False, False),
        use_position_encoding: bool = True,
        num_heads: int = 8,
        normalize_qk: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        buffer_dict: dict = None,
        sigma_data: float = 1.0,
        geo: Tensor = None
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_resolution=spatial_resolution,
            time_cond=time_cond,
            num_channels=num_channels,
            downsample_ratio=downsample_ratio,
            num_blocks=num_blocks,
            noise_embed_dim=noise_embed_dim,
            input_proj_channels=input_proj_channels,
            output_proj_channels=output_proj_channels,
            padding_method=padding_method,
            dropout_rate=dropout_rate,
            use_spatial_attention=use_spatial_attention,
            use_position_encoding=use_position_encoding,
            num_heads=num_heads,
            normalize_qk=normalize_qk,
            dtype=dtype,
            device=device,
            buffer_dict=buffer_dict
        )

        self.sigma_data = sigma_data
        self.cond_dropout_prob = cond_dropout_prob

        # preserve some references if you want
        self.num_channels = num_channels
        self.use_position_encoding = use_position_encoding
        self.normalize_qk = normalize_qk

    def forward(
        self,
        x: Tensor,            # noisy input
        sigma: Tensor,        # noise level
        y: Tensor = None,     # condition (spatial or channel-wise)
        time: Tensor = None,  # optional time conditioning
        cond: dict[str, Tensor] | None = None,
        geo: Tensor = None,
    ) -> Tensor:
        """
        Runs preconditioned denoising, substituting y with zeros if dropped.

        x: shape (batch, in_channels, D, H, W) typically
        y: shape (batch, y_channels, D, H, W), or whatever you typically cat
        """

        # Expand sigma if needed
        if sigma.ndim < 1:
            sigma = sigma.expand(x.size(0))
        if sigma.ndim != 1 or x.size(0) != sigma.shape[0]:
            raise ValueError("sigma must be 1D with the same batch dim as x")

        # Preconditioning constants
        total_var = self.sigma_data**2 + sigma**2
        c_skip = self.sigma_data**2 / total_var
        c_out = sigma * self.sigma_data / torch.sqrt(total_var)
        c_in = 1 / torch.sqrt(total_var)
        c_noise = 0.25 * torch.log(sigma)

        # Reshape for broadcasting over spatial dims
        expand_shape = [-1] + [1]*(self.kernel_dim+1)
        c_in   = c_in.view(*expand_shape)
        c_out  = c_out.view(*expand_shape)
        c_skip = c_skip.view(*expand_shape)

        # Scale the input
        inputs = c_in * x

        # Forward the base UNet3D with the partial or zeroed condition
        f_x = super().forward(
            inputs,
            sigma=c_noise,
            time=time,
            cond=None
        )

        # 6) Final scaling for preconditioned output
        return c_skip * x + c_out * f_x
