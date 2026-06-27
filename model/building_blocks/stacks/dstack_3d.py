# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich.
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

"""Downsampling Stack for 3D Data Dimensions"""
import sys
import os

# Add the parent directory of 'processing' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import torch
import torch.nn as nn
from typing import Any, Sequence

from model.building_blocks.layers.convolutions import ConvLayer, DownsampleConv
from model.building_blocks.blocks.convolution_blocks import ConvBlock
from model.building_blocks.blocks.attention_block import AxialSelfAttentionBlock
from utils.model_utils import default_init

Tensor = torch.Tensor

class DStack(nn.Module):
  """Downsampling stack.

  Repeated convolutional blocks with occasional strides for downsampling.
  Features at different resolutions are concatenated into output to use
  for skip connections by the UStack module.
  """

  def __init__(
      self, 
      in_channels: int,
      spatial_resolution: Sequence[int],
      emb_channels: int,
      num_channels: Sequence[int], 
      num_res_blocks: Sequence[int],
      downsample_ratio: Sequence[int], 
      use_spatial_attention: Sequence[bool],
      num_input_proj_channels: int = 128,
      padding_method: str='circular', # LATLON
      dropout_rate: float=0.0, 
      num_heads: int=8,
      channels_per_head: int=-1, 
      use_position_encoding: bool=False,
      normalize_qk: bool=False, 
      dtype: torch.dtype=torch.float32,
      device: Any | None = None
    ):
    super(DStack, self).__init__()

    self.in_channels =  in_channels
    self.kernel_dim = len(spatial_resolution) # number of dimensions
    self.emb_channels = emb_channels
    self.num_channels = num_channels
    self.num_res_blocks = num_res_blocks
    self.downsample_ratio = downsample_ratio
    self.padding_method = padding_method
    self.dropout_rate = dropout_rate
    self.use_spatial_attention = use_spatial_attention
    self.num_input_proj_channels = num_input_proj_channels
    self.num_heads = num_heads
    self.channels_per_head = channels_per_head
    self.use_position_encoding = use_position_encoding
    self.normalize_qk = normalize_qk
    self.dtype = dtype
    self.device = device

    # ConvLayer
    self.conv_layer = ConvLayer(
      in_channels=self.in_channels,
      out_channels=self.num_input_proj_channels,
      kernel_size=self.kernel_dim * (3,),
      padding_mode=self.padding_method,
      padding=1,
      case=self.kernel_dim,
      kernel_init=default_init(1.0),
      dtype=self.dtype,
      device=self.device,
    )

    # Input channels for the downsampling layer
    dsample_in_channels = [self.num_input_proj_channels, *self.num_channels[:-1]]
    list_spatial_resolution = [spatial_resolution]

    self.dsample_layers = nn.ModuleList() # DownsampleConv layer
    self.conv_blocks = nn.ModuleList() # ConvBlock
    self.attention_blocks = nn.ModuleList() # AxialSelfAttentionBlock

    for level, channel in enumerate(self.num_channels):

      # Compute resolution after downsampling:
      downsampled_resolution = tuple([int(res / self.downsample_ratio[level]) for res in list_spatial_resolution[-1]])
      list_spatial_resolution.append(downsampled_resolution)

      # Downsample Layers
      self.dsample_layers.append(
        DownsampleConv(
          in_channels=dsample_in_channels[level],
          out_channels=channel,
          spatial_resolution=spatial_resolution,
          ratios=(self.downsample_ratio[level],) * self.kernel_dim,
          kernel_init=default_init(1.0),
          case=self.kernel_dim,
          device=self.device,
          dtype=self.dtype,
        )
      )
      self.conv_blocks.append(nn.ModuleList())
      self.attention_blocks.append(nn.ModuleList())

      for block_id in range(self.num_res_blocks[level]):
        # Convblocks
        self.conv_blocks[level].append(
          ConvBlock(
            in_channels=channel,
            out_channels=channel,
            emb_channels=self.emb_channels,
            kernel_size=self.kernel_dim * (3,),
            padding_mode=self.padding_method,
            padding=1,
            case=self.kernel_dim,
            dropout=self.dropout_rate,
            dtype=self.dtype,
            device=self.device,
          )
        )

        if self.use_spatial_attention[level]:
            # attention requires input shape: (bs, x, y, z, c)
            attn_axes = [1, 2, 3] # attention along all spatial dimensions
            
            self.attention_blocks[level].append(
                AxialSelfAttentionBlock(
                    in_channels=channel,
                    spatial_resolution=list_spatial_resolution[-1],
                    attention_axes=attn_axes,
                    add_position_embedding=self.use_position_encoding,
                    num_heads=self.num_heads,
                    dtype=self.dtype,
                    device=self.device
                )
            )

        if block_id != 0:
          list_spatial_resolution.append(downsampled_resolution)

  def forward(self, x: Tensor, emb: Tensor) -> list[Tensor]:
    assert (x.ndim == 5)
    assert x.shape[0] == emb.shape[0]
    assert len(self.num_channels) == len(self.num_res_blocks)
    assert len(self.downsample_ratio) == len(self.num_res_blocks)

    skips = []
    h = self.conv_layer(x)
    skips.append(h)
    #print("WE ARE IN DSTACK")
    #print(f'INITIAL SHAPE: {h.shape}')

    for level, channel in enumerate(self.num_channels):
      #print(f'H SHAPE BEFORE DSAMPLE: {h.shape}')
      h = self.dsample_layers[level](h)
      #print(f'H SHAPE AFTER DSAMPLE: {h.shape}')

      for block_id in range(self.num_res_blocks[level]):
        h = self.conv_blocks[level][block_id](h, emb)
        #print(f'CURRENT SHAPE: {h.shape}')
        if self.use_spatial_attention[level]: 
          h = self.attention_blocks[level][block_id](h)
          #print(f'CURRENT SHAPE (ATTENTION): {h.shape}')
        skips.append(h)
    return skips


import torch
import torch.nn as nn
from typing import List, Tuple

def test_dstack_shape_flow():
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Input dimensions
    batch_size = 5
    in_channels = 3
    spatial_dims = (32, 32, 320)  # (H, W, D)
    
    # Create test input tensor - ensure it's on the correct device
    x = torch.randn(batch_size, in_channels, *spatial_dims, device=device)
    
    # Create embedding tensor (typically time embeddings) - ensure it's on the correct device
    emb_channels = 256
    emb = torch.randn(batch_size, emb_channels, device=device)
    
    # DStack configuration
    num_channels = [64, 128, 256, 512]  # Number of channels at each level
    num_res_blocks = [2, 2, 2, 2]  # Number of residual blocks at each level
    downsample_ratio = [2, 2, 2, 2]  # Downsampling ratio at each level
    use_spatial_attention = [False, True, True, True]  # Whether to use spatial attention at each level
    
    # Initialize DStack
    dstack = DStack(
        in_channels=in_channels,
        spatial_resolution=spatial_dims,
        emb_channels=emb_channels,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        downsample_ratio=downsample_ratio,
        use_spatial_attention=use_spatial_attention,
        num_input_proj_channels=128,
        padding_method='circular',
        dropout_rate=0.0,
        num_heads=8,
        channels_per_head=-1,
        use_position_encoding=False,
        normalize_qk=False,
        dtype=torch.float32,
        device=device
    )
    
    # Move the model to the same device as the inputs if needed
    dstack = dstack.to(device)
    
    # Forward pass
    skips = dstack(x, emb)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Number of skip connections: {len(skips)}")
    
    for i, skip in enumerate(skips):
        print(f"Skip {i} shape: {skip.shape}")
    
    # Compute and print the receptive field at each level
    current_resolution = spatial_dims
    print(f"\nResolution at each level:")
    print(f"Initial resolution: {current_resolution}")
    
    for level, ratio in enumerate(downsample_ratio):
        current_resolution = tuple(dim // ratio for dim in current_resolution)
        print(f"Level {level + 1} resolution: {current_resolution}")
    
    # Return memory to GPU if using CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return skips

# Run the test
if __name__ == "__main__":
    skips = test_dstack_shape_flow()