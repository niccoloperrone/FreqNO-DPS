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

"""Upsampling Stack for 3D Data Dimensions"""
import sys
import os

# Add the parent directory of 'processing' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from typing import Any, Sequence
import torch
import torch.nn as nn
from model.building_blocks.layers.upsample import ChannelToSpace
from model.building_blocks.layers.residual import CombineResidualWithSkip
from model.building_blocks.layers.convolutions import ConvLayer
from model.building_blocks.blocks.convolution_blocks import ConvBlock
from model.building_blocks.blocks.attention_block import AxialSelfAttentionBlock
from utils.model_utils import default_init

Tensor = torch.Tensor

class UStack(nn.Module):
    """Upsampling Stack.

    Takes in features at intermediate resolutions from the downsampling stack
    as well as final output, and applies upsampling with convolutional blocks
    and combines together with skip connections in typical UNet style.
    Optionally can use self attention at low spatial resolutions.

    Attributes:
        num_channels: Number of channels at each resolution level.
        num_res_blocks: Number of resnest blocks at each resolution level.
        upsample_ratio: The upsampling ration between levels.
        padding: Type of padding for the convolutional layers.
        dropout_rate: Rate for the dropout inside the transformed blocks.
        use_attention: Whether to use attention at the coarser (deepest) level.
        num_heads: Number of attentions heads inside the attention block.
        channels_per_head: Number of channels per head.
        dtype: Data type.
    """

    def __init__(
        self,
        spatial_resolution: Sequence[int],
        emb_channels: int,
        num_channels: tuple[int, ...],
        num_res_blocks: tuple[int, ...],
        upsample_ratio: tuple[int, ...],
        use_spatial_attention: Sequence[bool],
        num_input_proj_channels: int = 128,
        num_output_proj_channels: int = 128,
        padding_method: str = "circular",
        dropout_rate: float = 0.0,
        num_heads: int = 8,
        channels_per_head: int = -1,
        normalize_qk: bool = False,
        use_position_encoding: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
    ):
        super(UStack, self).__init__()

        self.kernel_dim = len(spatial_resolution)
        self.emb_channels = emb_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.upsample_ratio = upsample_ratio
        self.padding_method = padding_method
        self.dropout_rate = dropout_rate
        self.use_spatial_attention = use_spatial_attention
        self.num_input_proj_channels = num_input_proj_channels
        self.num_output_proj_channels = num_output_proj_channels
        self.num_heads = num_heads
        self.channels_per_head = channels_per_head
        self.normalize_qk = normalize_qk
        self.use_position_encoding = use_position_encoding
        self.dtype = dtype
        self.device = device

        # Calculate channels for the residual block
        in_channels = []

        # calculate list of upsample resolutions
        list_upsample_resolutions = [spatial_resolution]
        for level, channel in enumerate(self.num_channels):
            downsampled_resolution = tuple(
                [
                    int(res / self.upsample_ratio[level])
                    for res in list_upsample_resolutions[-1]
                ]
            )
            list_upsample_resolutions.append(downsampled_resolution)
        list_upsample_resolutions = list_upsample_resolutions[::-1]
        list_upsample_resolutions.pop()

        self.residual_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()  # ConvBlock
        self.attention_blocks = nn.ModuleList()  # AxialSelfAttentionBlock
        self.conv_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()  # ChannelToSpace

        for level, channel in enumerate(self.num_channels):
            self.conv_blocks.append(nn.ModuleList())
            self.attention_blocks.append(nn.ModuleList())
            self.residual_blocks.append(nn.ModuleList())

            for block_id in range(self.num_res_blocks[level]):
                if block_id == 0 and level > 0:
                    in_channels.append(self.num_channels[level - 1])
                else:
                    in_channels.append(channel)

                # Residual Block
                self.residual_blocks[level].append(
                    CombineResidualWithSkip(
                        residual_channels=in_channels[-1],
                        skip_channels=channel,
                        kernel_dim=self.kernel_dim,
                        project_skip=in_channels[-1] != channel,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
                # Convolution Block
                self.conv_blocks[level].append(
                    ConvBlock(
                        in_channels=in_channels[-1],
                        out_channels=channel,
                        emb_channels=self.emb_channels,
                        kernel_size=self.kernel_dim * (3,),
                        padding_mode=self.padding_method,
                        padding=1,
                        case=self.kernel_dim,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
                # Attention Block
                if self.use_spatial_attention[level]:
                    # attention requires input shape: (bs, x, y, z, c)
                    attn_axes = [1, 2, 3]  # attention along all spatial dimensions

                    self.attention_blocks[level].append(
                        AxialSelfAttentionBlock(
                            in_channels=channel,
                            spatial_resolution=list_upsample_resolutions[level],
                            attention_axes=attn_axes,
                            add_position_embedding=self.use_position_encoding,
                            num_heads=self.num_heads,
                            normalize_qk=self.normalize_qk,
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )

            # Upsampling step
            up_ratio = self.upsample_ratio[level]
            self.conv_layers.append(
                ConvLayer(
                    in_channels=channel,
                    out_channels=up_ratio**self.kernel_dim * channel,
                    kernel_size=self.kernel_dim * (3,),
                    padding_mode=self.padding_method,
                    padding=1,
                    case=self.kernel_dim,
                    kernel_init=default_init(1.0),
                    dtype=self.dtype,
                    device=self.device,
                )
            )

            self.upsample_layers.append(
                ChannelToSpace(
                    block_shape=self.kernel_dim * (up_ratio,),
                    in_channels=up_ratio**self.kernel_dim * channel,
                    kernel_dim=self.kernel_dim,
                    spatial_resolution=list_upsample_resolutions[level],
                )
            )

        # DStack Input - UStack Output Residual Connection
        self.res_skip_layer = CombineResidualWithSkip(
            residual_channels=self.num_channels[-1],
            skip_channels=self.num_input_proj_channels,
            kernel_dim=self.kernel_dim,
            project_skip=(self.num_channels[-1] != self.num_input_proj_channels),
            dtype=self.dtype,
            device=self.device,
        )

        # Add Output Layer
        self.conv_layers.append(
            ConvLayer(
                in_channels=self.num_channels[-1],
                out_channels=self.num_output_proj_channels,
                kernel_size=self.kernel_dim * (3,),
                padding_mode=self.padding_method,
                padding=1,
                case=self.kernel_dim,
                kernel_init=default_init(1.0),
                dtype=self.dtype,
                device=self.device,
            )
        )

    def forward(self, x: Tensor, emb: Tensor, skips: list[Tensor]) -> Tensor:
        assert x.ndim == 5
        assert x.shape[0] == emb.shape[0]
        assert len(self.num_channels) == len(self.num_res_blocks)
        assert len(self.upsample_ratio) == len(self.num_res_blocks)

        h = x

        for level, channel in enumerate(self.num_channels):
            for block_id in range(self.num_res_blocks[level]):
                # Residual
                h = self.residual_blocks[level][block_id](residual=h, skip=skips.pop())
                # Convolution Blocks
                h = self.conv_blocks[level][block_id](h, emb)
                # Spatial Attention Blocks
                if self.use_spatial_attention[level]:
                    h = self.attention_blocks[level][block_id](h)

            # Upsampling Block
            h = self.conv_layers[level](h)
            # Shift channels to increase the resolution, similar to torch.nn.PixelShift
            h = self.upsample_layers[level](h)

        # Output - Input Residual Connection
        h = self.res_skip_layer(residual=h, skip=skips.pop())
        # Output Layer
        h = self.conv_layers[-1](h)

        return h
    
import torch
import torch.nn as nn
from typing import List, Tuple
from model.building_blocks.stacks.dstack_3d import DStack

def test_stack_shape_flow():
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
    
    # Common configuration
    num_channels = [64, 128, 256, 512]  # Number of channels at each level
    num_res_blocks = [2, 2, 2, 2]  # Number of residual blocks at each level
    downsample_ratio = [2, 2, 2, 2]  # Downsampling ratio at each level
    use_spatial_attention = [False, True, True, True]  # Whether to use spatial attention at each level
    num_input_proj_channels = 128
    num_output_proj_channels = 128
    
    print("=== DStack Configuration ===")
    # Initialize DStack
    dstack = DStack(
        in_channels=in_channels,
        spatial_resolution=spatial_dims,
        emb_channels=emb_channels,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        downsample_ratio=downsample_ratio,
        use_spatial_attention=use_spatial_attention,
        num_input_proj_channels=num_input_proj_channels,
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
    
    # Forward pass through DStack
    print("\nRunning DStack forward pass...")
    skips = dstack(x, emb)
    
    # Print shapes of skip connections
    print(f"Input shape: {x.shape}")
    print(f"Number of skip connections: {len(skips)}")
    
    for i, skip in enumerate(skips):
        print(f"Skip {i} shape: {skip.shape}")
    
    # Print the last skip connection shape (will be the input to UStack)
    last_skip = skips[-1]
    print(f"\nLast skip (input to UStack) shape: {last_skip.shape}")
    
    print("\n=== UStack Configuration ===")
    # UStack needs the reverse of DStack's parameters
    ustack_num_channels = num_channels[::-1]  # Reverse the channels
    ustack_num_res_blocks = num_res_blocks[::-1]  # Reverse the res blocks
    ustack_ratio = downsample_ratio[::-1]  # Reverse the ratios
    ustack_attention = use_spatial_attention[::-1]  # Reverse the attention flags
    
    # Initialize UStack
    ustack = UStack(
        spatial_resolution=spatial_dims,
        emb_channels=emb_channels,
        num_channels=ustack_num_channels,
        num_res_blocks=ustack_num_res_blocks,
        upsample_ratio=ustack_ratio,
        use_spatial_attention=ustack_attention,
        num_input_proj_channels=num_input_proj_channels,
        num_output_proj_channels=num_output_proj_channels,
        padding_method='circular',
        dropout_rate=0.0,
        num_heads=8,
        channels_per_head=-1,
        normalize_qk=False,
        use_position_encoding=False,
        dtype=torch.float32,
        device=device
    )
    
    # Move the model to the same device as the inputs if needed
    ustack = ustack.to(device)
    
    # Create a copy of skips before passing to UStack (since UStack modifies the list)
    skips_copy = skips.copy()
    
    # Forward pass through UStack
    print("\nRunning UStack forward pass...")
    output = ustack(last_skip, emb, skips_copy)
    
    # Print output shape
    print(f"UStack output shape: {output.shape}")
    
    # Calculate the expected shape: should match input shape but with different channel count
    expected_channel_count = num_output_proj_channels
    expected_output_shape = (batch_size, expected_channel_count, *spatial_dims)
    print(f"Expected output shape: {expected_output_shape}")
    
    # Check if output shape matches expected
    shape_match = (output.shape == expected_output_shape)
    print(f"Output shape matches expected: {shape_match}")
    
    # Return memory to GPU if using CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return output

# Run the test
if __name__ == "__main__":
    output = test_stack_shape_flow()