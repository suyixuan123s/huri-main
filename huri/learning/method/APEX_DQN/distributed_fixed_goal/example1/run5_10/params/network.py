import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import NamedTuple, Tuple
class DDQN2(nn.Module):
    @staticmethod
    def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        """takes a tuple of (h,w) and returns a tuple of (h,w)"""

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
        w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
        return h, w

    def __init__(
            self,
            input_shape: Tuple,
            num_actions: int,
            num_category: int,
            num_res_block: int = 19,
            num_filters: int = 256,
            num_fc_units: int = 256,
            num_out_cnn_layers: int = 8
    ) -> None:
        super().__init__()
        c, h, w = input_shape
        conv_out_hw = self.calc_conv2d_output((h, w), 3, 1, 1)
        # FIX BUG, Python 3.7 has no math.prod()
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        # in_channels = 4
        # # # First convolutional block
        in_channels = 2

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                # in_channels=c,  # multiple channels design
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=num_filters),
            nn.Mish(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_blocks.append(ResNetBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.advantage_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_out_cnn_layers,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=8),
            # nn.Mish(),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(num_out_cnn_layers * conv_out, num_actions),  # num_actions
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_out_cnn_layers,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=8),
            # nn.Mish(),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(num_out_cnn_layers * conv_out, num_fc_units),
            # nn.Mish(),
            nn.Mish(),
            nn.Linear(num_fc_units, 1),
            # nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, ):
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""
        # 1. 使用mask导致无法求导？
        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)
        advantage = self.advantage_head(features)
        value = self.value_head(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
