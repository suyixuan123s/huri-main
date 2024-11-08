""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230719osaka

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import NamedTuple, Tuple
import numpy as np


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class ResNetBlock(nn.Module):
    """Basic redisual block."""

    def __init__(
            self,
            num_filters: int,
    ) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class DDQN(nn.Module):
    def __init__(
            self,
            input_shape: Tuple,
            num_actions: int,
            num_category: int,
            num_res_block: int = 19,
            num_filters: int = 256,
            num_fc_units: int = 256,
    ) -> None:
        super().__init__()
        c, h, w = input_shape
        conv_out_hw = calc_conv2d_output((h, w), 3, 1, 1)
        # FIX BUG, Python 3.7 has no math.prod()
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        # First convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c * num_category,
                out_channels=num_filters,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_blocks.append(ResNetBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.advantage_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(1 * conv_out, num_fc_units),
            nn.Mish(),
            nn.Linear(num_fc_units, 1),
        )

    def forward(self, x: torch.Tensor):
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""
        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)
        advantage = self.advantage_head(features)
        value = self.value_head(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


if __name__ == '__main__':
    from huri.learning.env.rack_v3 import create_env
    import numpy as np

    env = create_env(rack_sz=(3, 6),
                     num_tube_class=2,
                     seed=888,
                     toggle_curriculum=False,
                     toggle_goal_fixed=False,
                     num_history=1)
    input_shape = env.observation_space_dim
    num_actions = env.action_space_dim
    state = env.reset()
    goal = env.goal_pattern

    network = DDQN(input_shape,
                   num_actions,
                   num_category=2,
                   num_filters=32,
                   num_res_block=10,
                   num_fc_units=128)
    dqn_action_value = network(
        torch.as_tensor(
            np.concatenate((state.abs_state(env.num_classes), env.goal_pattern.abs_state(env.num_classes)), axis=0),
            dtype=torch.float32, ).unsqueeze(0))

    rnd_model = RNDModel(input_shape, env.num_classes)
    rnd_model(torch.as_tensor(
        np.concatenate((state.abs_state(env.num_classes), env.goal_pattern.abs_state(env.num_classes)), axis=0),
        dtype=torch.float32, ).unsqueeze(0))
