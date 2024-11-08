""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230719osaka

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import NamedTuple, Tuple


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
            # nn.BatchNorm2d(num_features=num_filters),
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
            # nn.BatchNorm2d(num_features=num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class HeadNet(nn.Module):
    def __init__(self, num_filters, conv_out, num_actions):
        super(HeadNet, self).__init__()
        self.advantage_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=2),
            # nn.Mish(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, num_actions),
        )

    def forward(self, x, value):
        advantage = self.advantage_head(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DDQN(nn.Module):
    def __init__(
            self,
            input_shape: Tuple,
            num_actions: int,
            num_head: int,
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
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_blocks.append(ResNetBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.head_net_list = nn.ModuleList(
            [HeadNet(num_filters, conv_out, num_actions) for _ in range(num_head)])

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=1),
            # nn.Mish(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * conv_out, num_fc_units),
            # nn.Mish(),
            nn.ReLU(),
            nn.Linear(num_fc_units, 1),
            # nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, k: int or tuple or list = None):
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""
        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)
        value = self.value_head(features)
        if k is not None:
            if isinstance(k, int):
                return self.head_net_list[k](features, value)
            else:
                return [net(features, value) for i, net in enumerate(self.head_net_list) if i in k]
        else:
            return [net(features, value)[:, None, :] for net in self.head_net_list]


class DDQN_pp(nn.Module):
    def __init__(
            self,
            conv_out: int,
            num_actions: int,
            num_filters: int = 256,
    ) -> None:
        super().__init__()
        self.advantage_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=2),
            # nn.Mish(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, num_actions),
        )

    def forward(self, features: torch.Tensor, value: torch.Tensor):
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DDQN2(nn.Module):
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
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_blocks.append(ResNetBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=1),
            # nn.Mish(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * conv_out, num_fc_units),
            # nn.Mish(),
            nn.ReLU(),
            nn.Linear(num_fc_units, 1),
            # nn.Tanh(),
        )
        self.net_pick = DDQN_pp(conv_out, num_actions, num_filters)
        self.net_place = DDQN_pp(conv_out, num_actions, num_filters)

    def forward(self, x):
        """Given raw state x, predict the raw logits probability distribution for all actions,
                and the evaluated value, all from current player's perspective."""
        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)
        value = self.value_head(features)
        return self.net_pick(features, value), self.net_place(features, value)

    # def forward(self, x: torch.Tensor, k: int or tuple or list = None):
    #     """Given raw state x, predict the raw logits probability distribution for all actions,
    #     and the evaluated value, all from current player's perspective."""
    #     conv_block_out = self.conv_block(x)
    #     features = self.res_blocks(conv_block_out)
    #     value = self.value_head(features)
    #     if k is not None:
    #         if isinstance(k, int):
    #             return self.head_net_list[k](features, value)
    #         else:
    #             return [net(features, value) for i, net in enumerate(self.head_net_list) if i in k]
    #     else:
    #         return [net(features, value)[:, None, :] for net in self.head_net_list]


if __name__ == '__main__':
    pass
