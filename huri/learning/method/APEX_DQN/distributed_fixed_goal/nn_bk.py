"""

Author: Hao Chen (chen960216@gmail.com)
Created: 20230719osaka

"""
import time

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
            # nn.BatchNorm2d(num_features=num_filters),
            # nn.ELU(),
            nn.LeakyReLU(),
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
        out = F.leaky_relu(out)
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

        in_channels = c * num_category
        # First convolutional block
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
                out_channels=8,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=8),
            # nn.Mish(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * conv_out, num_actions),  # num_actions
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=8,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=8),
            # nn.Mish(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * conv_out, num_fc_units),
            # nn.Mish(),
            nn.ReLU(),
            nn.Linear(num_fc_units, 1),
            # nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""
        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)
        advantage = self.advantage_head(features)
        value = self.value_head(features)
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

        in_channels = 4
        # First convolutional block
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
            # nn.ReLU(),
            nn.LeakyReLU(),
            # nn.ELU(),
        )

        # self.conv_block_1 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=num_filters,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     # nn.BatchNorm2d(num_features=num_filters),
        #     # nn.ReLU(),
        #     # nn.LeakyReLU(),
        #     nn.ELU(),
        # )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=in_channels,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     # nn.BatchNorm2d(num_features=num_filters),
        #     # nn.ReLU(),
        #     # nn.LeakyReLU(),
        #     nn.ELU(),
        # )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_blocks.append(ResNetBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        # # Residual blocks
        # res_blocks2 = []
        # for _ in range(num_res_block):
        #     res_blocks2.append(ResNetBlock(num_filters))
        # self.res_blocks2 = nn.Sequential(*res_blocks2)

        self.advantage_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=8,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=8),
            # nn.Mish(),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(8 * conv_out, num_actions),  # num_actions
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=8,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=8),
            # nn.Mish(),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(8 * conv_out, num_fc_units),
            # nn.Mish(),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.Linear(num_fc_units, 1),
            # nn.Tanh(),
        )

        # self.feature_category_encode = nn.Sequential(
        #     nn.Linear(153, num_fc_units),
        #     nn.LeakyReLU(),
        #     nn.Linear(num_fc_units, conv_out),
        #     nn.LeakyReLU()
        # )

        # self.gate_control = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(conv_out, 2)
        # )
        # self.num_filters = num_filters
        # self.conv_out_hw = conv_out_hw
        # self.h, self.w = h, w

    def forward(self, x: torch.Tensor, toggle_debug=False):
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""
        x_1, x_2, (feasible_category_1, feasible_category_2) = x
        # a = time.time()
        feature_1 = self.res_blocks(self.conv_block(x_1))
        feature_2 = self.res_blocks(self.conv_block(x_2))
        # b = time.time()
        # # Process the controlling input z
        # gating_distribution = F.softmax(self.gate_control(z), dim=1)
        # if toggle_debug:
        #     print(gating_distribution)
        # # Reshape as needed
        # gate_expanded = gating_distribution.view(-1, 2, 1, 1, 1).expand(-1, -1, 10, self.h, self.w)
        #
        # # Apply the gating mechanism
        # features = gate_expanded[:, 0] * feature_1 + gate_expanded[:, 1] * feature_2

        # encoded_feasible_category_1 = self.feature_category_encode(feasible_category_1)
        # encoded_feasible_category_2 = self.feature_category_encode(feasible_category_2)
        # encoded_feasible_category_1 = encoded_feasible_category_1.view(-1, 1, *self.conv_out_hw)
        # encoded_feasible_category_2 = encoded_feasible_category_2.view(-1, 1, *self.conv_out_hw)
        # feature_1 = torch.cat((feature_1, encoded_feasible_category_1), dim=1)
        # feature_2 = torch.cat((feature_2, encoded_feasible_category_2), dim=1)

        # value = self.value_head(torch.cat((feature_1, feature_2, feature_3), dim=1))
        value = self.value_head(feature_1 + feature_2)
        # value = self.value_head(torch.max(feature_1, feature_2))
        advantage1 = feasible_category_1 * self.advantage_head(feature_1)
        advantage2 = feasible_category_2 * self.advantage_head(feature_2)
        advantage = advantage1 + advantage2
        # adjust_advantage = advantage - advantage.sum(-1, keepdim=True) / torch.sum(
        #     feasible_category_1 + feasible_category_2)

        if toggle_debug:
            print("time consumption is ", b - a)
            print(value)
            print("advantage1", advantage1)
            print("advantage2", advantage2)
            print("advantage", advantage)
            print("advantage mean", advantage.mean(dim=-1, keepdim=True))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
        # return value + adjust_advantage


class ICM(nn.Module):
    def __init__(self, input_shape: Tuple,
                 num_actions: int,
                 num_res_block: int = 19,
                 num_filters: int = 256,
                 num_fc_units: int = 256, ):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(ICM, self).__init__()
        c, h, w = input_shape
        conv_out_hw = calc_conv2d_output((h, w), 3, 1, 1)
        # FIX BUG, Python 3.7 has no math.prod()
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        # First convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
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
        self.conv_feature_block = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, 512),
        )

        self.pred_module = nn.Sequential(
            nn.Linear(512 + num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.invpred_module = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def get_feature(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)
        features = self.conv_feature_block(x)
        return features

    def forward(self, x):
        # get feature
        feature_x = self.get_feature(x)
        return feature_x

    def get_full(self, x, x_next, a_vec):
        # get feature
        feature_x = self.get_feature(x)
        feature_x_next = self.get_feature(x_next)
        pred_s_next = self.pred(feature_x, a_vec)  # predict next state feature
        pred_a_vec = self.invpred(feature_x, feature_x_next)  # (inverse) predict action

        return pred_s_next, pred_a_vec, feature_x_next

    def pred(self, feature_x, a_vec):
        # Forward prediction: predict next state feature, given current state feature and action (one-hot)
        pred_s_next = self.pred_module(torch.cat([feature_x, a_vec.float()], dim=-1).detach())
        return pred_s_next

    def invpred(self, feature_x, feature_x_next):
        # Inverse prediction: predict action (one-hot), given current and next state features
        pred_a_vec = self.invpred_module(torch.cat([feature_x, feature_x_next], dim=-1))
        return F.softmax(pred_a_vec, dim=-1)


class RNDModel(nn.Module):
    def __init__(self, input_shape: Tuple, num_category: int):
        super(RNDModel, self).__init__()

        c, h, w = input_shape
        in_channels = c * num_category
        # in_channels = c
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(576, 128)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


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

        # in_channels = 4
        # # # First convolutional block
        in_channels = 2

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                # in_channels=c,  # multiple channels design
                out_channels=num_filters // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=num_filters // 2,
                # in_channels=c,  # multiple channels design
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=num_filters,
                # in_channels=c,  # multiple channels design
                out_channels=num_filters * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                # in_channels=c,  # multiple channels design
                out_channels=num_filters // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=num_filters // 2,
                # in_channels=c,  # multiple channels design
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=num_filters,
                # in_channels=c,  # multiple channels design
                out_channels=num_filters * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(),
        )

        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=in_channels,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters // 2,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.ReLU(),
        # )
        #
        #
        #
        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=num_filters,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(
        #         in_channels=num_filters,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters * 2,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.LeakyReLU(),
        # )

        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=in_channels,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters // 2,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.LeakyReLU(),
        # )
        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=in_channels,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.ReLU(),
        #     # nn.Conv2d(
        #     #     in_channels=num_filters,
        #     #     # in_channels=c,  # multiple channels design
        #     #     out_channels=2 * num_filters,
        #     #     kernel_size=5,
        #     #     stride=1,
        #     #     padding=2,
        #     #     bias=False,
        #     # ),
        #     # nn.ReLU(),
        # )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=in_channels,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters // 2,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.LeakyReLU(),
        # )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=in_channels,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.ReLU(),
        #     # nn.Conv2d(
        #     #     in_channels=num_filters,
        #     #     # in_channels=c,  # multiple channels design
        #     #     out_channels=2 * num_filters,
        #     #     kernel_size=3,
        #     #     stride=1,
        #     #     padding=1,
        #     #     bias=False,
        #     # ),
        #     # nn.ReLU(),
        # )

        # Residual blocks
        # res_blocks2 = []
        # for _ in range(10):
        #     res_blocks2.append(ResNetBlock(num_filters // 2))
        # self.res_blocks2 = nn.Sequential(*res_blocks2)

        # Residual blocks
        # res_blocks = []
        # for _ in range(num_res_block):
        #     res_blocks.append(ResNetBlock(num_filters))
        # self.res_blocks = nn.Sequential(*res_blocks)

        self.advantage_head = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=num_filters,
            #     out_channels=8,
            #     kernel_size=1,
            #     stride=1,
            #     bias=False,
            # ),
            # nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(num_filters * 2 * conv_out, num_filters * conv_out),  # num_actions
            nn.LeakyReLU(),
            nn.Linear(num_filters * conv_out, num_actions),
        )

        self.minus_advantage_head = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=num_filters,
            #     out_channels=8,
            #     kernel_size=1,
            #     stride=1,
            #     bias=False,
            # ),
            # nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(num_filters * 2 * conv_out, num_filters * conv_out),  # num_actions
            nn.LeakyReLU(),
            nn.Linear(num_filters * conv_out, num_actions),
        )

        self.value_head = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=num_filters,
            #     out_channels=8,
            #     kernel_size=1,
            #     stride=1,
            #     bias=False,
            # ),
            # nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(num_filters * 2 * conv_out, num_fc_units),
            nn.LeakyReLU(),
            nn.Linear(num_fc_units, 1),
        )

    # def forward(self, x: torch.Tensor, ):
    #     """Given raw state x, predict the raw logits probability distribution for all actions,
    #     and the evaluated value, all from current player's perspective."""
    #     x_1, x_2, (feasible_category_1, feasible_category_2) = x
    #     feature_1 = self.res_blocks(self.conv_block(x_1))
    #     feature_2 = self.res_blocks(self.conv_block(x_2))
    #     advantage1 = feasible_category_1 * self.advantage_head(feature_1)
    #     advantage2 = feasible_category_2 * self.advantage_head(feature_2)
    #     value = self.value_head(self.conv_block_value_head(torch.cat((x_1[:, :2], x_2[:, :2], x_1[:, 2:4]), axis=1)))
    #     advantage = advantage1 + advantage2
    #     return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor, ):
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""
        # 1. 使用mask导致无法求导？
        x_list, feasible_category_list = x
        # x_1, x_3, (feasible_category_1, feasible_category_2) = x
        # x_1, x_2, (feasible_category_1, feasible_category_2) = x
        # feature_1 = self.conv_block(x_1)
        # feature_2 = self.conv_block(x_2)
        # feature_3 = self.conv_block2(x_3)
        # self.res_blocks2(
        # feature_3 = self.conv_block2(x_3)
        # value_head_feature = self.conv_block_value_head(torch.cat((feature_1, feature_2, feature_3), dim=1))
        # feature_1 = self.res_blocks(torch.cat([feature_1, feature_3], dim=1))
        # feature_2 = self.res_blocks(torch.cat([feature_2, feature_3], dim=1))
        # feature_1 = torch.cat([feature_1, feature_3], dim=1)
        # feature_2 = torch.cat([feature_2, feature_3], dim=1)
        # feature_1 = self.res_blocks(feature_1)
        # feature_2 = self.res_blocks(feature_2)
        # value = self.value_head(feature_3)
        if len(x_list) > 1:
            feature_list = [self.conv_block(_x) for _x in x_list]
            feature_list2 = self.conv_block2(torch.stack([_x for _x in x_list], dim=0).sum(dim=0))
            # feature_list1 = self.conv_block(torch.cat([_x[:, [0]] for _x in x_list], dim=1))
            # feature_list2 = self.conv_block(torch.cat([_x[:, [1]] for _x in x_list], dim=1))
            # feature = self.conv_block(torch.cat([_x[:, [0]] for _x in x_list] + [_x[:, [1]] for _x in x_list], dim=1))
            # feature = self.conv_block2(torch.cat((feature_list1, feature_list2), dim=1))
            # feature = self.res_blocks(feature)
            value = self.value_head(torch.max(*feature_list))
            advantage_mix = self.minus_advantage_head(feature_list2)
            advantage_list = [self.advantage_head(_f) for _f in feature_list]
            advantage_1 = [advantage_list[i] * feasible_category_list[i] for i in range(len(advantage_list))]
            feasible_category = torch.stack(feasible_category_list, dim=0).sum(dim=0)
            # advantage = torch.stack(advantage_1, dim=0).sum(dim=0) + advantage_mix * feasible_category
            advantage = torch.stack(advantage_1, dim=0).sum(dim=0) - advantage_mix * feasible_category
            # advantage = advantage - advantage.sum(dim=-1, keepdim=True) / feasible_category.sum(dim=0).sum(axis=-1,
            #                                                                                                keepdim=True)
            # advantage = self.advantage_head(feature)
            # advantage = advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            feature = self.conv_block(x_list[0])
            value = self.value_head(feature)
            advantage = self.advantage_head(feature)
            advantage = advantage - advantage.mean(dim=-1, keepdim=True)

        # value2 = self.value_head(feature_2)
        # value = torch.max(value1, value2)
        # value = self.value_head(value_head_feature)
        # advantage1 = self.advantage_head(feature_1)
        # advantage2 = self.advantage_head(feature_2)
        # # advantage2 = feasible_category_2 * self.advantage_head(feature_2)
        # # advantage = advantage1 + advantage2
        # feasible_category = feasible_category_1 + feasible_category_2
        # advantage = (feasible_category_1 * advantage1 + feasible_category_2 * advantage2) + \
        #             (1 - feasible_category) * (advantage1 + advantage2) / 2
        # advantage = self.advantage_head(feature_1)
        return value + advantage
        # return advantage


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
