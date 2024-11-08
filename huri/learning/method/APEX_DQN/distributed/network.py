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


class Decoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Decoder, self).__init__()
        # Define the linear layers with non-linear activations
        self.linear1 = nn.Linear(input_channels, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 1024)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(1024,
                                 output_channels[0] *
                                 output_channels[1] *
                                 output_channels[2])
        self.output_channels = output_channels

    def forward(self, x):
        # Apply linear layers with activations
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = F.sigmoid(self.linear3(x))

        # Reshape to desired output shape
        x = x.view(-1, *self.output_channels)  # Reshape to (channel2, 32, 3, 6)
        return x


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
                out_channels=num_filters // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(),
            # nn.Conv2d(
            #     in_channels=num_filters // 4,
            #     # in_channels=c,  # multiple channels design
            #     out_channels=num_filters // 2,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     bias=False,
            # ),
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
        )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=in_channels,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters // 4,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(
        #         in_channels=num_filters // 4,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters // 2,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(
        #         in_channels=num_filters // 2,
        #         # in_channels=c,  # multiple channels design
        #         out_channels=num_filters,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.LeakyReLU(),
        # )

        self.advantage_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=8,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
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
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(8 * conv_out, num_fc_units),
            nn.LeakyReLU(),
            nn.Linear(num_fc_units, 1),
        )

        # self.decoder = Decoder(num_actions, output_channels=(num_filters, conv_out_hw[0], conv_out_hw[1]))

    def forward(self, x: torch.Tensor, ):
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""
        # 1. 使用mask导致无法求导？
        x_list, feasible_category_list = x

        if len(x_list) > 1:
            feature_list = [self.conv_block(_x) for _x in x_list]
            # feature_list2 = self.conv_block2(torch.stack([_x for _x in x_list], dim=0).sum(dim=0))
            # feature_list1 = self.conv_block(torch.cat([_x[:, [0]] for _x in x_list], dim=1))
            # feature_list2 = self.conv_block(torch.cat([_x[:, [1]] for _x in x_list], dim=1))
            # feature = self.conv_block(torch.cat([_x[:, [0]] for _x in x_list] + [_x[:, [1]] for _x in x_list], dim=1))
            # feature = self.conv_block2(torch.cat((feature_list1, feature_list2), dim=1))
            # feature = self.res_blocks(feature)
            value = self.value_head(torch.max(*feature_list))
            # advantage_mix = self.minus_advantage_head(feature_list2)
            advantage_list = [self.advantage_head(_f) for _f in feature_list]
            advantage_1 = [advantage_list[i] * feasible_category_list[i] for i in range(len(advantage_list))]
            feasible_category = torch.stack(feasible_category_list, dim=0).sum(dim=0)
            # advantage = torch.stack(advantage_1, dim=0).sum(dim=0) + advantage_mix * feasible_category
            # advantage = torch.stack(advantage_1, dim=0).sum(dim=0) - advantage_mix * feasible_category
            advantage = torch.stack(advantage_1, dim=0).sum(dim=0)
            advantage = advantage - advantage.sum(dim=-1, keepdim=True) / feasible_category.sum(dim=0).sum(axis=-1,
                                                                                                           keepdim=True)
            # advantage = self.advantage_head(feature)
            # advantage = advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            feature = self.conv_block(x_list[0])
            value = self.value_head(feature)
            A = self.advantage_head(feature)
            advantage = A * feasible_category_list[0]
            advantage = advantage - advantage.sum(dim=-1, keepdim=True) / feasible_category_list[0].sum(axis=-1,
                                                                                                        keepdim=True)

        return value + advantage

    def forward_features(self, x):
        x_list, feasible_category_list = x
        return self.conv_block(x_list[0])


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
