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
                out_channels=num_filters // 4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=num_filters // 4,
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
        )

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
            A = self.advantage_head(feature)
            advantage = A * feasible_category_list[0] + A * (1 - feasible_category_list[0])
            advantage = advantage - advantage.mean(dim=-1, keepdim=True)

        return value + advantage

    def forward_features(self, x):
        x_list, feasible_category_list = x
        return self.conv_block(x_list[0])
