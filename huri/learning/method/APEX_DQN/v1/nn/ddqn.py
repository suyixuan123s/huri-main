import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D
from einops import rearrange


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sz=3, stride=1, padding=1, bias=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_sz, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch=128, out_ch=128, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=True)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out


class DuelingDQNCNN4Sub(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNCNN4Sub, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_classes * 2, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(6400, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Linear(4096, action_dim),
        )

        self.num_classes = num_classes

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
        # s[state > 0] = 1
        # o_ft = self.state_overall_layer(s.reshape(state.shape[0], -1))
        # ft_list = [s]
        ft_list = []
        for _ in range(1, self.num_classes + 1):
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1
            ft = torch.cat((s.unsqueeze(1), g.unsqueeze(1)), axis=1)
            # s_ft = getattr(self, f"state_linear_{int(_)}")((torch.cat((s, g), axis=1).reshape(state.shape[0], -1)))
            # g_ft = getattr(self, f"goal_linear_{int(_)}")(g.reshape(goal.shape[0], -1))
            ft_list.append(ft)
        x = torch.cat(ft_list, axis=1)

        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DuelingDQNCNN4(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNCNN4, self).__init__()

        self.num_classes = num_classes
        self.net1 = DuelingDQNCNN4Sub(obs_dim, action_dim, num_classes)
        self.net2 = DuelingDQNCNN4Sub(obs_dim, action_dim, num_classes)

    def forward(self, state: torch.Tensor, goal: torch.Tensor,
                constraints: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.net1(state, goal)

    def Q(self, state: torch.Tensor, goal: torch.Tensor,
          constraints: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        q1 = self.net1(state, goal)
        q2 = self.net2(goal, state)
        return q1, q2


class DuelingDQNCNN3(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNCNN3, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_classes * 2, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(6400, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Linear(4096, action_dim),
        )

        self.num_classes = num_classes

    def forward(self, state: torch.Tensor, goal: torch.Tensor, *args) -> torch.Tensor:
        s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
        # s[state > 0] = 1
        # o_ft = self.state_overall_layer(s.reshape(state.shape[0], -1))
        # ft_list = [s]
        ft_list = []
        for _ in range(1, self.num_classes + 1):
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1
            ft = torch.cat((s.unsqueeze(1), g.unsqueeze(1)), axis=1)
            # s_ft = getattr(self, f"state_linear_{int(_)}")((torch.cat((s, g), axis=1).reshape(state.shape[0], -1)))
            # g_ft = getattr(self, f"goal_linear_{int(_)}")(g.reshape(goal.shape[0], -1))
            ft_list.append(ft)
        x = torch.cat(ft_list, axis=1)

        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


# class DuelingDQNCNN2(nn.Module):
#     def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
#         """Initialization."""
#         super(DuelingDQNCNN2, self).__init__()
#
#         self.feature_layer = nn.Sequential(
#             nn.Conv2d(in_channels=num_classes * 2, out_channels=16, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#
#         self.value_layer = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1)
#         )
#
#         self.advantage_layer = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, action_dim),
#         )
#
#         self.num_classes = num_classes
#
#     def forward(self, state: torch.Tensor, goal: torch.Tensor, *args) -> torch.Tensor:
#         s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
#         # s[state > 0] = 1
#         # o_ft = self.state_overall_layer(s.reshape(state.shape[0], -1))
#         # ft_list = [s]
#         ft_list = []
#         for _ in range(1, self.num_classes + 1):
#             s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
#             g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
#             s[state == _] = 1
#             g[goal == _] = 1
#             ft = torch.cat((s.unsqueeze(1), g.unsqueeze(1)), axis=1)
#             # s_ft = getattr(self, f"state_linear_{int(_)}")((torch.cat((s, g), axis=1).reshape(state.shape[0], -1)))
#             # g_ft = getattr(self, f"goal_linear_{int(_)}")(g.reshape(goal.shape[0], -1))
#             ft_list.append(ft)
#         x = torch.cat(ft_list, axis=1)
#
#         feature = self.feature_layer(x)
#         value = self.value_layer(feature)
#         advantage = self.advantage_layer(feature)
#         q = value + advantage - advantage.mean(dim=-1, keepdim=True)
#         return q

class DuelingDQNCNN2(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNCNN2, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_classes * 2, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        self.num_classes = num_classes

    def forward(self, state: torch.Tensor, goal: torch.Tensor, *args) -> torch.Tensor:
        s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
        # s[state > 0] = 1
        # o_ft = self.state_overall_layer(s.reshape(state.shape[0], -1))
        # ft_list = [s]
        ft_list = []
        for _ in range(1, self.num_classes + 1):
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1
            ft = torch.cat((s.unsqueeze(1), g.unsqueeze(1)), axis=1)
            # s_ft = getattr(self, f"state_linear_{int(_)}")((torch.cat((s, g), axis=1).reshape(state.shape[0], -1)))
            # g_ft = getattr(self, f"goal_linear_{int(_)}")(g.reshape(goal.shape[0], -1))
            ft_list.append(ft)
        x = torch.cat(ft_list, axis=1)

        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DuelingDQNTransformer(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=64,
                                                   dim_feedforward=128,
                                                   nhead=1,
                                                   batch_first=True, )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.input_layer = nn.Linear(16, 64, bias=False)

        self.value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        self.num_classes = num_classes

    def forward(self, state: torch.Tensor, goal: torch.Tensor, *args) -> torch.Tensor:
        x = torch.cat((state.unsqueeze(1), goal.unsqueeze(1)), axis=1).flatten(start_dim=2)
        x = F.relu(self.input_layer(x))
        x = self.transformer_encoder(x)
        x = F.relu(x.flatten(start_dim=1))
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class SelectionNet(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(SelectionNet, self).__init__()


"""AlphaZero Neural Network component."""
import math
from typing import Tuple


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
            nn.BatchNorm2d(num_features=num_filters),
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
            nn.BatchNorm2d(num_features=num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class DDQN(nn.Module):
    """Policy network for AlphaZero agent."""

    def __init__(
            self,
            input_shape: Tuple,
            num_actions: int,
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

        self.advantage_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, num_actions),
        )

        self.value_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * conv_out, num_fc_units),
            nn.ReLU(),
            nn.Linear(num_fc_units, 1),
            # nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor, *args):
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""
        x = torch.cat((state.unsqueeze(1), goal.unsqueeze(1)), axis=1)
        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)
        value = self.value_layer(features)
        advantage = self.advantage_layer(features)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


if __name__ == "__main__":
    net = DuelingDQNTransformer(obs_dim=(4, 4), action_dim=120, num_classes=1)
    src = torch.rand(10, 5, 10)
    goal = torch.rand(10, 5, 10)
    out = net(src, goal)
