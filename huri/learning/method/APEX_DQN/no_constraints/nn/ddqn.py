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


if __name__ == "__main__":
    net = DuelingDQNTransformer(obs_dim=(4, 4), action_dim=120, num_classes=1)
    src = torch.rand(10, 5, 10)
    goal = torch.rand(10, 5, 10)
    out = net(src, goal)
