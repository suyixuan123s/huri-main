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
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out


class DuelingDQNCNN3(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNCNN3, self).__init__()

        self.num_classes = num_classes

        self.conv = ConvBlock(in_ch=2, out_ch=64, bias=True)
        # for _ in range(1, self.num_classes + 1):
        #     setattr(self, f"encoder_{_}_{0}", ConvBlock(in_ch=16, out_ch=32, bias=True))
        #     setattr(self, f"encoder_{_}_{1}", ConvBlock(in_ch=32, out_ch=64, bias=True))
        #     setattr(self, f"encoder_{_}_{2}", ConvBlock(in_ch=64, out_ch=64, bias=True))
        for _ in range(1, self.num_classes + 1):
            setattr(self, f"encoder_{_}_{0}", ResBlock(in_ch=64, out_ch=64))
            setattr(self, f"encoder_{_}_{1}", ResBlock(in_ch=64, out_ch=64))
            setattr(self, f"encoder_{_}_{2}", ResBlock(in_ch=64, out_ch=64))
        # for block in range(3):
        #     setattr(self, "res_%i" % block, ResBlock(in_ch=64, out_ch=64))

        # setattr(self, f"encoder_{0}", ConvBlock(in_ch=16, out_ch=32, bias=True))
        # setattr(self, f"encoder_{1}", ConvBlock(in_ch=32, out_ch=64, bias=True))
        # setattr(self, f"encoder_{2}", ConvBlock(in_ch=64, out_ch=64, bias=True))

        # self.fc1 = nn.Linear(1600, 1200)
        # self.fc2 = nn.Linear(1200, 800)

        self.value_layer = nn.Sequential(
            nn.Linear(1600, 800),
            nn.ReLU(),
            nn.Linear(800, 1),
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(1600, 800),
            nn.ReLU(),
            nn.Linear(800, action_dim)
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        # state = state.unsqueeze(1)
        # goal = goal.unsqueeze(1)
        # x = torch.cat((state, goal), axis=1)

        ft_list = []
        # s_v = torch.zeros_like(state, dtype=state.dtype, device=state.device)
        # g_v = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
        # s_v[state > 0] = 1
        # g_v[goal > 0] = 1
        for _ in range(1, self.num_classes + 1):
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1
            x = torch.cat((s.unsqueeze(1), g.unsqueeze(1)), axis=1)
            x = self.conv(x)
            for block in range(3):
                x = getattr(self, f"encoder_{_}_{block}")(x)
            x = torch.flatten(F.relu(x), 1).unsqueeze(1)
            ft_list.append(x)
        x = torch.cat(ft_list, axis=1)
        x = torch.max(x, 1, keepdim=True)[0].reshape(x.shape[0], -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))

        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def forward2(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        # state = state.unsqueeze(1)
        # goal = goal.unsqueeze(1)
        # x = torch.cat((state, goal), axis=1)

        ft_list = []
        # s_v = torch.zeros_like(state, dtype=state.dtype, device=state.device)
        # g_v = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
        # s_v[state > 0] = 1
        # g_v[goal > 0] = 1
        for _ in range(1, self.num_classes + 1):
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1
            x = torch.cat((s.unsqueeze(1), g.unsqueeze(1)), axis=1)
            x = self.conv(x)
            for block in range(3):
                x = getattr(self, f"encoder_{_}_{block}")(x)
            x = torch.flatten(F.relu(x), 1).unsqueeze(1)
            ft_list.append(x)
        x = torch.cat(ft_list, axis=1)
        x = torch.max(x, 1, keepdim=True)[0].reshape(x.shape[0], -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))

        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DuelingDQNCNN2(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNCNN2, self).__init__()

        self.num_classes = num_classes

        self.conv = ConvBlock(in_ch=2 * self.num_classes, out_ch=32)
        for block in range(3):
            setattr(self, "res_%i" % block, ResBlock(in_ch=32, out_ch=32))

        self.value_layer = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, action_dim)
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        # state = state.unsqueeze(1)
        # goal = goal.unsqueeze(1)
        # x = torch.cat((state, goal), axis=1)

        ft_list = []
        for _ in range(1, self.num_classes + 1):
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1
            ft_list.append(torch.cat((s.unsqueeze(1), g.unsqueeze(1)), axis=1))
        x = torch.cat(ft_list, axis=1)

        x = self.conv(x)
        for block in range(3):
            x = getattr(self, "res_%i" % block)(x)
        x = torch.flatten(F.relu(x), 1)
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def forward2(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        # state = state.unsqueeze(1)
        # goal = goal.unsqueeze(1)
        # x = torch.cat((state, goal), axis=1)

        ft_list = []
        for _ in reversed(range(1, self.num_classes + 1)):
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1
            ft_list.append(torch.cat((s.unsqueeze(1), g.unsqueeze(1)), axis=1))
        x = torch.cat(ft_list, axis=1)

        x = self.conv(x)
        for block in range(3):
            x = getattr(self, "res_%i" % block)(x)
        x = torch.flatten(F.relu(x), 1)
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


# dueling DQN
class DuelingDQNCNN(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNCNN, self).__init__()

        self.num_classes = num_classes

        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels=2 * self.num_classes, out_channels=32, kernel_size=3, padding=1, stride=1),
            # nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        def conv2d_size_out(size, kernel_size, padding, stride=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        w1 = conv2d_size_out(3, 3, 1)
        h1 = conv2d_size_out(3, 3, 1)
        w2 = conv2d_size_out(w1, 3, 1)
        h2 = conv2d_size_out(h1, 3, 1)
        w3 = conv2d_size_out(w2, 1, 0)
        h3 = conv2d_size_out(h2, 1, 0)

        self.value_layer = nn.Sequential(
            nn.Linear(64 * w3 * h3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(64 * w3 * h3, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        # state = state.unsqueeze(1)
        # goal = goal.unsqueeze(1)
        # x = torch.cat((state, goal), axis=1)

        ft_list = []
        for _ in range(1, self.num_classes + 1):
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1
            ft_list.append(torch.cat((s.unsqueeze(1), g.unsqueeze(1)), axis=1))
        x = torch.cat(ft_list, axis=1)

        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


if __name__ == "__main__":
    rack_size = (3, 3)


    def conv2d_size_out(size, kernel_size, padding, stride=1):
        return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1


    w1 = conv2d_size_out(5, 3, 1)
    h1 = conv2d_size_out(10, 3, 1)
    w2 = conv2d_size_out(w1, 0, 1)
    h2 = conv2d_size_out(h1, 0, 1)

    dqn = DuelingDQNCNN3(obs_dim=rack_size, action_dim=np.prod(rack_size) ** 2, num_classes=2)
    goal = torch.tensor([[1, 1, 2, 2, 2],
                         [1, 1, 2, 2, 2],
                         [1, 1, 2, 2, 2],
                         [1, 1, 2, 2, 2],
                         [1, 1, 2, 2, 2], ], dtype=torch.float32)
    goal = torch.tile(goal, dims=[32, 1, 1])
    state = torch.tensor([[1, 1, 0, 1, 0, ],
                          [0, 0, 2, 1, 0, ],
                          [0, 1, 2, 1, 0, ],
                          [0, 1, 2, 1, 0, ],
                          [0, 1, 2, 1, 0, ], ], dtype=torch.float32)

    state = torch.tile(state, dims=[32, 1, 1])
    print(dqn(state, goal) - dqn.forward(state, goal))
