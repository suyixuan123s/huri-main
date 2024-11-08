import numpy as np
import torch.nn as nn
import torch


# dueling DQN
class DuelingDQN(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int):
        """Initialization."""
        super(DuelingDQN, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels=obs_dim[0], out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        def conv2d_size_out(size, kernel_size, padding, stride=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        w1 = conv2d_size_out(obs_dim[1], 3, 1)
        h1 = conv2d_size_out(obs_dim[2], 3, 1)
        w2 = conv2d_size_out(w1, 3, 1)
        h2 = conv2d_size_out(h1, 3, 1)
        w3 = conv2d_size_out(w2, 3, 1)
        h3 = conv2d_size_out(h2, 3, 1)
        w4 = conv2d_size_out(w3, 3, 1)
        h4 = conv2d_size_out(h3, 3, 1)
        w5 = conv2d_size_out(w4, 3, 1)
        h5 = conv2d_size_out(h4, 3, 1)

        self.value_layer = nn.Sequential(
            nn.Linear(w5 * h5 * 128 + 3 * obs_dim[1] * obs_dim[2], 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(w5 * h5 * 128 + 3 * obs_dim[1] * obs_dim[2], 4096),
            nn.ReLU(),
            nn.Linear(4096, action_dim)
        )

    def forward(self, abs_state: torch.Tensor, ) -> torch.Tensor:
        goal = abs_state[:, :, :5]
        state = abs_state[:, :, 5:10]
        fill_c = abs_state[:, :, 10:15]
        mov_c = abs_state[:, :, 15:20]

        goal_c_ft = goal.view(abs_state.shape[0], -1)
        fill_c_ft = fill_c.view(abs_state.shape[0], -1)
        mov_c_ft = mov_c.view(abs_state.shape[0], -1)
        """Forward method implementation."""
        feature = self.feature_layer(state)
        feature = torch.cat((feature, goal_c_ft, fill_c_ft, mov_c_ft), axis=1)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q
