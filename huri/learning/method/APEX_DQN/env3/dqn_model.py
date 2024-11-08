import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D
from einops import rearrange


class DuelingDQNMini(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNMini, self).__init__()

        self.num_classes = num_classes

        self.embed_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU())

        self.linear1 = nn.Linear(256, 256)

        self.value_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def _ff_block(self, x):
        return F.relu(self.linear1(x)) + x

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # state = torch.nn.functional.one_hot(state.long().reshape(state.shape[0], -1), self.num_classes + 1)[...,
        #         1:].float()
        # goal = torch.nn.functional.one_hot(goal.long().reshape(goal.shape[0], -1), self.num_classes + 1)[...,
        #        1:].float()

        _state = state.reshape(state.shape[0], -1)
        _goal = goal.reshape(goal.shape[0], -1)

        s_ft = self.embed_linear(_state)
        g_ft = self.embed_linear(_goal)

        sg_ft = torch.cat((s_ft, g_ft), axis=1)

        feature = self._ff_block(sg_ft)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q



class DuelingDQN(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQN, self).__init__()

        self.num_classes = num_classes

        self.embed_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 256), nn.ReLU(),)

        self.linear1 = nn.Linear(512, 512)

        # self.value_layer = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # )

        self.advantage_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim)
        )

    def _ff_block(self, x):
        return F.relu(self.linear1(x)) + x

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # state = torch.nn.functional.one_hot(state.long().reshape(state.shape[0], -1), self.num_classes + 1)[...,
        #         1:].float()
        # goal = torch.nn.functional.one_hot(goal.long().reshape(goal.shape[0], -1), self.num_classes + 1)[...,
        #        1:].float()

        _state = state.reshape(state.shape[0], -1)
        _goal = goal.reshape(goal.shape[0], -1)

        s_ft = self.embed_linear(_state)
        g_ft = self.embed_linear(_goal)

        sg_ft = torch.cat((s_ft, g_ft), axis=1)

        feature = self._ff_block(sg_ft)
        # value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        # q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        q = advantage
        return q

if __name__ == "__main__":
    rack_size = (5, 10)


    def conv2d_size_out(size, kernel_size, padding, stride=1):
        return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1


    w1 = conv2d_size_out(5, 3, 1)
    h1 = conv2d_size_out(10, 3, 1)
    w2 = conv2d_size_out(w1, 0, 1)
    h2 = conv2d_size_out(h1, 0, 1)

    print(w2, h2)

    dqn = DuelingDQNMini(obs_dim=rack_size, action_dim=np.prod(rack_size) ** 2, num_classes=3)
    goal = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]], dtype=torch.float32)
    goal = torch.tile(goal, dims=[32, 1, 1])
    state = torch.tensor([[2, 3, 1, 0, 2, 2, 0, 1, 3, 0],
                          [1, 0, 0, 0, 0, 1, 3, 3, 0, 0],
                          [2, 2, 3, 0, 1, 0, 0, 2, 3, 0],
                          [0, 1, 0, 2, 0, 0, 2, 0, 2, 3],
                          [3, 3, 2, 0, 3, 3, 0, 1, 3, 1]], dtype=torch.float32)

    state = torch.tile(state, dims=[32, 1, 1])
    dqn.forward(state, goal)
