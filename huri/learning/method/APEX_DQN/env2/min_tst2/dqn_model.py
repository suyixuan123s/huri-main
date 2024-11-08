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

        for _ in range(1, self.num_classes + 1):
            setattr(self, f'state_linear_n{_}', nn.Sequential(nn.Linear(np.prod(obs_dim), 64), nn.ReLU()))
            setattr(self, f'goal_linear_n{_}', nn.Sequential(nn.Linear(np.prod(obs_dim), 64), nn.ReLU()))

        self.feedfoward_linear = nn.Sequential(nn.Linear(128 * self.num_classes, 128 * self.num_classes), nn.ReLU())

        self.value_layer = nn.Sequential(
            nn.Linear(128 * self.num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(128 * self.num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def feedforward_layer(self, x):
        ft = self.feedfoward_linear(x)
        x = x + ft
        x = F.relu(x)
        return x

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        ft_list = []
        for _ in range(1, self.num_classes + 1):
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1

            s_ft = getattr(self, f'state_linear_n{_}')(s.reshape(s.shape[0], -1))
            g_ft = getattr(self, f'goal_linear_n{_}')(g.reshape(g.shape[0], -1))

            ft_list.append(torch.cat((s_ft, g_ft), axis=1))

        feature = self.feedforward_layer(torch.cat(ft_list, axis=1))
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DuelingDQN(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQN, self).__init__()

        self.num_classes: int = num_classes
        self.obs_dim_prod: int = np.prod(obs_dim)

        for _ in range(1, self.num_classes + 1):
            setattr(self, f'state_linear_n{_}', self.gen_conv_feature_extractor())
            setattr(self, f'goal_linear_n{_}', self.gen_conv_feature_extractor())

        self.feedfoward_linear = nn.Sequential(nn.Linear(3200 * self.num_classes, 3200 * self.num_classes), nn.ReLU())

        self.value_layer = nn.Sequential(
            nn.Linear(3200 * self.num_classes, 1600),
            nn.ReLU(),
            nn.Linear(1600, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(3200 * self.num_classes, 1600),
            nn.ReLU(),
            nn.Linear(1600, action_dim)
        )

    def gen_conv_feature_extractor(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def feedforward_layer(self, x):
        ft = self.feedfoward_linear(x)
        x = x + ft
        x = F.relu(x)
        return x

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        ft_list = []
        for _ in range(1, self.num_classes + 1):
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1

            s_ft = getattr(self, f'state_linear_n{_}')(s.unsqueeze(1))
            g_ft = getattr(self, f'goal_linear_n{_}')(g.unsqueeze(1))

            ft_list.append(torch.cat((s_ft, g_ft), axis=1))

        feature = self.feedforward_layer(torch.cat(ft_list, axis=1))
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class AttentionEncoder(nn.Module):
    def __init__(self, embed_dim: int, feedforward_dim: int, n_head: int = 1):
        super(AttentionEncoder, self).__init__()
        self.attention_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_head, batch_first=True)
        self.ff_linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.ff_linear2 = nn.Sequential(
            nn.Linear(feedforward_dim, embed_dim), nn.ReLU())
        self.atten_output_weights = None

    def _ff_block(self, x: torch.Tensor):
        # embeddingwise MLP
        ft = self.ff_linear1(x)
        return self.ff_linear2(F.relu(ft))

    def _sa_block(self, x: torch.Tensor, ):
        q_ft = k_ft = v_ft = x
        atten_ft, attn_output_weights = self.attention_layer(q_ft, k_ft, v_ft)
        self.atten_output_weights = attn_output_weights
        return atten_ft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._sa_block(x) + x
        x = self._ff_block(x) + x
        return x


class DuelingDQNAttentionMini(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNAttentionMini, self).__init__()

        self.num_classes: int = num_classes
        self.obs_dim_prod: int = np.prod(obs_dim)

        fiter_len = 3
        embed_len = self.num_classes
        # self.conv_feature = nn.Conv2d(in_channels=1, out_channels=fiter_len, kernel_size=3, padding=1)
        # self.channelwise_pooling = torch.nn.MaxPool3d((fiter_len, 1, 1), stride=1, padding=0)

        # self.transformer_layer_en_0 = nn.TransformerEncoderLayer(d_model=embed_len,
        #                                                          nhead=1,
        #                                                          dim_feedforward=self.num_classes * 2 * self.obs_dim_prod,
        #                                                          batch_first=True, )
        self.self_atten = AttentionEncoder(embed_dim=embed_len, n_head=1,
                                           feedforward_dim=self.num_classes * 2 * self.obs_dim_prod)

        self.positonal_encoding = PositionalEncoding1D(embed_len)

        self.value_layer = nn.Sequential(
            nn.Linear(self.num_classes * 2 * self.obs_dim_prod, 250),
            nn.ReLU(),
            nn.Linear(250, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(self.num_classes * 2 * self.obs_dim_prod, 250),
            nn.ReLU(),
            nn.Linear(250, action_dim)
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor,  ) -> torch.Tensor:
        state = torch.nn.functional.one_hot(state.long().reshape(state.shape[0], -1), self.num_classes + 1)[...,
                1:].float()
        goal = torch.nn.functional.one_hot(goal.long().reshape(goal.shape[0], -1), self.num_classes + 1)[...,
               1:].float()
        # state = rearrange(self.conv_feature(state.unsqueeze(1)), 'b c h w -> b (h w) c')
        # goal = self.conv_feature(goal.unsqueeze(1)).permute(0, 2, 3, 1)

        pe = self.positonal_encoding(torch.zeros_like(state))
        pstate = state + pe
        pgoal = goal + pe
        sg_ft = torch.cat((pstate, pgoal), axis=1)
        feature = self.self_atten(sg_ft)
        feature = rearrange(feature, 'b l c -> b (l c)')

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
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

    dqn = DuelingDQNAttentionMini(obs_dim=rack_size, action_dim=np.prod(rack_size) ** 2, num_classes=3)
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
