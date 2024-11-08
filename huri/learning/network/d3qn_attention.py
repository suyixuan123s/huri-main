import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


# dueling DQN
class DuelingDQN(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int):
        """Initialization."""
        super(DuelingDQN, self).__init__()

        self.state_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 256), nn.ReLU())
        self.goal_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 256), nn.ReLU())

        # self.query_linear_s = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        # self.query_linear_g = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        # self.key_linear_s =nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        # self.key_linear_g = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        # self.value_linear_s = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        # self.value_linear_g = nn.Sequential(nn.Linear(128, 128), nn.ReLU())

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)

        self.feedfoward_linear = nn.Sequential(nn.Linear(512, 512), nn.ReLU())

        self.value_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, action_dim)
        )

        self.relu = nn.ReLU()

    def feedforward_layer(self, x):
        ft = self.feedfoward_linear(x)
        x = x + ft
        x = self.relu(x)
        return x

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        s_ft = self.state_linear(state.reshape(state.shape[0], -1))
        g_ft = self.goal_linear(goal.reshape(state.shape[0], -1))
        # feature = torch.cat((s_ft, g_ft), axis=1)
        # q_ft = torch.cat((self.query_linear_s(s_ft), self.query_linear_g(g_ft)), axis=1)
        # k_ft = torch.cat((self.key_linear_s(s_ft), self.key_linear_g(g_ft)), axis=1)
        # v_ft = torch.cat((self.value_linear_s(s_ft), self.value_linear_g(g_ft)), axis=1)
        sg_ft = torch.cat((s_ft, g_ft), axis=1)
        q_ft = k_ft = v_ft = sg_ft
        atten_ft, attn_output_weights = self.attention(q_ft, k_ft, v_ft)
        feature = atten_ft.reshape(atten_ft.shape[0], -1) + sg_ft.reshape(sg_ft.shape[0], -1)

        feature = self.feedforward_layer(feature)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    # def forward_value(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    #     s_ft = self.state_linear(state.reshape(state.shape[0], -1))
    #     g_ft = self.goal_linear(goal.reshape(state.shape[0], -1))
    #     feature = torch.cat((s_ft, g_ft), axis=1)
    #     # q_ft = torch.cat((self.query_linear_s(s_ft), self.query_linear_g(g_ft)), axis=1)
    #     # k_ft = torch.cat((self.key_linear_s(s_ft), self.key_linear_g(g_ft)), axis=1)
    #     # v_ft = torch.cat((self.value_linear_s(s_ft), self.value_linear_g(g_ft)), axis=1)
    #     # q_ft = k_ft = v_ft = torch.cat((s_ft, g_ft), axis=1)
    #     # feature, attn_output_weights = self.attention(q_ft, k_ft, v_ft)
    #     feature = feature.reshape(feature.shape[0], -1)
    #     feature = self.feedforward_layer(feature)
    #     value = self.value_layer(feature)
    #     return value


class DuelingDQNL(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNL, self).__init__()

        self.query_state_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU())
        self.query_goal_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU())

        self.key_state_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU())
        self.key_goal_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU())

        self.value_state_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU())
        self.value_goal_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU())

        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

        self.feedfoward_linear = nn.Sequential(nn.Linear(256, 256), nn.ReLU())

        self.value_layer = nn.Sequential(
            nn.Linear(256, 1),
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)

        )

        self.num_classes = num_classes

    def feedforward_layer(self, x):
        ft = self.feedfoward_linear(x)
        x = x + ft
        x = F.relu(x)
        return x

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        q_s_ft = self.query_state_linear(state.reshape(state.shape[0], -1))
        q_g_ft = self.query_goal_linear(goal.reshape(state.shape[0], -1))

        k_s_ft = self.key_state_linear(state.reshape(state.shape[0], -1))
        k_g_ft = self.key_goal_linear(goal.reshape(state.shape[0], -1))

        v_s_ft = self.value_state_linear(state.reshape(state.shape[0], -1))
        v_g_ft = self.value_goal_linear(goal.reshape(state.shape[0], -1))

        q_ft = torch.cat((q_s_ft, q_g_ft), axis=1)
        k_ft = torch.cat((k_s_ft, k_g_ft), axis=1)
        v_ft = torch.cat((v_s_ft, v_g_ft), axis=1)

        feature, attn_output_weights = self.attention(q_ft, k_ft, v_ft)
        feature = feature.reshape(feature.shape[0], -1)

        feature = self.feedforward_layer(feature)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DuelingDQNM(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNM, self).__init__()

        self.state_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU())
        self.goal_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU())

        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

        self.feedfoward_linear = nn.Sequential(nn.Linear(256, 256), nn.ReLU())

        self.value_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.num_classes = num_classes

    def feedforward_layer(self, x):
        ft = self.feedfoward_linear(x)
        x = x + ft
        x = F.relu(x)
        return x

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        s_ft = self.state_linear(state.reshape(state.shape[0], -1))
        g_ft = self.goal_linear(goal.reshape(state.shape[0], -1))

        q_ft = k_ft = v_ft = torch.cat((s_ft, g_ft), axis=1)
        feature, attn_output_weights = self.attention(q_ft, k_ft, v_ft)
        feature = feature.reshape(feature.shape[0], -1)

        feature = self.feedforward_layer(feature)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DuelingDQNMini(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNMini, self).__init__()

        self.state_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU())
        self.goal_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU())

        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

        self.feedfoward_linear = nn.Sequential(nn.Linear(256, 256), nn.ReLU())

        self.value_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.relu = nn.ReLU()

        self.num_classes = num_classes

    def feedforward_layer(self, x):
        ft = self.feedfoward_linear(x)
        x = x + ft
        x = F.relu(x)
        return x

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        s_ft = self.state_linear(state.reshape(state.shape[0], -1))
        g_ft = self.goal_linear(goal.reshape(state.shape[0], -1))

        q_ft = k_ft = v_ft = torch.cat((s_ft, g_ft), axis=1)
        feature, attn_output_weights = self.attention(q_ft, k_ft, v_ft)
        feature = feature.reshape(feature.shape[0], -1)

        feature = self.feedforward_layer(feature)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


# dueling DQN
class DuelingDQN2(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQN2, self).__init__()
        self.num_classes = torch.arange(1, num_classes + 1, dtype=torch.float32)

        self.state_overall_layer = nn.Sequential(nn.Linear(np.prod(obs_dim), 256), nn.ReLU(), nn.Linear(256, 256))

        self.classify_layer = nn.Sequential(nn.Linear(np.prod(obs_dim) * (num_classes * 2 + 1), 256),
                                            nn.ReLU(),
                                            nn.Linear(256, num_classes))

        for _ in self.num_classes:
            setattr(self, f"state_linear_{int(_)}",
                    nn.Sequential(nn.Linear(np.prod(obs_dim) * 2, 256), nn.ReLU(), nn.Linear(256, 256)))

            # setattr(self, f"goal_linear_{int(_)}",
            #         nn.Sequential(nn.Linear(np.prod(obs_dim), 128), nn.ReLU(), nn.Linear(128, 128)))
            # setattr(self, f"attention_{int(_)}",
            #         nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True))
        # self.attention = nn.MultiheadAttention(embed_dim=256 * (len(self.num_classes) ),
        #                                        num_heads=1 + len(self.num_classes), batch_first=True)
        # self.query_linear_s = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        # self.query_linear_g = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        # self.key_linear_s =nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        # self.key_linear_g = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        # self.value_linear_s = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        # self.value_linear_g = nn.Sequential(nn.Linear(128, 128), nn.ReLU())

        fl_dim = 100

        self.feedfoward_linear1 = nn.Sequential(
            nn.Linear(fl_dim, fl_dim), nn.ReLU())

        self.feedfoward_linear2 = nn.Sequential(
            nn.Linear(fl_dim, fl_dim), nn.ReLU())

        self.value_layer = nn.Sequential(
            nn.Linear(fl_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(fl_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, action_dim)
        )

        self.relu = nn.ReLU()

    def feedforward_layer1(self, x):
        ft = self.feedfoward_linear(x)
        x = x + ft
        x = self.relu(x)
        return x

    def feedforward_layer2(self, x):
        ft = self.feedfoward_linear2(x)
        x = x + ft
        x = self.relu(x)
        return x

    def forward(self, state: torch.Tensor, goal: torch.Tensor, ) -> torch.Tensor:
        s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
        s[state > 0] = 1
        # o_ft = self.state_overall_layer(s.reshape(state.shape[0], -1))
        ft_list = [s]
        for _ in self.num_classes:
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1
            ft = torch.cat((s, g), axis=1)
            # s_ft = getattr(self, f"state_linear_{int(_)}")((torch.cat((s, g), axis=1).reshape(state.shape[0], -1)))
            # g_ft = getattr(self, f"goal_linear_{int(_)}")(g.reshape(goal.shape[0], -1))
            ft_list.append(ft)
        feature = torch.cat(ft_list, axis=1)

        classify_ft = torch.nn.functional.softmax(self.classify_layer(feature.reshape(state.shape[0], -1)), dim=1)

        ft1 = self.feedfoward_linear1(ft_list[1].reshape(state.shape[0], -1))

        ft2 = self.feedfoward_linear2(ft_list[2].reshape(state.shape[0], -1))

        # ft = torch.cat((ft1.unsqueeze(1), ft2.unsqueeze(1)), axis=1)
        # ft.gather(1, classify_ft.argmax(dim=1, keepdims=True))
        classify_ft_argmax = classify_ft.argmax(dim=1, keepdims=True)
        feature = ft1 * (classify_ft_argmax == 0) + ft2 * (classify_ft_argmax == 1)
        # q_ft = k_ft = v_ft = torch.cat(ft_list, axis=1)
        # feature, attn_output_weights = self.attention(q_ft, k_ft, v_ft)
        feature = feature.reshape(feature.shape[0], -1)

        # feature = self.feedforward_layer(feature)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def forward_value(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
        s[state > 0] = 1
        # o_ft = self.state_overall_layer(s.reshape(state.shape[0], -1))
        ft_list = [s]
        for _ in self.num_classes:
            s = torch.zeros_like(state, dtype=state.dtype, device=state.device)
            g = torch.zeros_like(goal, dtype=goal.dtype, device=goal.device)
            s[state == _] = 1
            g[goal == _] = 1
            ft = torch.cat((s, g), axis=1)
            # s_ft = getattr(self, f"state_linear_{int(_)}")((torch.cat((s, g), axis=1).reshape(state.shape[0], -1)))
            # g_ft = getattr(self, f"goal_linear_{int(_)}")(g.reshape(goal.shape[0], -1))
            ft_list.append(ft)
        feature = torch.cat(ft_list, axis=1)

        classify_ft = torch.nn.functional.softmax(self.classify_layer(feature.reshape(state.shape[0], -1)), dim=1)

        ft1 = self.feedfoward_linear1(ft_list[1].reshape(state.shape[0], -1))

        ft2 = self.feedfoward_linear2(ft_list[2].reshape(state.shape[0], -1))

        # ft = torch.cat((ft1.unsqueeze(1), ft2.unsqueeze(1)), axis=1)
        # ft.gather(1, classify_ft.argmax(dim=1, keepdims=True))
        classify_ft_argmax = classify_ft.argmax(dim=1, keepdims=True)
        feature = ft1 * (classify_ft_argmax == 0) + ft2 * (classify_ft_argmax == 1)
        # q_ft = k_ft = v_ft = torch.cat(ft_list, axis=1)
        # feature, attn_output_weights = self.attention(q_ft, k_ft, v_ft)
        feature = feature.reshape(feature.shape[0], -1)

        # feature = self.feedforward_layer(feature)
        value = self.value_layer(feature)
        return value


class DuelingDQN3(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQN3, self).__init__()

        self.state_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 256), nn.ReLU())
        self.goal_linear = nn.Sequential(nn.Linear(np.prod(obs_dim), 256), nn.ReLU())

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)

        self.feedfoward_linear = nn.Sequential(nn.Linear(256, 512), nn.ReLU())

        self.value_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, action_dim)
        )

        self.relu = nn.ReLU()

        self.num_classes = num_classes + 1

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        s_onehot = nn.functional.one_hot(state.type(torch.int64), num_classes=self.num_classes).type(state.dtype)
        g_onehot = nn.functional.one_hot(goal.type(torch.int64), num_classes=self.num_classes).type(goal.dtype)
        s_ft = self.state_linear(state.reshape(s_onehot.shape[0], -1))
        # g_ft = self.goal_linear(goal.reshape(g_onehot.shape[0], -1))
        # feature = torch.cat((s_ft, g_ft), axis=1)
        feature = s_ft
        # q_ft = torch.cat((self.query_linear_s(s_ft), self.query_linear_g(g_ft)), axis=1)
        # k_ft = torch.cat((self.key_linear_s(s_ft), self.key_linear_g(g_ft)), axis=1)
        # v_ft = torch.cat((self.value_linear_s(s_ft), self.value_linear_g(g_ft)), axis=1)

        # q_ft = k_ft = v_ft = torch.cat((s_ft, g_ft), axis=1)
        # feature, attn_output_weights = self.attention(q_ft, k_ft, v_ft)
        feature = feature.reshape(feature.shape[0], -1)

        feature = self.feedfoward_linear(feature)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DuelingDQN4(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQN4, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_classes * 2, out_channels=32, kernel_size=3, padding=1, stride=1),
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

        w1 = conv2d_size_out(obs_dim[0], 3, 1)
        h1 = conv2d_size_out(obs_dim[1], 3, 1)
        w2 = conv2d_size_out(w1, 3, 1)
        h2 = conv2d_size_out(h1, 3, 1)
        w3 = conv2d_size_out(w2, 3, 1)
        h3 = conv2d_size_out(h2, 3, 1)
        w4 = conv2d_size_out(w3, 3, 1)
        h4 = conv2d_size_out(h3, 3, 1)
        w5 = conv2d_size_out(w4, 3, 1)
        h5 = conv2d_size_out(h4, 3, 1)

        self.value_layer = nn.Sequential(
            nn.Linear(w5 * h5 * 128, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(w5 * h5 * 128, 4096),
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


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["output_feature"]
        )

    def forward(self, state, goal):
        features = torch.cat(
            (state.reshape(state.shape[0], -1), goal.reshape(goal.shape[0], -1)), axis=1)
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        return code


class DuelingDQN5(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQN5, self).__init__()

        self.encoder_hidden_layer = nn.Linear(
            in_features=2 * np.prod(obs_dim), out_features=512
        )
        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=256
        )
        self.decoder_output_layer = nn.Linear(
            in_features=256, out_features=512
        )
        self.value_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim),
        )

        self.num_classes = num_classes

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (state.reshape(state.shape[0], -1), goal.reshape(goal.shape[0], -1)), axis=1)
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        feature = torch.relu(activation)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DuelingDQN5Mini(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQN5Mini, self).__init__()

        self.encoder_hidden_layer = nn.Linear(
            in_features=2 * np.prod(obs_dim), out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=64
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=64, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=256
        )
        self.value_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        self.num_classes = num_classes

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (state.reshape(state.shape[0], -1), goal.reshape(goal.shape[0], -1)), axis=1)
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        feature = torch.relu(activation)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DuelingDQN5_(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQN5_, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1, stride=1),
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

        encoder_feature = 128
        self.feasible_act_encoder = Encoder(input_shape=2 * np.prod(obs_dim), output_feature=encoder_feature)
        self.off_pattern_tube_encoder = Encoder(input_shape=2 * np.prod(obs_dim), output_feature=encoder_feature)

        def conv2d_size_out(size, kernel_size, padding, stride=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        w1 = conv2d_size_out(obs_dim[0], 3, 1)
        h1 = conv2d_size_out(obs_dim[1], 3, 1)
        w2 = conv2d_size_out(w1, 3, 1)
        h2 = conv2d_size_out(h1, 3, 1)
        w3 = conv2d_size_out(w2, 3, 1)
        h3 = conv2d_size_out(h2, 3, 1)
        w4 = conv2d_size_out(w3, 3, 1)
        h4 = conv2d_size_out(h3, 3, 1)
        w5 = conv2d_size_out(w4, 3, 1)
        h5 = conv2d_size_out(h4, 3, 1)

        self.value_layer = nn.Sequential(
            nn.Linear(w5 * h5 * 128 + encoder_feature, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU()
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(w5 * h5 * 128 + encoder_feature, 4096),
            nn.ReLU(),
            nn.Linear(4096, action_dim),
            nn.ReLU()
        )

        self.num_classes = num_classes

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        ft_list = []
        feasible_act_feature = self.feasible_act_encoder(state, goal)
        ft_list.append(feasible_act_feature.detach())

        # off_pattern_tube_feature = self.off_pattern_tube_encoder(state, goal)
        # ft_list.append(off_pattern_tube_feature.detach())
        x = torch.cat((state.unsqueeze(1), goal.unsqueeze(1)), axis=1)
        custom_feature = self.feature_layer(x)
        ft_list.append(custom_feature)

        feature = F.relu(torch.cat(ft_list, axis=1))
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DuelingDQNMini(nn.Module):
    def __init__(self, obs_dim: tuple, action_dim: int, num_classes: int):
        """Initialization."""
        super(DuelingDQNMini, self).__init__()

        self.encoder_hidden_layer = nn.Linear(
            in_features=np.prod(obs_dim) * 2, out_features=64
        )

        self.encoder_output_layer = nn.Linear(
            in_features=64, out_features=32
        )

        self.q_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
        )

        self.num_classes = num_classes

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (state.reshape(state.shape[0], -1), goal.reshape(goal.shape[0], -1)), axis=1)
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        feature = F.relu(code)
        q = self.q_layer(feature)
        return q


if __name__ == "__main__":
    rack_size = (5, 10)
    dqn = DuelingDQN5(obs_dim=rack_size, action_dim=np.prod(rack_size) ** 2, num_classes=3)
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
