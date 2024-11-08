import torch
import copy
from huri.learning.network.d3qn import DuelingDQN
import numpy as np
from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv, RackStatePlot
import huri.core.file_sys as fs


def test_n_episode(env: RackArrangementEnv,
                   net: DuelingDQN,
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   reset_num=50,
                   num_eps=100,
                   toggle_show=False,
                   toggle_failed_path = False):
    _failed_eps = 0
    _success_eps = 0
    average_success_length = 0
    path_len_list = []
    failed_ids = []
    with torch.no_grad():
        for _eps in range(num_eps):
            score = 0
            state = env.reset()
            actions = []
            path = []
            for step in range(reset_num):
                feasible_action_set = torch.as_tensor(state.feasible_action_set, dtype=torch.int64, device=device)
                dqn_action_value = net(
                    torch.as_tensor(state.state, dtype=torch.float32, device=device)[None, None]).detach()
                selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
                # print(env._expr_action(selected_action))
                state, reward, done, _ = env.step(action=selected_action)
                actions.append(selected_action)
                score += reward
                if done:  # done
                    # print(env.reward_history)
                    # print(env.state_r_history)
                    # print(score)
                    # print("----------------")
                    # print(test_data[epsiode][0])
                    # print(test_data[epsiode][1])
                    # print(env.rack_state_history)
                    # print(env.state)
                    # print("----------------")
                    if reward > 0:
                        _success_eps += 1
                    else:
                        failed_ids.append(_eps)
                    path = copy.deepcopy(env.rack_state_history + [env.state])
                    # if smooth_path:
                    #     path = refine_path(path, max_iter=len(path) * 2)
                    path_len_list.append(len(path))
                    if toggle_show:
                        drawer = RackStatePlot(env.goal_pattern)
                        drawer.plot_states(path, row=6)
                    score = 0
                    # [print(a) for a in actions]
                    # print(len(actions))
                    break
            else:
                failed_ids.append(_eps)
                if toggle_failed_path:
                    path = copy.deepcopy(env.rack_state_history + [env.state])
                    drawer = RackStatePlot(env.goal_pattern)
                    drawer.plot_states(path, row=6)

                _failed_eps += 1
    print(f"success episode: {_success_eps}")
    print(f"failed episode: {_failed_eps}")
    print(f"average length of the success episode is {sum(path_len_list)/len(path_len_list)}")


if __name__ == "__main__":
    # initialize the environment
    num_tube_classes = 3
    rack_size = (5, 10)
    action_space_dim = np.prod(rack_size) ** 2
    observation_space_dim = (1, *rack_size)
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_tube_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=True,
                             is_goalpattern_fixed=True,
                             seed=1988,
                             difficulty=26)
    env.goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])

    # data_path = fs.workdir_learning / "run" / f"dqn_2021_12_01_18_05_30"
    # model_name = "model_4130000-4140000.pth"
    # model_path = data_path / "model" / model_name

    data_path = fs.workdir_learning / "run" / f"dqn_2022_01_08_21_13_41"
    # model_name = "model_1858000-1860000.pth"
    model_name = "model_4818000-4820000.pth"
    model_path = data_path / "model" / model_name

    # load neural network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DuelingDQN(obs_dim=observation_space_dim, action_dim=action_space_dim).to(device)
    net.load_state_dict(torch.load(model_path))

    test_n_episode(env, net, reset_num=50, device=device, num_eps=100, toggle_show=True, toggle_failed_path=False)
