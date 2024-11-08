import cv2

from huri.learning.method.APEX_DQN.env3.actor import Actor, Eval
from huri.learning.utils import select_device, LOGGER
from main import create_agent
import numpy as np
import copy
import torch
import huri.core.file_sys as fs
import torch.multiprocessing as mp
from env_tst import env, num_tube_classes, rack_size, action_space_dim, observation_space_dim
import itertools
from env_tst import plot


def refined_path_to_transitions_recur(env, refined_path, goal_pattern):
    # if not np.all(refined_path[-1] == goal_pattern):
    #     print("ERROR ")
    transitions = []
    reward_debug = []
    for _i in np.arange(len(refined_path) - 1):
        s_current, s_next = refined_path[_i], refined_path[_i + 1]
        action = env.action_between_states(s_current=s_current, s_next=s_next)
        is_finsihed = env.is_finished(s_next, goal_pattern)
        # check if it is a unfinished state
        # if _i == len(refined_path) - 2 and not is_finsihed:
        #     reward = -50
        #     print("!!!!!!!!!!", s_current, s_next)
        # else:
        reward = env._get_reward(is_finsihed, s_current, s_next, goal_pattern)
        reward_debug.append(reward)
        tran_tmp = [goal_pattern,
                    s_current,
                    action,
                    reward,
                    s_next,
                    is_finsihed]
        transitions.append(tran_tmp)
        if is_finsihed:
            break
    print(f"Reward sequence is {reward_debug}")
    print(transitions)
    return transitions


if __name__ == "__main__":
    env1 = env.copy()
    device = select_device()

    net = create_agent(observation_space_dim, action_space_dim, num_tube_classes, device)

    eval_net_path = fs.workdir_learning / "run" / f"dqn_debug" / "model_last.pt"
    # eval_net_path = fs.Path("E:\huri_shared\huri\learning\method\APEX_DQN\env2\min_tst\\transfer_learning_weight.pt")
    net.load_state_dict(torch.load(str(eval_net_path)))

    shared_net = copy.deepcopy(net)
    shared_net.eval()
    shared_net.share_memory()
    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_buffer = [mp.Queue()]
    shared_state['state_level'] = 2
    shared_state['class_level'] = env.num_classes

    eval_proc = Eval(net=copy.deepcopy(net), env_test=env1,
                     reset_num=3,
                     eval_num=20,
                     eval_interval=1,
                     shared_net=shared_net,
                     shared_state=shared_state,
                     difficult_case_buffer_list=shared_buffer,
                     device=device,
                     inc_diff_threshold=48,
                     toggle_visual=True,
                     save_model_path_best=None,
                     save_model_path_last=None)

    while 1:
        # infeasible_data_list = fs.load_pickle("debug_data.pkl")
        infeasible_data_list = [[
            np.array([[0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 2, 1, 0, 1, 0, 0],
                      [0, 0, 2, 2, 2, 1, 1, 1, 1, 0],
                      [0, 0, 0, 2, 0, 1, 0, 1, 0, 0],
                      [0, 0, 2, 0, 0, 1, 1, 0, 0, 0]]),
            np.array([[1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                      [1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                      [1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                      [1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                      [1, 1, 0, 0, 0, 0, 0, 0, 3, 3]])
        ]]

        for _ in reversed(infeasible_data_list):
            reward_pool = []
            start = _[0]
            goal = _[1]

            state_init = env1.reset_state_goal(start, goal)

            num_dones = 0
            total_return = 0
            total_len = 0
            t_score = 0
            v_states = [state_init]
            t_state = state_init
            for t in itertools.count(1):
                t_action = eval_proc.dqn_select_action(t_state.feasible_action_set,
                                                       eval_proc.to_abs_state(t_state, env=env1),
                                                       toggle_debug=False)
                t_next_state, t_reward, t_done, _ = env1.step(t_action)
                reward_pool.append(t_reward)
                t_reward = t_reward
                t_state = t_next_state  # state = next_state
                t_score += t_reward  # reward
                v_states.append(t_next_state)
                if t_done:
                    if t_reward >= 0:
                        num_dones += 1
                    total_return += t_score
                    total_len += t
                    break
                if t % 100 == 0:
                    break
            print(f"Reward Get is {reward_pool}")
            refined_path_to_transitions_recur(env, v_states, goal)
            plot(v_states, goal)
            cv2.waitKey(0)
