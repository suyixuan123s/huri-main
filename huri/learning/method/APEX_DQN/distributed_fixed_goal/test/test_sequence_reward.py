""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231125osaka

"""
from typing import List
import numpy as np
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackState, RackStatePlot


def get_movable(state):
    feasible_actions = state.feasible_action_set_detail
    q = np.zeros(state.shape)
    q[feasible_actions[:, 0], feasible_actions[:, 1]] = 1
    return q


def get_trajectory_reward(env: RackArrangementEnv,
                          state_list: List[List[np.ndarray]],
                          goal_pattern,
                          toggle_debug=False):
    reward_list = []
    movable_list = [get_movable(state_list[0])]
    state_valid_list = []
    for _i in np.arange(len(state_list) - 1):
        s_current, s_next = state_list[_i], state_list[_i + 1]
        action = env.action_between_states(s_current=s_current, s_next=s_next)
        is_finsihed = env.is_finished(s_next, goal_pattern)
        try:
            reward, info = env._get_reward(is_finsihed, s_current, s_next, goal_pattern, toggle_debug=toggle_debug,
                                           toggle_reward_info=True)
        except:
            print("--------")
            print(env._get_reward(is_finsihed, s_current, s_next, goal_pattern, toggle_debug=toggle_debug,
                                           toggle_reward_info=True))
        reward_list.append(f'{reward} ({info})')
        # movable
        movable_list.append(get_movable(s_next))
        state_valid_list.append(env.is_state_valid(s_next))
        if is_finsihed:
            # if toggle_debug:
            #     cv2.destroyWindow('debug')
            break
    return reward_list, movable_list, state_valid_list


from huri.learning.method.APEX_DQN.distributed_fixed_goal.env import create_fixe_env, \
    GOAL_PATTERN_5x10_1
import cv2


def main(state_list, goal_pattern):
    rack_size = goal_pattern.shape
    num_class = len(np.unique(goal_pattern[goal_pattern > 0]))
    env = create_fixe_env(rack_sz=rack_size,
                          goal_pattern=goal_pattern,
                          num_tube_class=num_class,
                          num_history=1,
                          seed=np.random.randint(0, 1000000),
                          toggle_curriculum=True)
    reward_list, movable_list, state_valid_list = get_trajectory_reward(env, state_list, goal_pattern=goal_pattern, toggle_debug=False)
    print("Reward list is: ", reward_list)
    print("State valid list is:", state_valid_list)
    # print("Accumulated Reward is: ", np.sum(reward_list))

    rsp = RackStatePlot(goal_pattern, )
    img = rsp.plot_states(state_list,
                          row=7,
                          reward_list=reward_list,
                          movable_mtx=movable_list,
                          img_scale=2).get_img()
    cv2.imshow(f"original", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    import huri.core.file_sys as fs

    human = fs.load_pickle(
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\test\debug_data\human_traj_saved.pkl')
    print(repr(np.asarray(human[0])))
    GOAL_PATTERN = np.array([[1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
                             [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [3, 3, 3, 3, 0, 0, 4, 4, 4, 4],
                             [3, 3, 3, 3, 0, 0, 4, 4, 4, 4]])

    main(human, GOAL_PATTERN)

    # human, goal_pattern = fs.load_pickle('20231125124214_eval_0.pkl')
    # main(human, goal_pattern)
    # #
    # human, goal_pattern = fs.load_pickle('20231125125309_eval_17.pkl')
    # main(human, goal_pattern)

    # seq = fs.load_pickle(
    #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\test\evaluate_score\debug_failed_path_2.pkl')
    # seq = fs.load_pickle(
    #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\test\evaluate_score\debug_failed_path.pkl')
    seq = fs.load_pickle(
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\test\debug_data\debug_failed_path3.pkl')
    print(repr(np.array(seq[0])))
    print(len(seq))
    seq_start_id = 0
    for seq_id, (s, g) in enumerate(seq[seq_start_id:]):
        print(seq_id + seq_start_id)
        main(s, g)

# id: 2 3
