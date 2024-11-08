""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231129osaka

"""
from huri.learning.method.APEX_DQN.distributed_fixed_goal.env import create_fixe_env, \
    GOAL_PATTERN_5x10_1
import cv2
import numpy as np
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackState, RackStatePlot


def main(state_list, goal_pattern):
    rack_size = goal_pattern.shape
    num_class = len(np.unique(goal_pattern[goal_pattern > 0]))
    env = create_fixe_env(rack_sz=rack_size,
                          goal_pattern=goal_pattern,
                          num_tube_class=num_class,
                          num_history=1,
                          seed=np.random.randint(0, 1000000),
                          toggle_curriculum=True)
    movable_list = []
    for current_state in state_list:
        feasible_actions = current_state.feasible_action_set_detail
        q = np.zeros(current_state.shape)
        q[feasible_actions[:, 0], feasible_actions[:, 1]] = 1
        rsp = RackStatePlot(goal_pattern, )
        img = rsp.plot_states(state_list,
                              row=7,
                              img_scale=2).get_img()
        cv2.imshow(f"original", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    pass
