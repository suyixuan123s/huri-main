""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231125osaka

"""
import time

import numpy as np

if __name__ == '__main__':
    from huri.learning.method.APEX_DQN.distributed_fixed_goal.env import (create_fixe_env,
                                                                          GOAL_PATTERN_5x5_2,
                                                                          RackStatePlot,
                                                                          RackState)

    import cv2

    GOAL_PATTERN = GOAL_PATTERN_5x5_2
    rack_size = GOAL_PATTERN.shape
    category = len(np.unique(GOAL_PATTERN[GOAL_PATTERN > 0]))
    env = create_fixe_env(rack_sz=rack_size,
                          goal_pattern=GOAL_PATTERN,
                          num_tube_class=category,
                          num_history=1,
                          toggle_curriculum=True)
    env.scheduler.class_level = 2
    for j in range(1, len(env.scheduler.training_plan)):
        env.scheduler.set_training_level(j)
        print("state level -> ", env.scheduler.state_level, "training level ->", j)
        for i in range(30):
            # a = env.reset()
            a = env.reset_state(np.array([[0, 0, 0, 0, 0],
                                          [0, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 0],
                                          [0, 1, 1, 0, 0]]))

            goal_pattern = env.goal_pattern
            current_state = a
            feasible_actions = a.feasible_action_set_detail
            mismatch_mask_state = (goal_pattern.state != current_state.state) & (current_state.state != 0)
            unsatisfied_state = np.where(mismatch_mask_state, current_state, 0)
            q = np.zeros((*current_state.shape, *current_state.shape))
            q[feasible_actions[:, 0], feasible_actions[:, 1], feasible_actions[:, 2], feasible_actions[:, 3]] = 1
            movable_map = np.zeros((current_state.shape), dtype=bool)
            movable_map[feasible_actions[:, 0], feasible_actions[:, 1]] = 1
            fillable_map = np.zeros((current_state.shape), dtype=bool)
            fillable_map[feasible_actions[:, 2], feasible_actions[:, 3]] = 1
            fm, mm = a.fillable_movable_region
            rp = RackStatePlot(goal_pattern=env.goal_pattern)
            # ------------
            rp = RackStatePlot(goal_pattern=env.goal_pattern)
            img = rp.plot_states(rack_states=[a],
                                 # movable_mtx=[movable_map],
                                 fillable_mtx=[fm],
                                 row=1,
                                 img_scale=10).save_fig("TTe.jpg")

            # --------------
            pa = time.time()
            num_unsatisfied_tube, num_feasible_goal_slot, fillable_map = env.is_safe2(RackState(a),
                                                                                      RackState(env.goal_pattern),
                                                                                      category_num=category,
                                                                                      toggle_fillable_map=True)
            pb = time.time()
            print("feasible goal current", num_feasible_goal_slot)
            print("number of tube not at the goal", num_unsatisfied_tube)
            print("time consumption:", pb - pa)
            # --------------
            # for i in np.argwhere(unsatisfied_state):
            #     movable = np.zeros_like(a)
            #     movable[i[0], i[1]] = 1
            #     fillable = q[i[0], i[1]]
            #     rp = RackStatePlot(goal_pattern=env.goal_pattern)
            #     img = rp.plot_states(rack_states=[a],
            #                          fillable_mtx=[fillable],
            #                          movable_mtx=[movable],
            #                          row=1,
            #                          img_scale=10).get_img()
            #     cv2.imshow("img", img)
            #     cv2.moveWindow("img", 300, 300)
            #     cv2.waitKey(0)
            rp = RackStatePlot(goal_pattern=env.goal_pattern)
            img = rp.plot_states(rack_states=[a],
                                 movable_mtx=[fillable_map],
                                 row=1,
                                 img_scale=10).get_img()
            cv2.imshow("img", img)
            cv2.moveWindow("img", 300, 300)
            cv2.waitKey(0)
