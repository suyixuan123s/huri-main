""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231122osaka

"""
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackState, RackStatePlot

import numpy as np

GOAL_PATTERN_5x5_2 = np.array([[1, 1, 2, 3, 4],
                               [1, 2, 3, 4, 3],
                               [2, 3, 4, 3, 2],
                               [3, 4, 3, 2, 1],
                               [4, 3, 2, 1, 1]])

GOAL_PATTERN_5x10_1 = np.array([[1, 1, 0, 0, 2, 2, 0, 0, 3, 3],
                                [1, 1, 0, 0, 2, 2, 0, 0, 3, 3],
                                [1, 1, 0, 0, 2, 2, 0, 0, 3, 3],
                                [1, 1, 0, 0, 2, 2, 0, 0, 3, 3],
                                [1, 1, 0, 0, 2, 2, 0, 0, 3, 3]])

GOAL_PATTERN_5x10_2 = np.array([[1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
                                [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [3, 3, 3, 3, 0, 0, 4, 4, 4, 4],
                                [3, 3, 3, 3, 0, 0, 4, 4, 4, 4]])

GOAL_PATTERN_5x10_3 = np.array([[1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
                                [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
                                [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
                                [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
                                [1, 0, 2, 0, 3, 0, 4, 0, 5, 0]])

GOAL_PATTERN_3x5_3 = np.array([[1, 0, 2, 0, 3,],
                               [1, 0, 2, 0, 3,],
                               [1, 0, 2, 0, 3,]])

GOAL_PATTERN_10x10_3 = np.array([[1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                                 [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                                 [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                                 [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                                 [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                                 [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                                 [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                                 [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                                 [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                                 [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ], ])


# GOAL_PATTERN_5x10_2 = np.array([[1, 1, 0, 0, 2, 2, 0, 0, 0, 0],
#                                 [1, 1, 0, 0, 2, 2, 0, 0, 0, 0],
#                                 [1, 1, 0, 0, 2, 2, 0, 0, 0, 0],
#                                 [1, 1, 0, 0, 2, 2, 0, 0, 0, 0],
#                                 [1, 1, 0, 0, 2, 2, 0, 0, 0, 0]])


# GOAL_PATTERN_5x10_1 = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

# GOAL_PATTERN_5x5_1 = np.array([[1, 1, 0, 0, 0, ],
#                                [1, 1, 0, 0, 0, ],
#                                [1, 1, 0, 0, 0, ],
#                                [1, 1, 0, 0, 0, ],
#                                [1, 1, 0, 0, 0, ]])
#
# GOAL_PATTERN_5x5_2 = np.array([[1, 2, 2, 1, 0, ],
#                                [1, 2, 2, 1, 0, ],
#                                [1, 2, 2, 1, 0, ],
#                                [1, 2, 2, 1, 0, ],
#                                [1, 2, 2, 1, 0, ]])
#
# GOAL_PATTERN_5x5_3 = np.array([[1, 1, 0, 2, 2, ],
#                                [1, 1, 0, 2, 2, ],
#                                [1, 1, 0, 2, 2, ],
#                                [1, 1, 0, 2, 2, ],
#                                [1, 1, 0, 2, 2, ]])


# GOAL_PATTERN_5x5_1 = np.array([[1, 0, 4, 0, 2],
#                                [1, 1, 4, 0, 2],
#                                [1, 5, 4, 5, 2],
#                                [0, 5, 5, 0, 5],
#                                [0, 3, 3, 5, 5]])


def create_fixe_env(rack_sz,
                    goal_pattern,
                    num_tube_class,
                    seed=888,
                    num_history=4,
                    scheduler='RackStateScheduler2_5',
                    toggle_curriculum=False, ) -> 'RackArrangementEnv':
    observation_space_dim, action_space_dim = RackState.get_obs_act_dim_by_size(tuple(rack_sz))
    env = RackArrangementEnv(rack_size=tuple(rack_sz),
                             num_classes=num_tube_class,
                             num_history=num_history,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=toggle_curriculum,
                             is_goalpattern_fixed=True,
                             scheduler=scheduler,
                             seed=seed)
    env.set_goal_pattern(np.asarray(goal_pattern))
    env.scheduler.class_level = env.num_classes
    env.reset()
    return env


if __name__ == '__main__':
    import cv2

    # GOAL_PATTERN = GOAL_PATTERN_10x10_3
    GOAL_PATTERN = GOAL_PATTERN_3x5_3
    rack_size = GOAL_PATTERN.shape
    class_num = len(np.unique(GOAL_PATTERN[GOAL_PATTERN > 0]))
    env = create_fixe_env(rack_sz=rack_size,
                          goal_pattern=GOAL_PATTERN,
                          num_tube_class=class_num,
                          num_history=1,
                          # scheduler='RackStateScheduler2_5',
                          scheduler='RackStateScheduler',
                          toggle_curriculum=True)

    # env.reset_state(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                           [0, 3, 0, 0, 1, 4, 0, 0, 0, 0],
    #                           [0, 0, 0, 1, 2, 1, 3, 0, 0, 0],
    #                           [1, 0, 0, 1, 2, 1, 3, 0, 0, 0],
    #                           [0, 2, 0, 0, 4, 1, 0, 0, 0, 0]]))
    # print("Is it finished", env.is_finished())
    # exit(0)
    for j in range(1, np.count_nonzero(GOAL_PATTERN) + 1):
        env.scheduler.set_training_level(j)
        print("state level -> ", env.scheduler.state_level, "training level ->", j)
        for i in range(5):
            a = env.reset()
            # print(repr(a))
            rp = RackStatePlot(goal_pattern=env.goal_pattern)
            img = rp.plot_states(rack_states=[a], row=1, img_scale=10).get_img()
            cv2.imshow("img", img)
            cv2.moveWindow("img", 300, 300)
            cv2.waitKey(0)

    print(env.scheduler.training_plan)
    # env.scheduler.class_level = 2
    print(f"Number of actions: {env.action_space_dim}")
    print(f"total number of plans: {len(env.scheduler.training_plan)}")
    for j in range(35, len(env.scheduler.training_plan) + 1):
        env.scheduler.set_training_level(j)
        print("state level -> ", env.scheduler.state_level, "training level ->", j)
        for i in range(5):
            a = env.reset()
            # print(repr(a))
            rp = RackStatePlot(goal_pattern=env.goal_pattern)
            img = rp.plot_states(rack_states=[a], row=1, img_scale=10).get_img()
            cv2.imshow("img", img)
            cv2.moveWindow("img", 300, 300)
            cv2.waitKey(0)
