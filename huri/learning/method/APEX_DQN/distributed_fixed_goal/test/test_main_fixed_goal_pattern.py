""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231002osaka

TODO:
    1. Change the set_training_level

"""

import time
import unittest
import logging

import numpy as np
from huri.learning.env.rack_v3 import create_env, RackStatePlot
import cv2


class TestMainFix(unittest.TestCase):
    def setUp(self):
        self.env = create_env(rack_sz=(5, 10),
                              num_tube_class=3,
                              seed=888,
                              toggle_curriculum=True,
                              toggle_goal_fixed=True,
                              num_history=1)
        GOAL_PATTERN = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                                 [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                                 [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                                 [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                                 [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
        self.env.set_goal_pattern(GOAL_PATTERN)
        self.env.reset()

    def test_goal_pattern(self):
        rsp = RackStatePlot(self.env.goal_pattern)
        self.env.scheduler.set_training_level(4)
        for level in range(4, 40):
            self.env.scheduler.update_training_level()
            state = self.env.reset()
            state_list = [state]
            for i in range(10):
                action = self.env.sample(state)
                next_state, reward, done, info = self.env.step(action)
                print(next_state)
                state = next_state
                state_list.append(state)
                if done:
                    break
            plot = rsp.plot_states(state_list, row=8, img_scale=1.2).get_img()
            cv2.imshow('plot', plot)
            cv2.waitKey(0)
