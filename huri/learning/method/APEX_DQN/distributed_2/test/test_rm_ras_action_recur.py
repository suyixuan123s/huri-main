""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231002osaka

"""
import time
import unittest
import logging

import numpy as np
from huri.learning.env.rack_v3 import create_env, RackStatePlot
from huri.learning.method.APEX_DQN.distributed.pipeline import Trajectory
from huri.learning.method.APEX_DQN.distributed.reanalyzer import extract_path_from_traj, rm_ras_actions_recur3
import cv2


class TestRMRASACTIONRECUR(unittest.TestCase):
    def setUp(self):
        self.env = create_env(rack_sz=(5, 10),
                              num_tube_class=1,
                              seed=888,
                              toggle_curriculum=True,
                              toggle_goal_fixed=True,
                              num_history=1)
        GOAL_PATTERN = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                                 [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]])
        self.env.set_goal_pattern(GOAL_PATTERN)
        self.env.reset()
        rsp = RackStatePlot(self.env.goal_pattern)
        traj = Trajectory(self.env.goal_pattern)
        Trajectory.action_dim = self.env.action_space_dim
        self.env.scheduler.set_training_level(25)
        state = self.env.reset()
        state_list = [state]
        for i in range(100):
            action = self.env.sample(state)
            next_state, reward, done, info = self.env.step(action)
            traj.add_transition(state, action, reward, next_state, done)
            state = next_state
            state_list.append(state)
            if done:
                break
        self.trajectory = traj
        # plot = rsp.plot_states(state_list, row=8, img_scale=1.2).get_img()
        # cv2.imshow('plot', plot)
        # cv2.waitKey(0)

    def test_rms(self):
        horizon = 8
        a = time.time()
        redundant_path, redundant_abs_state_paired_str, goal_pattern = extract_path_from_traj(self.trajectory)
        goal_state_np = goal_pattern
        refined_path, refined_path_her = rm_ras_actions_recur3(redundant_path,
                                                               h=horizon,
                                                               goal_pattern=redundant_path[-1],
                                                               infeasible_dict={}, )
        b = time.time()
        print(f"time consumption: {b - a}")
