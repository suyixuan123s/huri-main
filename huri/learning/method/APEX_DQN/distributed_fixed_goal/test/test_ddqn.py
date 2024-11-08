""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231002osaka

"""

import time
import unittest
import logging
import torch
import numpy as np
from huri.learning.env.rack_v3 import create_env, RackStatePlot
from huri.components.utils.matlibplot_utils import Plot
from huri.learning.method.APEX_DQN.distributed.network import DDQN, DDQN2
import matplotlib.pyplot as plt
import cv2

class TestDDQN(unittest.TestCase):
    def setUp(self):
        # fixed env
        self.env_fixed = create_env(rack_sz=(5, 10),
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
        self.env_fixed.set_goal_pattern(GOAL_PATTERN)
        self.env_fixed.reset()

        # unfixed env
        self.env = create_env(rack_sz=(5, 10),
                              num_tube_class=3,
                              seed=888,
                              toggle_curriculum=True,
                              toggle_goal_fixed=False,
                              num_history=4)

    def test_DDQN_env_fixed(self):
        env = self.env_fixed
        input_shape = env.observation_space_dim
        num_actions = env.action_space_dim
        print("input_shape: ", input_shape)
        print("num_actions: ", num_actions)
        network = DDQN(input_shape, num_actions, num_filters=10, num_fc_units=128)
        state = torch.from_numpy(env.reset().state).float()[None,]
        goal = torch.from_numpy(env.goal_pattern.state).float()[None,]
        state_t = torch.cat((state[:, None, ...], goal[:, None, ...]), axis=1)
        v = network(state_t)
        print("DDQN output: ", v)

    def test_DDQN2_env_fixed(self):
        env = self.env_fixed
        input_shape = env.observation_space_dim
        num_actions = env.action_space_dim
        print("input_shape: ", input_shape)
        print("num_actions: ", num_actions)
        network = DDQN2(input_shape, num_actions,
                        num_filters=5,
                        num_res_block=10,
                        num_fc_units=int(num_actions / 2))
        state = torch.from_numpy(env.reset().state).float()[None,]
        goal = torch.from_numpy(env.goal_pattern.state).float()[None,]
        state_t = torch.cat((state[:, None, ...], goal[:, None, ...]), axis=1)
        a = time.time()
        v = network(state_t)
        b = time.time()
        print("DDQN output: ", v)
        print("Predict time: ", (b - a) * 1000, "ms")

    def test_visualize_conv_block_out(self):
        x, conv_s, conv_g = torch.load('debug_data/conv_block_out.pt')
        # visulize x using matplotlib
        x = x[0, 0, ...].numpy()
        img = visualize_lattice(x).get_img()
        cv2.imshow('x', img)
        cv2.waitKey(0)
