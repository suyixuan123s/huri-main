""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231028osaka

"""

import unittest
from huri.learning.env.rack_v3.env2 import RackState, RackArrangementEnv, RackStatePlot
import numpy as np
import networkx as nx
import cv2

np.set_printoptions(threshold=np.inf)

class_num = 2
rack_size = (3, 8)


def random_rack_state() -> RackState:
    # Create an instance of the class to test
    while 1:
        initstate = np.random.choice(class_num, rack_size, p=[.5, *([.5 / (class_num - 1)] * (class_num - 1))])
        if np.sum(initstate) > 0:
            break
    return RackState(initstate)


class RackStateTestCase(unittest.TestCase):

    def setUp(self) -> None:
        obs_dim, head_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)
        print(f"obs dimension is {obs_dim}, head dimension is {head_dim}, action dimension is {act_dim}")
        self.env = RackArrangementEnv(rack_size=rack_size, num_classes=class_num, observation_space_dim=obs_dim,
                                      action_space_dim=act_dim, is_goalpattern_fixed=False, is_curriculum_lr=True)

    def test_rack_state_possible_actions(self):
        s = self.env.reset()
        possible_actions = s.feasible_action_set
        print("State:\n", s)
        print("Possible actions:\n", possible_actions)

    def test_sample(self):
        s = self.env.reset()
        a = self.env.sample()
        while a is not None:
            print("State:\n", s)
            print("Sampled actions:\n", a)
            s = self.env.step(a)[0]
            a = self.env.sample()
        print(len(self.env.rack_state_history))

    def test_action_between_two_states(self):
        r = []
        for i in range(100):
            s = self.env.reset()
            a = self.env.sample()
            sn = self.env.step(a)[0]
            a_diff1 = self.env.action_between_states(s, sn)
            a_diff2 = self.env.action_between_states_constraint_free(s, sn)
            print(f"state:\n{s}, next state:\n{sn}, action is {a}, "
                  f"a_diff1 is {a_diff1}, action_diff2 is {a_diff2}")
            r.append(np.all(a==a_diff1) and np.all(a==a_diff2))
        return self.assertTrue(np.all(r))

    def test_random_goal_pattern_scheduler3(self):
        scheduler_name = 'GoalRackStateScheduler3'
        obs_dim, head_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)
        env = RackArrangementEnv(rack_size=rack_size, num_classes=class_num, observation_space_dim=obs_dim,
                                 action_space_dim=act_dim, is_curriculum_lr=True, is_goalpattern_fixed=False,
                                 scheduler=scheduler_name)
        for i in range(np.prod(rack_size)):
            for j in range(5):
                a = env.reset()
                rp = RackStatePlot(goal_pattern=env.goal_pattern)
                img = rp.plot_states(rack_states=[a], row=1).get_img()
                cv2.imshow("img", img)
                cv2.moveWindow("img", 300, 300)
                cv2.waitKey(0)
            print(env.scheduler.state_level, env.scheduler.class_level)
            print("training level:", i + 1)
            env.scheduler.set_training_level(i + 1)

    def test_traj(self):
        s = self.env.reset()
        traj = [s]
        for i in range(30):
            a = self.env.sample()
            s = self.env.step(a)[0]
            traj.append(s)
        rp = RackStatePlot(goal_pattern=self.env.goal_pattern)
        img = rp.plot_states(rack_states=traj, row=10).get_img()
        cv2.imshow("img", img)
        cv2.moveWindow("img", 300, 300)
        cv2.waitKey(0)