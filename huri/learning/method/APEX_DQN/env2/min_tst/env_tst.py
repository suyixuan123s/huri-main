from huri.learning.method.APEX_DQN.actor import Actor, Eval
from huri.learning.utils import select_device, LOGGER
from huri.learning.env.rack_v2.env import RackArrangementEnv, RackStatePlot
from huri.learning.network.d3qn_attention import DuelingDQN5_
import numpy as np
import copy
import torch
import huri.core.file_sys as fs
import torch.multiprocessing as mp
import cv2
from huri.core.print_tool import text_pd


def plot(states, goal):
    drawer = RackStatePlot(goal)
    fig = drawer.plot_states(states).get_img()
    cv2.imshow("window", fig)
    cv2.waitKey(0)


num_tube_classes = 1
rack_size = (3, 3)
action_space_dim = np.prod(rack_size) ** 2
observation_space_dim = (rack_size[0], rack_size[1])
observation_space_dim_nn = (1, *rack_size)
env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_tube_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_curriculum_lr=True,
                         is_goalpattern_fixed=False,
                         seed=888)
if __name__ == "__main__":

    env1 = env.copy()
    env1.scheduler.state_level = 2
    env1.scheduler.class_level = 4
    while True:
        state = env1.reset()
        goal = env1.goal_pattern
        print(text_pd(state.state.reshape(1,-1)))
        print("----")
        print(text_pd(state.adjacency_matrix))
        plot([state], goal)
        print("?")
