import cv2

from huri.learning.method.APEX_DQN.env3.actor import Actor, Eval
from huri.learning.utils import select_device, LOGGER
from main import create_agent
import numpy as np
import copy
import torch
import huri.core.file_sys as fs
import torch.multiprocessing as mp
from env_tst import env, num_tube_classes, rack_size, action_space_dim, observation_space_dim,RackStatePlot
from huri.components.utils.img_utils import combine_images

def get_im(states, goal, w_name="window"):
    drawer = RackStatePlot(goal)
    fig = drawer.plot_states(states).get_img()
    return fig


if __name__ == "__main__":
    env1 = env.copy()

    device = select_device()
    while True:
        infeasible_data_list = fs.load_pickle("debug_data.pkl")
        fig_list = []
        for _ in infeasible_data_list:
            print(repr(_[0]))
            print(repr(_[1]))
            fig_list.append(get_im([_[0]],_[1]))
        plot = combine_images(fig_list, columns=12)
        # plot = np.concatenate((plot, rsp.plot_states([t_state]).get_img()), axis=1)
        cv2.imshow(f"plot_eval", plot)
        cv2.waitKey(0)

