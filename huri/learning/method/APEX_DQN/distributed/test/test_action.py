"""

Author: Hao Chen (chen960216@gmail.com)
Created: 20230727osaka

"""
import time

processes = []
import torch
import hydra
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed.network import DDQN2 as DDQN
import numpy as np
import os
import matplotlib.patches as patches
from huri.learning.method.APEX_DQN.distributed.utils import to_onehot, swap_values, abs_state
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackStatePlot, RackState
from huri.learning.method.APEX_DQN.distributed.utils import abs_state, category_feasible_action

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict

color = ['green', 'red', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white', 'purple', 'pink', 'brown',
         'orange']
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 9  # You can adjust the size as needed
# font_properties = {'family': 'Arial', 'fontsize': 12, 'weight': 'bold'}
font_properties = {'family': 'Arial', 'fontsize': 12, }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path='../params', config_name='20230517_3x6_2.yaml', version_base='1.3')
def main(cfg):
    env_meta: RackArrangementEnv = create_env(rack_sz=cfg['env']['rack_sz'],
                                              num_tube_class=cfg['env']['num_tube_class'],
                                              seed=cfg['env']['seed'],
                                              toggle_curriculum=cfg['env']['toggle_curriculum'],
                                              toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                                              scheduler='GoalRackStateScheduler3',
                                              num_history=1)

    env_meta.reset_state(np.array([[0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 2., 0., 1.],
                                   [0., 0., 0., 0., 0., 0.]], dtype=int))
    print(env_meta._expr_action(118))

if __name__ == '__main__':
    main()
