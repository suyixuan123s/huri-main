"""

Author: Hao Chen (chen960216@gmail.com)
Created: 20230727osaka

"""
processes = []
import torch
import hydra
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed.network import DDQN2 as DDQN
import numpy as np
from huri.learning.method.APEX_DQN.distributed.utils import to_onehot, swap_values, abs_state
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path='../params', config_name='20230517_3x6_2.yaml', version_base='1.3')
def main(cfg):
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                          scheduler='GoalRackStateScheduler3',
                          num_history=1)
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    network = dqn = DDQN(input_shape,
                         num_actions,
                         num_category=cfg['env']['num_tube_class'],
                         num_filters=cfg['ddqn']['num_filters'],
                         num_res_block=cfg['ddqn']['num_res_block'],
                         num_fc_units=cfg['ddqn']['num_fc_units'], )
    network.to(device)
    # Update the default rc settings
    from collections import defaultdict

    env = env_meta.copy()
    states = []
    goals = []
    for i in range(64):
        state = env.reset().state
        goal = env.goal_pattern.state
        states.append(state)
        goals.append(goal)
    states = np.array(states)
    goals = np.array(goals)

    input_value = abs_state(torch.as_tensor(states, dtype=torch.float32, device=device),
                            torch.as_tensor(goals, dtype=torch.float32, device=device),
                            env.num_classes)
    dqn(input_value)

if __name__ == '__main__':
    main()
