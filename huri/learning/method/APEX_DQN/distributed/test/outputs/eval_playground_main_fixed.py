"""

Author: Hao Chen (chen960216@gmail.com)
Created: 20230727osaka

"""
import copy
import time
import torch
import hydra
import huri.core.file_sys as fs
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed.pipeline import Eval2, SharedState
from huri.learning.method.APEX_DQN.distributed.network import DDQN2
import numpy as np


@hydra.main(config_path='../../params', config_name='20230517_fixed.yaml', version_base='1.3')
def main(cfg):
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed' / 'run'
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                          num_history=1)
    GOAL_PATTERN = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])

    env_meta.set_goal_pattern(GOAL_PATTERN)
    env_meta.reset()
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    network = DDQN2(input_shape, num_actions, num_filters=30, num_res_block=10, num_fc_units=int(num_actions / 2),
                    toggle_debug=True)
    ckpnt = {'train_steps': 1,
             'weights':
                 torch.load(
                     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\run\20231002212516\data\model_best.chkpt')[
                     'dqn_state_dict'],
             'training_level': 10,
             'trajectory_list': []
             }
    ckpnt['weights'] = {k: v.cpu() for k, v in ckpnt['weights'].items()}
    shared_state = SharedState.remote(ckpnt)

    eval = Eval2(env=env_meta,
                net=copy.deepcopy(network),
                cfg=cfg['eval'],
                shared_state=shared_state,
                save_path=save_path.joinpath('data'),
                log_path=None,
                toggle_visual=True)

    v = env_meta.copy()
    # v.is_goalpattern_fixed = True
    # GOAL = np.array([[1, 1, 0, 1, 0, 0],
    #                  [1, 1, 0, 1, 0, 0],
    #                  [1, 1, 0, 1, 0, 0],
    #                  [1, 1, 0, 1, 0, 0],
    #                  [1, 1, 0, 1, 0, 0],
    #                  [1, 1, 0, 1, 0, 0]])
    # v.set_goal_pattern(GOAL)

    eval.single_test(v, eval_num=100)
    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
