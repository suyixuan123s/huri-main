"""

Author: Hao Chen (chen960216@gmail.com)
Created: 20230727osaka

"""
processes = []
import torch

import copy
import time

import ray
import hydra
import huri.core.file_sys as fs
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed.pipeline import Eval, SharedState
from huri.learning.method.APEX_DQN.distributed.network import DDQN2 as DDQN
from huri.learning.method.APEX_DQN.distributed.network_old import DDQN as DDQN2
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import numpy as np


@hydra.main(config_path='../params', config_name='20230517_3x6_2.yaml', version_base='1.3')
def main(cfg):
    ray.init(local_mode=True, )
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed' / 'run'
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                          scheduler='GoalRackStateScheduler3',
                          num_history=1)
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    network = DDQN(input_shape,
                   num_actions,
                   num_category=cfg['env']['num_tube_class'],
                   num_filters=cfg['ddqn']['num_filters'],
                   num_res_block=cfg['ddqn']['num_res_block'],
                   num_fc_units=cfg['ddqn']['num_fc_units'], )

    # network = DDQN2(input_shape, num_actions,
    #                 num_category=2,
    #                 num_filters=10,
    #                 num_res_block=19,
    #                 num_fc_units=128)

    ckpnt = {'train_steps': 1,
             'weights':
                 torch.load(
                     # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\runs\run_1\data\model_best_3.chkpt')[
                     # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\runs\R2\run4\data\model_best_7.chkpt')[
                     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\run\data\model_last.chkpt')[
                     # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\exp3\run7\data\model_last.chkpt')[
                     # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\exp2\run\data\model_best_12.chkpt')[
                     # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\exp2\run\data\model_best_12.chkpt')[
                     'dqn_state_dict'],
             'training_level': 6,
             'trajectory_list': []
             }
    ckpnt['weights'] = {k: v.cpu() for k, v in ckpnt['weights'].items()}
    shared_state = SharedState.remote(ckpnt)

    eval = Eval.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(env=env_meta,
                                                                               net=copy.deepcopy(network),
                                                                               cfg=cfg['eval'],
                                                                               shared_state=shared_state,
                                                                               H_buffer=None,
                                                                               save_path=save_path.joinpath('data'),
                                                                               log_path=None,
                                                                               toggle_visual=True)

    v = env_meta.copy()
    v.is_goalpattern_fixed = False
    # GOAL = np.array([[1, 1, 0, 1, 0, 0],
    #                  [1, 1, 0, 1, 0, 0],
    #                  [1, 1, 0, 1, 0, 0],
    #                  [1, 1, 0, 1, 0, 0],
    #                  [1, 1, 0, 1, 0, 0],
    #                  [1, 1, 0, 1, 0, 0]])
    # GOAL = np.array([[0, 0, 2, 1, 0, 0],
    #                  [0, 0, 0, 2, 2, 0],
    #                  [0, 0, 0, 0, 0, 0], ])
    # GOAL = np.array([[1, 1, 1, 0, 0, 0],
    #                  [1, 1, 1, 0, 0, 0],
    #                  [1, 1, 1, 0, 0, 0]])
    # STATE = np.array([[0, 0, 0, 1, 1, 1],
    #                   [0, 0, 0, 1, 1, 1],
    #                   [0, 0, 1, 1, 1, 1]])
    # GOAL = np.array([[0, 0, 0, 0, 2, 2],
    #                  [1, 1, 0, 0, 2, 2],
    #                  [1, 1, 0, 0, 0, 0]])
    # v.set_goal_pattern(GOAL)

    # GOAL = np.array([[1, 1, 1, 0, 0, 1],
    #                  [1, 1, 1, 1, 0, 1],
    #                  [1, 0, 1, 1, 1, 1]])
    # STATE = np.array([[1, 0, 1, 1, 0, 1],
    #                   [0, 1, 0, 0, 1, 1],
    #                   [0, 0, 1, 0, 0, 1]])
    # GOAL = np.array([[0, 0, 2, 0, 0, 0],
    #                   [0, 1, 2, 0, 0, 0],
    #                   [0, 2, 2, 0, 0, 0]])
    # STATE = np.array([[0, 2, 0, 0, 0, 2],
    #                   [0, 2, 0, 2, 0, 0],
    #                   [0, 0, 1, 0, 0, 0]])
    # v.reset_state_goal(STATE, GOAL)

    #

    # GOAL = np.array([[0, 1, 2, 0, 0, 0],
    #                  [0, 1, 2, 0, 0, 0],
    #                  [0, 2, 1, 0, 0, 0]])
    # STATE = np.array([[0, 1, 0, 0, 0, 2],
    #                   [0, 2, 0, 2, 0, 0],
    #                   [0, 0, 1, 0, 0, 1]])
    # v.reset_state_goal(STATE, GOAL)
    # GOAL = np.array([[0, 0, 0, 0, 0, 0],
    #                  [0, 0, 1, 2, 1, 0],
    #                  [0, 0, 0, 0, 0, 0]])
    # STATE = np.array([[0, 1, 0, 0, 0, 0],
    #                   [0, 0, 0, 2, 0, 0],
    #                   [0, 0, 1, 0, 0, 0]])
    # GOAL = np.array([[0, 0, 0, 0, 0, 0],
    #                  [0, 0, 2, 1, 2, 0],
    #                  [0, 0, 0, 0, 0, 0]])
    # STATE = np.array([[0, 2, 0, 0, 0, 0],
    #                   [0, 0, 0, 1, 0, 0],
    #                   [0, 0, 2, 0, 0, 0]])
    # v.reset_state_goal(STATE, GOAL)


    processes.extend([shared_state, eval])
    v.scheduler.set_training_level(ckpnt['training_level'])
    eval.single_test.remote(v,
                            toggle_reset_state=True,
                            toggle_sync_training=False,
                            toggle_show_every_step=False,
                            toggle_reset = True,
                            toggle_visual=True,)
    while True:
        time.sleep(1)


if __name__ == '__main__':
    from huri.learning.method.APEX_DQN.distributed.update_distribution_pool import update_restore

    update_restore()
    try:
        main()
    except:
        for p in processes:
            ray.kill(p)
