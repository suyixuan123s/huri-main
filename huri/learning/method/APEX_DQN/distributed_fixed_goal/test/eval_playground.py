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
from huri.learning.method.APEX_DQN.distributed_fixed_goal.env import create_fixe_env
from huri.learning.method.APEX_DQN.distributed_fixed_goal.pipeline import Eval, SharedState
from huri.learning.method.APEX_DQN.distributed_fixed_goal.network import DDQN2 as DDQN
from huri.learning.method.APEX_DQN.distributed_fixed_goal.network_old import DDQN as DDQN2
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import numpy as np

# config_path = '../params'
# config_name = '20230517_5x10.yaml'
# model_path = r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\run\data\model_best_55.chkpt'
# ----
# config_path = r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\example1\run5x5\params'
# config_name = 'params.yaml'
# model_path = r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\example1\run5x5\data\model_best_55.chkpt'
# ---
config_path = r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\success_train\5x10x3\params'
config_name = 'params.yaml'
model_path = r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\success_train\5x10x3\data\model_best_73.chkpt'


@hydra.main(config_path=config_path, config_name=config_name, version_base='1.3')
def main(cfg):
    ray.init(local_mode=True, )
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed_fixed_goal' / 'test' / 'debug_data'
    env_meta = create_fixe_env(rack_sz=cfg['env']['rack_sz'],
                               goal_pattern=cfg['env']['goal_pattern'],
                               num_tube_class=cfg['env']['num_tube_class'],
                               seed=cfg['env']['seed'],
                               toggle_curriculum=cfg['env']['toggle_curriculum'],
                               scheduler='RackStateScheduler',
                               num_history=1)
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    network = DDQN(input_shape,
                   num_actions,
                   num_category=cfg['env']['num_tube_class'],
                   num_filters=cfg['ddqn']['num_filters'],
                   num_res_block=cfg['ddqn']['num_res_block'],
                   num_fc_units=cfg['ddqn']['num_fc_units'],
                   num_out_cnn_layers=cfg['ddqn']['num_out_cnn_layers'], )

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
                     model_path)[
                     # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\example1\run_good\data\model_best_45.chkpt')[
                     # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\exp3\run7\data\model_best_37.chkpt')[
                     # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\exp2\run\data\model_best_12.chkpt')[
                     # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\exp2\run\data\model_best_12.chkpt')[
                     'dqn_state_dict'],
             'training_level': 61,
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
                                                                               shared_traj_buffer=None,
                                                                               H_buffer=None,
                                                                               save_path=save_path.joinpath('data'),
                                                                               log_path=None,
                                                                               toggle_visual=True)

    v = env_meta.copy()

    # processes.extend([shared_state, eval])
    # v.scheduler.set_training_level(ckpnt['training_level'])
    # v.scheduler.set_training_level(35)

    # v.reset_state(np.array([[1, 1, 1, 2, 1, 2, 0, 1, 0, 0],
    #                         [0, 0, 2, 1, 0, 0, 0, 0, 0, 0],
    #                         [3, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    #                         [0, 0, 3, 0, 0, 2, 1, 3, 0, 0],
    #                         [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    v.scheduler.state_level = 30
    print(repr(v.reset().state))
    print(f"State level: {v.scheduler.state_level} | Class level: {v.scheduler.class_level} | "
          f"Evolve level: {v.scheduler.evolve_level if hasattr(v.scheduler, 'evolve_level') else 'None'}")
    eval.single_test.remote(v,
                            toggle_reset_state=True,
                            toggle_sync_training=False,
                            toggle_show_every_step=False,
                            toggle_reset=False,
                            toggle_show_action_value=False,
                            toggle_visual=False, )
    while True:
        time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except:
        for p in processes:
            ray.kill(p)
