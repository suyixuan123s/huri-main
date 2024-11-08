""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230720osaka

"""
import torch

""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230720osaka

"""

import copy
import time

import ray
import hydra
import huri.core.file_sys as fs
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed.pipeline import Eval, SharedState
from huri.learning.method.APEX_DQN.distributed.network import DDQN
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@hydra.main(config_path='../params', config_name='20230517', version_base='1.3')
def main(cfg):
    ray.init(_node_ip_address='100.80.147.16',
             runtime_env={"working_dir": r'E:\learning\.',
                          "pip": ['shapely', 'numba', 'gym', 'open3d', 'lz4']})
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed' / 'run'
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                          num_history=1)
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    network = DDQN(input_shape, num_actions, num_filters=10, num_fc_units=128)
    ckpnt = {'train_steps': 1,
             'weights':
                 torch.load(r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\exp3\level_20\data\model_best.chkpt')[
                     'dqn_state_dict'],
             'training_level': 20,
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
                                                                               save_path=save_path.joinpath('data'),
                                                                               log_path=None,
                                                                               toggle_visual=True)
    eval.start.remote()
    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
