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
from huri.learning.method.APEX_DQN.distributed.pipeline import Actor, SharedState
from huri.learning.method.APEX_DQN.distributed.reanalyzer import Reanalyzer
from huri.learning.method.APEX_DQN.distributed.network import DDQN
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer


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
    replay_buffer = ray.remote(PrioritizedReplayBuffer).remote(capacity=10,
                                                               alpha=0.6)
    ckpnt = {'weights': network.state_dict(),
             'training_level': 10,
             'trajectory_list': []
             }
    cfg['rl']['eps_decay'] = 0
    shared_state = SharedState.remote(ckpnt)
    actor = Actor.remote(actor_id=1,
                         env=env_meta,
                         net=copy.deepcopy(network),
                         cfg=cfg['rl'],
                         replay_buffer=replay_buffer,
                         shared_state=shared_state,
                         log_path=save_path.joinpath('log'),
                         toggle_visual=False)
    reanalyzer_list = []
    for i in range(1):
        reanalyzer = Reanalyzer.remote(uid=i,
                                       env=env_meta.copy(),
                                       replay_buffer=replay_buffer,
                                       shared_state=shared_state,
                                       log_path=save_path.joinpath('log'), )

        reanalyzer.start.remote()
        reanalyzer_list.append(reanalyzer)

    actor.start.remote()
    while True:
        buff_len = ray.get(replay_buffer.__len__.remote())
        trajectory_list_len = len(ray.get(shared_state.get_info.remote('trajectory_list')))
        # if buff_len > 1:
        #     a = ray.get(replay_buffer.sample.remote(1, .6))
        #     print(a)
        print(buff_len, trajectory_list_len)
        time.sleep(.5)


if __name__ == '__main__':
    main()
