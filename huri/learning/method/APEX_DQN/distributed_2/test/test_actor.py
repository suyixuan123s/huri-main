""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230719osaka

"""
import copy
import time

import ray
import hydra
import huri.core.file_sys as fs
from huri.learning.env.rack_v3 import create_env2
from huri.learning.method.APEX_DQN.distributed_2.pipeline import Actor, SharedState
from huri.learning.method.APEX_DQN.distributed_2.network import DDQN
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

@hydra.main(config_path='../params', config_name='20230517_3x6_2.yaml', version_base='1.3')
def main(cfg):
    ray.init(_node_ip_address='100.80.147.16',
             runtime_env={"working_dir": r'E:\learning\.',
                          "pip": ['shapely', 'numba', 'gym', 'open3d', 'lz4']})
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed_2' / 'run'
    env_meta = create_env2(rack_sz=cfg['env']['rack_sz'],
                           num_tube_class=cfg['env']['num_tube_class'],
                           seed=cfg['env']['seed'],
                           toggle_curriculum=cfg['env']['toggle_curriculum'],
                           toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                           num_history=1)
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    num_heads = env_meta.head_space_dim
    network = DDQN(input_shape,
                   num_actions,
                   num_heads,
                   num_category=cfg['env']['num_tube_class'],
                   num_filters=30,
                   num_res_block=10,
                   num_fc_units=128)
    replay_buffer = ray.remote(PrioritizedReplayBuffer).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(capacity=int(cfg['rl']['replay_sz']),
                                                                               alpha=0.6)
    ckpnt = {'train_steps': 0,
             'weights': network.state_dict(),
             'training_level': cfg['env']['init_state_level'],
             'trajectory_list': []}
    shared_state = SharedState.remote(ckpnt)
    actor = Actor.remote(actor_id=1,
                         env=env_meta,
                         net=copy.deepcopy(network),
                         cfg=cfg['rl'],
                         replay_buffer=replay_buffer,
                         shared_state=shared_state,
                         log_path=save_path.joinpath('log'),
                         toggle_visual=True)
    actor.start.remote()
    while True:
        buff_len = ray.get(replay_buffer.__len__.remote())
        trajectory_list_len = len(ray.get(shared_state.get_info.remote('trajectory_list')))
        if buff_len > 1:
            a = ray.get(replay_buffer.sample.remote(1, .6))
            print(a)
        print(buff_len, trajectory_list_len)
        time.sleep(1)


if __name__ == '__main__':
    import os
    from huri.learning.method.APEX_DQN.distributed_2.update_distribution_pool import update_restore
    os.system('ray stop')
    time.sleep(5)
    os.system('ray start --head')
    update_restore()
    main()
