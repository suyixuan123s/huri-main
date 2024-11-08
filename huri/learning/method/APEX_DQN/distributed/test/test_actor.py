""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230719osaka

"""
import copy
import time

import ray
import hydra
import huri.core.file_sys as fs
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed.pipeline import Actor, SharedState
from huri.learning.method.APEX_DQN.distributed.network import DDQN2 as DDQN
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@hydra.main(config_path='../params', config_name='20230517_3x6_2.yaml', version_base='1.3')
def main(cfg):
    ray.init(local_mode=True, )
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed' / 'run'
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                          num_history=1)
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    env_meta.scheduler.set_training_level(7)
    network = DDQN(input_shape,
                   num_actions,
                   num_category=cfg['env']['num_tube_class'],
                   num_filters=32,
                   num_res_block=2,
                   num_fc_units=32)
    replay_buffer = ray.remote(PrioritizedReplayBuffer).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(capacity=int(cfg['rl']['replay_sz']),
                                                                               alpha=0.6)
    # cfg['env']['init_state_level']
    ckpnt = {'weights': network.state_dict(),
             'training_level': 9,
             'trajectory_list': []
             }
    shared_state = SharedState.remote(ckpnt)
    actor = Actor.remote(actor_id=1,
                         env=env_meta,
                         net=copy.deepcopy(network),
                         cfg=cfg['rl'],
                         H_buffer=None,
                         replay_buffer=replay_buffer,
                         shared_state=shared_state,
                         log_path=save_path.joinpath('log'),
                         toggle_visual=False)
    actor.start.remote()
    while True:
        buff_len = ray.get(replay_buffer.__len__.remote())
        trajectory_list_len = len(ray.get(shared_state.get_info.remote('trajectory_list')))
        # if buff_len > 1:
        #     a = ray.get(replay_buffer.sample.remote(1, .6))
        #     print(a)
        print(buff_len, trajectory_list_len)
        time.sleep(1)


if __name__ == '__main__':
    from huri.learning.method.APEX_DQN.distributed.update_distribution_pool import update_restore

    update_restore()
    main()
