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
from huri.learning.method.APEX_DQN.distributed.learner import Learner
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
    network = DDQN(input_shape, num_actions,
                   num_category=cfg['env']['num_tube_class'],
                   num_filters=10,
                   num_fc_units=128)
    # icm = ICM(input_shape, num_actions, num_filters=10, num_fc_units=128)
    replay_buffer = ray.remote(PrioritizedReplayBuffer).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(capacity=int(cfg['rl']['replay_sz']),
                                                                               alpha=0.6)
    replay_buffer2 = ray.remote(PrioritizedReplayBuffer).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(capacity=int(cfg['rl']['replay_sz']),
                                                                               alpha=0.6)
    ckpnt = {'weights': network.state_dict(),
             'training_level': cfg['env']['init_state_level'],
             'trajectory_list': []
             }
    cfg['rl']['eps_decay'] = .1
    shared_state = SharedState.remote(ckpnt)

    traj_list = fs.load_pickle('debug_traj_list2.pkl')
    print(len(traj_list))
    for i in range(50):
        [replay_buffer.add.remote(traj[:]) for traj in traj_list]
    print("Finished loading trajectories.", ray.get(replay_buffer.__len__.remote()))

    learner = Learner.remote(env_action_space_dim=env_meta.action_space_dim,
                             env_classes=cfg['env']['num_tube_class'],
                             net=copy.deepcopy(network),
                             icm=None,
                             cfg=cfg['rl'],
                             shared_state=shared_state,
                             rnd_model=None,
                             replay_buffer2=None,
                             replay_buffer=replay_buffer,
                             log_path=save_path.joinpath('log'), )
    learner.start.remote()
    while True:
        buff_len1 = ray.get(replay_buffer.__len__.remote())
        buff_len2 = ray.get(replay_buffer2.__len__.remote())
        trajectory_list_len = len(ray.get(shared_state.get_info.remote('trajectory_list')))
        if buff_len1 > 1:
            a = ray.get(replay_buffer.sample.remote(1, .6))
            print(a)
        print(buff_len1, buff_len2, trajectory_list_len)
        time.sleep(20)


if __name__ == '__main__':
    from huri.learning.method.APEX_DQN.distributed.update_distribution_pool import update_restore

    update_restore()
    main()
