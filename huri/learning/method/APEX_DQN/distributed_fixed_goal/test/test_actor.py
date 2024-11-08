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
from huri.learning.method.APEX_DQN.distributed_fixed_goal.pipeline import Actor, SharedState
from huri.learning.method.APEX_DQN.distributed_fixed_goal.network import DDQN2 as DDQN
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from huri.learning.method.APEX_DQN.distributed_fixed_goal.env import create_fixe_env


@hydra.main(config_path='../params', config_name='10x10x3.yaml', version_base='1.3')
def main(cfg):
    ray.init(local_mode=True, )
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed' / 'run'
    env_meta = create_fixe_env(rack_sz=cfg['env']['rack_sz'],
                               goal_pattern=cfg['env']['goal_pattern'],
                               num_tube_class=cfg['env']['num_tube_class'],
                               seed=cfg['env']['seed'],
                               toggle_curriculum=cfg['env']['toggle_curriculum'],
                               num_history=1)
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    env_meta.scheduler.set_training_level(70)
    network = DDQN(input_shape,
                   num_actions,
                   num_category=cfg['env']['num_tube_class'],
                   num_filters=cfg['ddqn']['num_filters'],
                   num_res_block=cfg['ddqn']['num_res_block'],
                   num_fc_units=cfg['ddqn']['num_fc_units'],
                   num_out_cnn_layers=cfg['ddqn']['num_out_cnn_layers']
                   )
    replay_buffer = ray.remote(PrioritizedReplayBuffer).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(capacity=10,
                                                                               alpha=0.6)
    # cfg['env']['init_state_level']
    ckpnt = {'weights': network.state_dict(),
             'training_level': 60,
             }
    cfg['rl']['eps_max'] = .2
    ckpnt_traj_list = {'trajectory_list': [], }
    shared_traj_buffer = SharedState.options(num_cpus=1).remote(ckpnt_traj_list)
    shared_state = SharedState.remote(ckpnt)
    actor = Actor.remote(actor_id=1,
                         env=env_meta,
                         net=copy.deepcopy(network),
                         cfg=cfg['rl'],
                         H_buffer=None,
                         replay_buffer=replay_buffer,
                         shared_traj_buffer = shared_traj_buffer,
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
    main()
