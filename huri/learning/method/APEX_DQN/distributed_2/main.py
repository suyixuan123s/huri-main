""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230720osaka
TODO:
    - num classes can be set:
"""

import copy
import time

import ray
import hydra
import torch
import huri.core.file_sys as fs
from huri.learning.env.rack_v3 import create_env2
from huri.learning.env.rack_v3.env2 import RackArrangementEnv
from huri.learning.method.APEX_DQN.distributed_2.pipeline import Actor, Eval, SharedState
from huri.learning.method.APEX_DQN.distributed_2.learner import Learner
from huri.learning.method.APEX_DQN.distributed_2.reanalyzer import Reanalyzer
from huri.learning.method.APEX_DQN.distributed_2.network import DDQN, ICM
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from huri.learning.method.AlphaZero.utils import delete_all_files_in_directory
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_folder():
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed_2' / 'run'
    delete_all_files_in_directory(str(save_path))
    save_path.joinpath('log').mkdir(exist_ok=True)
    save_path.joinpath('data').mkdir(exist_ok=True)
    return save_path


@hydra.main(config_path='params', config_name='20230517_3x6_2.yaml', version_base='1.3')
def main(cfg):
    ray.init(_node_ip_address='100.80.147.16',
             runtime_env={"working_dir": r'E:\learning\.',
                          "pip": ['shapely',
                                  'numba',
                                  'gym',
                                  'open3d',
                                  'lz4', ]})
    print("Start ... ")
    # setup seed
    seed_everything(cfg['env']['seed'])
    # init saving path
    save_path = init_folder()
    # setup environment
    env_meta: RackArrangementEnv = create_env2(rack_sz=cfg['env']['rack_sz'],
                                               num_tube_class=cfg['env']['num_tube_class'],
                                               seed=cfg['env']['seed'],
                                               toggle_curriculum=cfg['env']['toggle_curriculum'],
                                               toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                                               num_history=1)
    env_meta_difficult: RackArrangementEnv = create_env2(rack_sz=cfg['env']['rack_sz'],
                                                         num_tube_class=cfg['env']['num_tube_class'],
                                                         seed=cfg['env']['seed'],
                                                         toggle_curriculum=cfg['env']['toggle_curriculum'],
                                                         toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                                                         scheduler='GoalRackStateScheduler3',
                                                         num_history=1)
    # setup neural network
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    num_heads = env_meta.head_space_dim
    network = DDQN(input_shape,
                   num_actions,
                   num_heads,
                   num_category=cfg['env']['num_tube_class'],
                   num_filters=32,
                   num_res_block=5,
                   num_fc_units=128)
    icm = ICM(input_shape, num_actions, num_filters=10, num_fc_units=128)
    # start replay buffer
    replay_buffer = ray.remote(PrioritizedReplayBuffer).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(capacity=int(cfg['rl']['replay_sz']),
                                                                               alpha=0.6)
    # start shared state
    ckpnt = {'train_steps': 0,
             'weights': network.state_dict(),
             'training_level': cfg['env']['init_state_level'],
             'trajectory_list': []}
    shared_state = SharedState.remote(ckpnt)
    # start Learner
    learner = Learner.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(
        env_action_space_dim=env_meta.action_space_dim,
        env_classes=cfg['env']['num_tube_class'],
        net=copy.deepcopy(network),
        icm=icm,
        cfg=cfg['rl'],
        shared_state=shared_state,
        replay_buffer=replay_buffer,
        log_path=save_path.joinpath('log'), )
    learner.start.remote()
    # start actor
    actor_procs = []
    one_actor_show = False
    num_env_difficult = cfg['num_difficult']
    for i in range(cfg['num_actor']):
        if num_env_difficult > 0:
            env = env_meta_difficult.copy()
            num_env_difficult -= 1
        else:
            env = env_meta.copy()
        env.set_seed((i + 10) * cfg['env']['seed'])
        # if i == 5:
        #     one_actor_show = True
        actor = Actor.remote(actor_id=i,
                             env=env,
                             net=copy.deepcopy(network),
                             cfg=cfg['rl'],
                             replay_buffer=replay_buffer,
                             shared_state=shared_state,
                             log_path=save_path.joinpath('log') if i == 0 else None,
                             toggle_visual=one_actor_show)
        if one_actor_show:
            one_actor_show = False
        actor.start.remote()
        actor_procs.append(actor)
    # start reanalyzer
    reanlyz_procs = []
    for i in range(cfg['num_reanalyzer']):
        env = env_meta.copy()
        env.set_seed((100 + i) * cfg['env']['seed'])
        reanalyzer = Reanalyzer.remote(uid=i,
                                       env=env,
                                       replay_buffer=replay_buffer,
                                       shared_state=shared_state,
                                       log_path=save_path.joinpath('log') if i == 0 else None, )
        reanalyzer.start.remote()
        reanlyz_procs.append(reanalyzer)
    # start evaluator
    env = env_meta_difficult.copy()
    env.set_seed(8888 * cfg['env']['seed'])
    eval = Eval.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(env=env,
                                                                               net=copy.deepcopy(network),
                                                                               cfg=cfg['eval'],
                                                                               shared_state=shared_state,
                                                                               save_path=save_path.joinpath('data'),
                                                                               log_path=save_path.joinpath('log'),
                                                                               toggle_visual=False)
    eval.start.remote()

    print('Waiting...')

    start_additional_reanalyzer = False

    e_tmp = env_meta.copy()

    while True:
        buff_len = ray.get(replay_buffer.__len__.remote())
        trajectory_list_len = len(ray.get(shared_state.get_info.remote('trajectory_list')))
        level = ray.get(shared_state.get_info.remote('training_level'))
        e_tmp.scheduler.set_training_level(level)
        print(f"Buffer samples: {buff_len}, Traj wait for update samples: {trajectory_list_len}, Level: {level}"
              f", State Level: {e_tmp.scheduler.state_level}, Class Level: {e_tmp.scheduler.class_level}")

        if e_tmp.scheduler.state_level >= 9 and not start_additional_reanalyzer:
            start_additional_reanalyzer = True
            print("Start Additional reanalyzer! ... ")
            for i in range(cfg['num_reanalyzer'], 10 + cfg['num_reanalyzer']):
                env = env_meta.copy()
                env.set_seed((100 + i) * cfg['env']['seed'])
                reanalyzer = Reanalyzer.remote(uid=i,
                                               env=env,
                                               replay_buffer=replay_buffer,
                                               shared_state=shared_state,
                                               log_path=save_path.joinpath('log') if i == 0 else None, )
                reanalyzer.start.remote()
                reanlyz_procs.append(reanalyzer)
            print("Reanalyzer added! ... ")
        time.sleep(10)


if __name__ == '__main__':
    import os
    from huri.learning.method.APEX_DQN.distributed_2.update_distribution_pool import update_restore

    os.system('ray stop')
    time.sleep(5)
    os.system('ray start --head')
    update_restore()
    main()
