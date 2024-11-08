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
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed.pipeline import Actor, Eval, SharedState, FIFOBufferActor
from huri.learning.method.APEX_DQN.distributed.learner import Learner
from huri.learning.method.APEX_DQN.distributed.reanalyzer import Reanalyzer
from huri.learning.method.APEX_DQN.distributed.network import RNDModel
from huri.learning.method.APEX_DQN.distributed.network import DDQN2 as DDQN
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from huri.learning.method.AlphaZero.utils import delete_all_files_in_directory
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import random

# config_name = '20230517_5x10_3.yaml'
config_name = '20230517_3x6_2.yaml'


def seed_everything(seed: int):
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def copy_file(src, dst):
    import shutil
    shutil.copy(src, dst)


def init_folder():
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed' / 'run'
    delete_all_files_in_directory(str(save_path))
    save_path.joinpath('log').mkdir(exist_ok=True)
    save_path.joinpath('data').mkdir(exist_ok=True)
    save_path.joinpath('params').mkdir(exist_ok=True)
    # copy config file
    copy_file(fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed' / 'params' / config_name,
              save_path.joinpath('params', 'params.yaml'))
    import_text = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\nfrom typing import NamedTuple, Tuple\n'
    save_path.joinpath('params', 'network.py').write_text(import_text + get_network_source_code(DDQN), encoding='utf-8')
    return save_path


def get_network_source_code(module_function):
    import inspect
    # Get the source code of the module function
    source_code = inspect.getsource(module_function)
    return source_code


@hydra.main(config_path='params', config_name=config_name, version_base='1.3')
def main(cfg):
    delete_all_files_in_directory(str(r'G:\chen\ray_session'))
    ray.init(
        # address='auto',
        # log_to_driver=False,
        # runtime_env={"working_dir": r'E:\learning\.',
        #              "pip": ['shapely',
        #                      'numba',
        #                      'gym',
        #                      'open3d',
        #                      'lz4', ]}
        dashboard_host='0.0.0.0',
        _temp_dir=r"G:\chen\ray_session",
        object_store_memory=30 * 10 ** 9,
    )
    print("Start ... ")
    # setup seed
    seed_everything(cfg['env']['seed'])
    # init saving path
    save_path = init_folder()
    # delete all sesstion files in ray
    # setup environment
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                          num_history=1)
    env_meta_difficult = create_env(rack_sz=cfg['env']['rack_sz'],
                                    num_tube_class=cfg['env']['num_tube_class'],
                                    seed=cfg['env']['seed'],
                                    toggle_curriculum=cfg['env']['toggle_curriculum'],
                                    toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                                    scheduler='GoalRackStateScheduler3',
                                    num_history=1)
    # setup neural network
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    network = DDQN(input_shape,
                   num_actions,
                   num_category=cfg['env']['num_tube_class'],
                   num_filters=cfg['ddqn']['num_filters'],
                   num_res_block=cfg['ddqn']['num_res_block'],
                   num_fc_units=cfg['ddqn']['num_fc_units'], )
    if cfg['rl']['toggle_H_buffer']:
        rnd = RNDModel(input_shape, cfg['env']['num_tube_class'])
    else:
        rnd = None
    # start replay buffer
    alpha = .6
    replay_buffer = ray.remote(PrioritizedReplayBuffer).options(
        num_cpus=3, ).remote(capacity=int(cfg['rl']['replay_sz']), alpha=alpha, )
    if cfg['toggle_replay_buffer_2']:
        replay_buffer2 = ray.remote(PrioritizedReplayBuffer).options(num_cpus=1,
                                                                     ).remote(capacity=int(cfg['rl']['replay_sz']),
                                                                              alpha=alpha)
    else:
        replay_buffer2 = None

    # start shared state
    ckpnt = {'train_steps': 0,
             'weights': network.state_dict(),
             # 'weights': torch.load(
             #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\runs\run_2\data\model_last.chkpt')[
             #     'dqn_state_dict'],
             'training_level': cfg['env']['init_state_level'],
             'trajectory_list': [],
             'eval_average_len': 0, }
    if cfg['rl']['toggle_H_buffer']:
        ckpnt['rnd_model_weights'] = rnd.state_dict()
        shared_state = SharedState.options(num_cpus=1).remote(ckpnt)
        H_buffer = FIFOBufferActor.remote(capacity=cfg['rl']['H_buffer_sz'],
                                          rnd_model=copy.deepcopy(rnd),
                                          num_classes=cfg['env']['num_tube_class'],
                                          shared_state=shared_state,
                                          device=cfg['rl']['device'])
    else:
        H_buffer = None
        shared_state = SharedState.options(num_cpus=1).remote(ckpnt)
    print(ckpnt.keys())
    # start Learner
    learner = Learner.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(
        env_action_space_dim=env_meta.action_space_dim,
        env_classes=cfg['env']['num_tube_class'],
        net=copy.deepcopy(network),
        icm=None,
        rnd_model=rnd,
        cfg=cfg['rl'],
        shared_state=shared_state,
        replay_buffer=replay_buffer,
        replay_buffer2=replay_buffer2,
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
        if replay_buffer2 is not None:
            r_b = replay_buffer if i % 2 == 0 else replay_buffer2
        else:
            r_b = replay_buffer
        # r_b = replay_buffer
        actor = Actor.remote(actor_id=i,
                             env=env,
                             net=copy.deepcopy(network),
                             cfg=cfg['rl'],
                             H_buffer=H_buffer,
                             replay_buffer=r_b,
                             shared_state=shared_state,
                             log_path=save_path.joinpath('log') if i == 0 else None,
                             toggle_visual=one_actor_show,
                             toggle_H_buffer=cfg['rl']['toggle_H_buffer'], )
        if one_actor_show:
            one_actor_show = False
        actor.start.remote()
        actor_procs.append(actor)
    # start reanalyzer
    reanlyz_procs = []
    for i in range(cfg['num_reanalyzer']):
        env = env_meta.copy()
        env.set_seed((100 + i) * cfg['env']['seed'])
        if replay_buffer2 is not None:
            r_b = replay_buffer if i % 2 == 0 else replay_buffer2
        else:
            r_b = replay_buffer
        # r_b = replay_buffer
        reanalyzer = Reanalyzer.remote(uid=i,
                                       env=env,
                                       replay_buffer=r_b,
                                       shared_state=shared_state,
                                       pop_buff_len = cfg['rl']['pop_buff_len'],
                                       log_path=save_path.joinpath('log') if i == 0 else None, )
        reanalyzer.start.remote()
        reanlyz_procs.append(reanalyzer)
        print("Reanalyzer starts")
    # start evaluator
    env = env_meta_difficult.copy()
    env.set_seed(8888 * cfg['env']['seed'])
    eval = Eval.options(
        num_cpus=1,
        num_gpus=.2,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(env=env,
                                                                               net=copy.deepcopy(network),
                                                                               cfg=cfg['eval'],
                                                                               H_buffer=H_buffer,
                                                                               shared_state=shared_state,
                                                                               save_path=save_path.joinpath('data'),
                                                                               log_path=save_path.joinpath('log'),
                                                                               toggle_visual=False)
    eval.start.remote()

    print('Waiting...')

    start_additional_reanalyzer = False

    e_tmp = env_meta.copy()
    buff_len_tmp = 0
    s = time.time()
    while True:
        buff_len1 = ray.get(replay_buffer.__len__.remote())
        if replay_buffer2 is not None:
            buff_len2 = ray.get(replay_buffer2.__len__.remote())
        else:
            buff_len2 = 0
        if H_buffer is not None:
            H_buffer_len = ray.get(H_buffer.len.remote())
        else:
            H_buffer_len = 0
        trajectory_list_len = len(ray.get(shared_state.get_info.remote('trajectory_list')))
        level = ray.get(shared_state.get_info.remote('training_level'))
        e_tmp.scheduler.set_training_level(level)
        print(
            f"Buffer samples: {buff_len1}, {buff_len2}, {(buff_len1 + buff_len2 - buff_len_tmp) / (time.time() - s):.1f}/s H_buffer samples: {H_buffer_len},"
            # f"Buffer samples: {buff_len1}, {(buff_len1 - buff_len_tmp) / (time.time() - s):1f}/s H_buffer samples: {H_buffer_len},"
            f"  Traj wait for update samples: {trajectory_list_len}, Level: {level}"
            f", State Level: {e_tmp.scheduler.state_level}, Class Level: {e_tmp.scheduler.class_level}")
        s = time.time()
        buff_len_tmp = buff_len1 + buff_len2
        # buff_len_tmp = buff_len1
        if e_tmp.scheduler.state_level >= 5 and not start_additional_reanalyzer:
            start_additional_reanalyzer = True
            print("Start Additional reanalyzer! ... ")
            for i in range(cfg['num_reanalyzer'], 3 + cfg['num_reanalyzer']):
                env = env_meta.copy()
                env.set_seed((100 + i) * cfg['env']['seed'])
                if replay_buffer2 is not None:
                    r_b = replay_buffer if i % 2 == 0 else replay_buffer2
                else:
                    r_b = replay_buffer
                reanalyzer = Reanalyzer.remote(uid=i,
                                               env=env,
                                               replay_buffer=r_b,
                                               shared_state=shared_state,
                                               log_path=save_path.joinpath('log') if i == 0 else None, )
                reanalyzer.start.remote()
                reanlyz_procs.append(reanalyzer)
            print("Reanalyzer added! ... ")
        time.sleep(cfg['eval']['eval_interval'] + 3)
        # auto_garbage_collect()


if __name__ == '__main__':
    import os, sys
    from huri.learning.method.APEX_DQN.distributed.update_distribution_pool import update_restore

    # sys.path.append('E:\\huri_shared\\')
    # os.environ['RAY_PROMETHEUS_HOST'] = 'http://127.0.0.1:9090'
    # os.environ['RAY_GRAFANA_HOST'] = 'http://127.0.0.1:3030'
    # os.environ['RAY_PROMETHEUS_NAME'] = 'Prometheus'

    # os.chdir(r'E:\huri_shared')
    # set PYTHONPATH=E:\huri_shared\
    # $env:PYTHONPATH = "E:\huri_shared\"
    # E:\Venv\Scripts\python.exe E:\huri_shared\huri\learning\method\APEX_DQN\distributed\main.py

    # os.system('ray stop')
    # time.sleep(5)
    # os.system('ray start --head')
    # time.sleep(5)
    update_restore()
    main()

# $env:RAY_PROMETHEUS_HOST = "http://127.0.0.1:9090"

#
# & "C:\Program Files\GrafanaLabs\grafana\bin\grafana-server.exe" --config C:\Users\WRS\AppData\Local\Temp\ray\session_2023-11-14_21-09-19_121613_37752\metrics\grafana\grafana.ini
# C:\Users\WRS\AppData\Local\Temp\ray\prom_metrics_service_discovery.json
