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
from huri.learning.method.APEX_DQN.distributed_fixed_goal.env import create_fixe_env
from huri.learning.method.APEX_DQN.distributed_fixed_goal.pipeline import Actor, Eval, SharedState, FIFOBufferActor
from huri.learning.method.APEX_DQN.distributed_fixed_goal.learner import Learner
from huri.learning.method.APEX_DQN.distributed_fixed_goal.reanalyzer import Reanalyzer
# from huri.learning.method.APEX_DQN.distributed_fixed_goal.network import RNDModel
from huri.learning.method.APEX_DQN.distributed_fixed_goal.network import DDQN2 as DDQN
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from huri.learning.method.AlphaZero.utils import delete_all_files_in_directory
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import random

# config_name = '20230517_5x10_3.yaml'

# config_name = '20230517_5x10_3.yaml'
# config_name = '5x10_5_no_A_star_refine.yaml'
# config_name = '5x10_5_no_A_star_completer.yaml'
# config_name = '5x10_5_apex.yaml'
# config_name = '5x10_5_no_curriculum.yaml'
# config_name = '5x10_5_A_15.yaml'
config_name = '3x5_3.yaml'
# config_name = '10x10x3.yaml'
# config_name = '5x10_5.yaml'
# config_name = '5x10_5_no_A_star_refine.yaml'
# config_name = '5x10_5_no_A_star_completer.yaml'
# config_name = '5x10_5_no_reanalyzer.yaml'

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
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed_fixed_goal' / 'run'
    delete_all_files_in_directory(str(save_path))
    save_path.joinpath('log').mkdir(exist_ok=True)
    save_path.joinpath('data').mkdir(exist_ok=True)
    save_path.joinpath('params').mkdir(exist_ok=True)
    # copy config file
    copy_file(fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed_fixed_goal' / 'params' / config_name,
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
    delete_all_files_in_directory(str(r'E:\chen\ray_session'))
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
        _temp_dir=r"E:\chen\ray_session",
        object_store_memory=30 * 10 ** 9,
    )
    print("cfg information: ", cfg)
    print("Start ... ")
    # setup seed
    seed_everything(cfg['env']['seed'])
    # init saving path
    save_path = init_folder()
    # delete all sesstion files in ray
    # setup environment
    env_meta = create_fixe_env(rack_sz=cfg['env']['rack_sz'],
                               goal_pattern=cfg['env']['goal_pattern'],
                               num_tube_class=cfg['env']['num_tube_class'],
                               seed=cfg['env']['seed'],
                               toggle_curriculum=cfg['env']['toggle_curriculum'],
                               num_history=1)
    # setup neural network
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    network = DDQN(input_shape,
                   num_actions,
                   num_category=cfg['env']['num_tube_class'],
                   num_filters=cfg['ddqn']['num_filters'],
                   num_res_block=cfg['ddqn']['num_res_block'],
                   num_fc_units=cfg['ddqn']['num_fc_units'],
                   num_out_cnn_layers=cfg['ddqn']['num_out_cnn_layers']
                   )
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
             'eval_average_len': 0, }
    print(f"Init training level is {ckpnt['training_level']},")
    ckpnt_traj_list = {'trajectory_list': [],
                       'failed_state': []}
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
    shared_traj_buffer = SharedState.options(num_cpus=1).remote(ckpnt_traj_list)
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
    # start actor
    actor_procs = []
    one_actor_show = False
    num_env_difficult = cfg['num_difficult']
    for i in range(cfg['num_actor']):
        if num_env_difficult > 0:
            env = env_meta.copy()
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
                             shared_traj_buffer=shared_traj_buffer,
                             log_path=save_path.joinpath('log') if i == 0 else None,
                             toggle_visual=one_actor_show,
                             toggle_H_buffer=cfg['rl']['toggle_H_buffer'], )
        if one_actor_show:
            one_actor_show = False
        actor_procs.append(actor)

    # print("warmup start..")
    # [actor_procs[i].warmup.remote() for i in range(1, max(len(actor_procs), 5))]
    # v = ray.get(actor_procs[0].warmup.remote())
    # print("warmup finished..")

    print("start learner")
    learner.start.remote()
    print("start actor")
    [actor.start.remote() for actor in actor_procs]

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
                                       shared_traj_buffer=shared_traj_buffer,
                                       pop_buff_len=cfg['rl']['pop_buff_len'],
                                       toggle_visual=cfg['toggle_reanalyzer_debug'],
                                       toggle_completer=cfg['rl']['toggle_completer'],
                                       toggle_refiner=cfg['rl']['toggle_refiner'],
                                       log_path=save_path.joinpath('log'), )
        reanalyzer.start.remote()
        reanlyz_procs.append(reanalyzer)
        print("Reanalyzer starts")
    # start evaluator
    env = env_meta.copy()
    env.set_seed(8888 * cfg['env']['seed'])
    eval = Eval.options(
        num_cpus=1,
        num_gpus=.2,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                           soft=False)).remote(env=env,
                                                                               net=copy.deepcopy(network),
                                                                               cfg=cfg['eval'],
                                                                               H_buffer=H_buffer,
                                                                               shared_traj_buffer=shared_traj_buffer,
                                                                               shared_state=shared_state,
                                                                               save_path=save_path.joinpath('data'),
                                                                               log_path=save_path.joinpath('log'),
                                                                               toggle_visual=False)
    print("start eval")
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
        trajectory_list_len = len(ray.get(shared_traj_buffer.get_info.remote('trajectory_list')))
        level = ray.get(shared_state.get_info.remote('training_level'))
        e_tmp.scheduler.set_training_level(level)
        print(
            f"Buffer samples: {buff_len1}, {buff_len2}, {(buff_len1 + buff_len2 - buff_len_tmp) / (time.time() - s):.1f}/s H_buffer samples: {H_buffer_len},"
            # f"Buffer samples: {buff_len1}, {(buff_len1 - buff_len_tmp) / (time.time() - s):1f}/s H_buffer samples: {H_buffer_len},"
            f"  Traj wait for update samples: {trajectory_list_len}, Level: {level}"
            f", State Level: {e_tmp.scheduler.state_level}, "
            f"Class Level: {e_tmp.scheduler.class_level}, "
            f"Evolve Level: {e_tmp.scheduler.evolve_level if hasattr(e_tmp.scheduler, 'evolve_level') else None}"
            f"if reanalyzer should starts: {level >= cfg['num_reanalyzer_added_level']}")
        s = time.time()
        buff_len_tmp = buff_len1 + buff_len2
        # buff_len_tmp = buff_len1
        if level >= cfg['num_reanalyzer_added_level'] and not start_additional_reanalyzer:
            start_additional_reanalyzer = True
            print("Start Additional reanalyzer! ... ")
            for i in range(cfg['num_reanalyzer'], cfg['num_reanalyzer_added'] + cfg['num_reanalyzer']):
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
                                               pop_buff_len=cfg['rl']['pop_buff_len'],
                                               shared_traj_buffer=shared_traj_buffer,
                                               log_path=save_path.joinpath('log'),
                                               toggle_completer=cfg['rl']['toggle_completer'],
                                               toggle_refiner =cfg['rl']['toggle_refiner'],
                                               )
                reanalyzer.start.remote()
                reanlyz_procs.append(reanalyzer)
            print("Reanalyzer added! ... ")
        time.sleep(cfg['eval']['eval_interval'] + 3)
        # auto_garbage_collect()


if __name__ == '__main__':
    main()
