import os
import huri.learning.method.AlphaZero.utils as utils
import logging
import hydra
import timeit

import numpy as np
import torch
import copy

import threading
from ray.util.multiprocessing import Pool
from huri.learning.method.AlphaZero.network import AlphaZeroNet
from huri.learning.method.AlphaZero.replay import UniformReplay
from huri.learning.env.rack_v3 import create_env
from huri.learning.utils import select_device
from huri.learning.method.AlphaZero.parallel.pipeline_remote import (run_data_collector,
                                                                     run_self_play_remote,
                                                                     run_training,
                                                                     run_evaluation_remote,
                                                                     CheckpointFile,
                                                                     CurriculumInfo,
                                                                     SharedStorage,
                                                                     )
from huri.learning.method.AlphaZero.utils import delete_all_files_in_directory
import ray
from ray.util.queue import Queue


def get_state_to_save(network, optimizer, lr_scheduler, train_steps):
    return {
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'train_steps': train_steps,
    }


@hydra.main(config_path='../../params', config_name='params20230531', version_base='1.3')
def main(cfg):
    # init ray
    ray.init(address='127.0.0.1:6379')

    seed_counter = 0
    device = select_device(device=cfg['device'])
    # init seed
    torch.manual_seed(cfg['torch_seed'])
    random_state = np.random.RandomState(cfg['numpy_seed'])

    # define env
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'])
    # env_meta.scheduler.state_level = 1
    # input_shape =
    # num_actions =
    board = env_meta.rack_size
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim

    # network
    network = AlphaZeroNet(input_shape, num_actions, num_res_block=int(np.sqrt(np.prod(cfg['env']['rack_sz']))),
                           num_filters=32, num_fc_units=64)

    data_queue = Queue()
    shared_storage_worker = SharedStorage.remote()

    # Start self-play actors
    actors = []
    actor_network = copy.deepcopy(network)
    actor_network.eval()
    for i in range(cfg['num_actors']):
        # init seed counter
        actor_env = env_meta.copy(toggle_reset=True)
        seed_counter += 2 ** 7
        actor_env.set_seed(seed=cfg['env']['seed'] + seed_counter)
        args = (
            i,
            actor_network,
            device,
            actor_env,
            data_queue,
            cfg['MCTS']['c_puct_base'],
            cfg['MCTS']['c_puct_init'],
            cfg['MCTS']['warm_up_steps'],
            cfg['MCTS']['num_simulations'],
            cfg['MCTS']['num_parallel'],
            cfg['MCTS']['gamma'],
            cfg['MCTS']['reset_num'],
            curriculum_info,
        )
        actor = run_self_play_remote.remote(*args)
        actors.append(actor)

    for actor in actors:
        ray.get(actor)

    learner.join()
    data_collector.join()
    ray.get(evaluator)


if __name__ == "__main__":
    main()
