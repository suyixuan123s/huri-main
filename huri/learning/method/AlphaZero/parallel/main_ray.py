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
                                                                     run_training_remote,
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


@hydra.main(config_path='params', config_name='params20230531', version_base='1.3')
def main(cfg):
    # init ray
    ray.init()

    seed_counter = 0
    delete_all_files_in_directory(r'E:\huri_shared\huri\learning\method\AlphaZero\run')
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

    # Optimizer
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=cfg['optim']['lr'],
                                momentum=cfg['optim']['momentum'],
                                weight_decay=cfg['optim']['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=tuple(cfg['optim']['lr_decay_milestones']),
                                                        gamma=cfg['optim']['lr_decay_gamma'])

    # ---
    # actor_network = copy.deepcopy(network)
    # actor_network.share_memory()
    # old_ckpt_network = copy.deepcopy(network)
    # new_ckpt_network = copy.deepcopy(network)
    # ---

    replay = UniformReplay(cfg['replay']['capacity'], random_state)

    train_steps = 0
    if cfg['load_checkpoint_path'] is not None and os.path.isfile(cfg['load_checkpoint_path']):
        network.to(device=device)
        loaded_state = utils.load_checkpoint(cfg['load_checkpoint_path'], device)
        network.load_state_dict(loaded_state['network'])
        optimizer.load_state_dict(loaded_state['optimizer'])
        lr_scheduler.load_state_dict(loaded_state['lr_scheduler'])
        train_steps = loaded_state['train_steps']

        # actor_network.load_state_dict(loaded_state['network'])
        # old_ckpt_network.load_state_dict(loaded_state['network'])
        # new_ckpt_network.load_state_dict(loaded_state['network'])

        logging.info(f'Loaded state from checkpoint {cfg["load_checkpoint_path"]}')
        logging.info(f'Current state: train steps {train_steps}, learning rate {lr_scheduler.get_last_lr()}')

    # Use the stop_event to signaling actors to stop running.
    # Transfer samples from self-play process to training process.
    data_queue = Queue()
    # A shared list to store most recent new checkpoint file paths.
    checkpoint_file_remote = CheckpointFile.remote()
    curriculum_info = CurriculumInfo.remote()
    curriculum_info.set.remote(1)

    # Start to collect samples generated by self-play actors
    data_collector = threading.Thread(
        target=run_data_collector,
        args=(data_queue, replay),
    )
    data_collector.start()

    # # Start learner
    learner = threading.Thread(
        target=run_training,
        args=(
            network,
            optimizer,
            lr_scheduler,
            device,
            actor_network,
            replay,
            data_queue,
            cfg['batch'],
            cfg['num_train_steps'],
            cfg['checkpoint_frequency'],
            cfg['checkpoint_dir'],
            checkpoint_file_remote,
            cfg['train_csv_file'],
            cfg['train_delay'],
            train_steps,
        ),
    )
    learner.start()

    # # Start evaluator
    evaluation_env = env_meta.copy(toggle_reset=True)
    seed_counter += 2 ** 7
    evaluation_env.set_seed(seed=cfg['env']['seed'] + seed_counter)
    # evaluation_env.scheduler.state_level = cfg['eval']['state_level']
    # evaluation_env.scheduler.state_level = cfg['eval']['class_level']
    args = (
        old_ckpt_network,
        new_ckpt_network,
        device,
        evaluation_env,
        cfg['MCTS']['c_puct_base'],
        cfg['MCTS']['c_puct_init'],
        0.01,
        cfg['MCTS']['num_simulations'],
        cfg['MCTS']['num_parallel'],
        checkpoint_file_remote,
        curriculum_info,
        cfg['eval_csv_file'],
        cfg['eval']['eval_num'],
        cfg['eval']['reset_num'],
    )
    evaluator = run_evaluation_remote.remote(*args)

    # Start self-play actors
    actors = []
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
