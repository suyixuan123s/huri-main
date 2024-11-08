import os
import utils
import logging
import hydra
import timeit

import numpy as np
import torch
import copy

import threading
import multiprocessing as mp
from network import AlphaZeroNet
from replay import UniformReplay
from huri.learning.env.rack_v3 import create_env, RackStatePlot
from huri.learning.utils import select_device
from pipeline import (run_data_collector,
                      run_self_play,
                      run_training,
                      run_evaluation)
from mcts_player import create_mcts_player
from utils import load_checkpoint, create_checkpoint, get_time_stamp, handle_exit_signal, disable_auto_grad
from huri.components.utils.img_utils import combine_images
import cv2


def get_state_to_save(network, optimizer, lr_scheduler, train_steps):
    return {
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'train_steps': train_steps,
    }


@hydra.main(config_path='params', config_name='params20230707', version_base='1.3')
def eval(cfg):
    toggle_show = True
    device = select_device(device=cfg['device'])
    # init seed
    torch.manual_seed(cfg['torch_seed'])
    random_state = np.random.RandomState(cfg['numpy_seed'])

    # define env
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          # num_history=cfg['env']['num_obs_history'],
                          num_history=4,
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'])
    # env_meta.scheduler.state_level = 1
    # input_shape =
    # num_actions =
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim

    # network
    network = AlphaZeroNet(input_shape, num_actions, num_res_block=int(np.sqrt(np.prod(cfg['env']['rack_sz']))),
                           num_filters=64, num_fc_units=128)
    # network.load_state_dict(torch.load('./pretrained/test_encoder.pt'))
    network.to(device=device)
    cfg['load_checkpoint_path'] = r'E:\huri_shared\huri\learning\method\AlphaZero\parallel\test\run\checkpoints\train_steps_700'
    eval_level = 3

    if cfg['load_checkpoint_path'] is not None and os.path.isfile(cfg['load_checkpoint_path']):
        loaded_state = utils.load_checkpoint(cfg['load_checkpoint_path'], device)
        network.load_state_dict(loaded_state['network'])
        logging.info(f'Loaded state from checkpoint {cfg["load_checkpoint_path"]}')

    # # Start evaluator
    env = env_meta.copy(toggle_reset=True)
    evaluation_env = env_meta.copy(toggle_reset=True)
    seed_counter = 2 ** 7
    evaluation_env.set_seed(seed=cfg['env']['seed'] + seed_counter)
    env = evaluation_env
    logging.info('Start evaluator')
    # Remove the checkpoint file path from the shared list.
    disable_auto_grad(network)
    network.eval()

    # Black is the new checkpoint, white is last checkpoint.
    mcts_player = create_mcts_player(
        network=network,
        device=device,
        num_simulations=cfg['MCTS']['num_simulations'],
        num_parallel=cfg['MCTS']['num_parallel'],
        root_noise=False,
        deterministic=True,
    )

    # TEST
    n_success = 0
    avg_len = 0
    temperature = .01
    eval_num = cfg['eval']['eval_num']
    eval_num = 1000
    env.scheduler.set_training_level(eval_level)
    cfg['eval']['reset_num'] = 5
    for i in range(eval_num):
        env.reset()
        done = False
        n_step = 0
        rewards = []
        if toggle_show:
            rsp = RackStatePlot(env.goal_pattern, )
            plot = rsp.plot_states([env.state]).get_img()
            img_list = [plot]
            cv2.imwrite('result.jpg', plot)
        while not done:
            n_step += 1
            move, search_pi, root_node = mcts_player(env,
                                                     None,
                                                     cfg['MCTS']['c_puct_base'],
                                                     cfg['MCTS']['c_puct_init'],
                                                     temperature)

            # player_ids.append(0)
            obs_s, reward, done, _ = env.step(move)
            rewards.append(reward)
            # steps are out of the limit
            # print(n_step)
            if n_step >= cfg['eval']['reset_num']:
                break
            obs = env.observation()
            if toggle_show:
                img_list.append(rsp.plot_states([obs_s]).get_img())
                plot = combine_images(img_list, columns=20)
                cv2.imshow(f"plot", plot)
                cv2.waitKey(100)
        if done:
            if toggle_show:
                cv2.waitKey(0)
            n_success += 1
            avg_len += n_step
    if toggle_show:
        cv2.imwrite('result.jpg', plot)
    logging.info(f"[Evaluation] Number of success "
                 f"{n_success}/{eval_num}, "
                 f"Average Length: "
                 f"{avg_len / n_success}, ")


if __name__ == "__main__":
    eval()
