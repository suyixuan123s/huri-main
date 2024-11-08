import timeit
import logging
import collections
import queue
import time

import torch
import torch.nn.functional as F

import multiprocessing as mp
from huri.learning.method.AlphaZero.replay import Transition, UniformReplay
from typing import List, Tuple, Mapping, Union, Text, Any, Dict
from huri.learning.method.AlphaZero.log import CsvWriter, write_to_csv
import huri.core.file_sys as fs
from huri.learning.method.AlphaZero.utils import load_checkpoint, create_checkpoint, get_time_stamp, handle_exit_signal, \
    disable_auto_grad
from huri.learning.env.rack_v3 import RackArrangementEnv
from huri.learning.method.AlphaZero.mcts_player import create_mcts_player
import numpy as np
from tqdm import tqdm


def calc_acc_rets(returns: List[float], gamma: float) -> float:
    acc_ret = 0
    for i, ret in enumerate(returns):
        acc_ret += ret * (gamma ** i)
    return acc_ret


def calc_loss(
        network: torch.nn.Module, device: torch.device, transitions: Transition, argument_data: bool = False
) -> torch.Tensor:
    """Compute the AlphaZero loss."""
    # [B, state_shape]
    state = torch.from_numpy(transitions.state).to(device=device, dtype=torch.float32, non_blocking=True)
    # [B, num_actions]
    pi_prob = torch.from_numpy(transitions.pi_prob).to(device=device, dtype=torch.float32, non_blocking=True)
    # [B, ]
    value = torch.from_numpy(transitions.value).to(device=device, dtype=torch.float32, non_blocking=True)
    if any(np.isnan(transitions.value)):
        raise Exception("Dirty Training dat")
    if argument_data:
        # Argument data by apply random rotation and reflection.
        state, pi_prob = random_rotation_and_reflection(state, pi_prob)

    network_out = network(state)

    # value MSE loss
    value_loss = F.mse_loss(network_out.value.squeeze(1), value, reduction='mean')

    # policy cross-entropy loss
    policy_loss = F.cross_entropy(network_out.pi_logits, pi_prob, reduction='mean')

    return policy_loss, value_loss


def run_self_play(
        rank: int,
        network: torch.nn.Module,
        device: torch.device,
        env,
        data_queue: mp.Queue,
        c_puct_base: float,
        c_puct_init: float,
        warm_up_steps: int,
        num_simulations: int,
        num_parallel: int,
        gamma: float,
        reset_num: int,
        stop_event: mp.Event,
        curriculum_info: Dict,
) -> None:
    """Run self-play loop to generate training samples.
    Only stop the loop if `stop_event` is set to True.

    Args:
        rank: the rank of the self-play process.
        network: neural network to evaluate position,
            this is the current best network.
        device: torch runtime device.
        env: a BoardGameEnv type environment.
        data_queue: a multiprocessing.Queue to send samples to training process.
        c_puct: a constant controls the level of exploration during MCTS search.
        warm_up_steps: number of opening environment steps to
            sample action according search policy instead of choosing most visited child.
        num_simulations: number of simulations for each MCTS search.
        num_parallel: Number of parallel leaves for MCTS search.
        stop_event: a multiprocessing.Event that will signal the end of training.

    Raises:
        ValueError:
            if `env` is not a valid BoardGameEnv instance.
    """
    if not isinstance(env, RackArrangementEnv):
        raise ValueError(f'Expect env to be a valid BoardGameEnv instance, got {env}')

    # init_absl_logging()
    handle_exit_signal()
    logging.info(f'Start self-play actor {rank}')
    writer = CsvWriter(f'run/log/self_play_{mp.current_process().pid}.csv')
    disable_auto_grad(network)
    network = network.to(device=device)
    network.eval()

    played_games = 0
    mcts_player = create_mcts_player(
        network=network,
        device=device,
        num_simulations=num_simulations,
        num_parallel=num_parallel,
        root_noise=True,
        deterministic=False,
    )

    steps = 0
    while not stop_event.is_set():
        # For each new game.
        env.scheduler.set_training_level(curriculum_info['level'])
        obs_s = env.reset()
        obs = env.observation()
        done = False

        episode_states = []
        episode_search_pis = []
        episode_values = []
        # player_ids = []

        root_node = None

        # Play and record transitions.
        n_step = 0
        while not done:
            n_step += 1
            temperature = 0.01 if steps >= warm_up_steps else 1.0
            # temperature = 0.01 if env.steps >= warm_up_steps else 1.0
            move, search_pi, root_node = mcts_player(env, root_node, c_puct_base, c_puct_init, temperature)

            episode_states.append(obs)
            episode_search_pis.append(search_pi)
            # player_ids.append(0)
            obs_s, reward, done, _ = env.step(move)
            obs = env.observation()
            episode_values.append(reward)
            if n_step >= reset_num:
                break
        # if reward != 0:
        #     for i, play_id in enumerate(player_ids):
        #         if play_id == env.last_player:
        #             episode_values[i] = reward
        #         else:
        #             episode_values[i] = -reward

        # scale_factor = 16
        episode_returns = []
        for i in range(len(episode_values)):
            episode_returns.append(calc_acc_rets(episode_values[i:], gamma))

        data_queue.put(
            [Transition(state=x, pi_prob=pi, value=v) for x, pi, v in
             zip(episode_states, episode_search_pis, episode_returns)]
        )
        played_games += 1
        steps += 1
        write_to_csv(writer, [
            ('timestamp', get_time_stamp(), '%1s'),
            ('played_games', played_games, '%1d'),
            ('episode_lens', len(episode_values), '%1d'),
        ])
    logging.info(f'Stop self-play actor {rank}')


def run_training(
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
        device: torch.device,
        actor_network: torch.nn.Module,
        replay: UniformReplay,
        data_queue: mp.SimpleQueue,
        batch_size: int,
        num_train_steps: int,
        checkpoint_frequency: int,
        checkpoint_dir: str,
        checkpoint_files: List,
        csv_file: str,
        stop_event: mp.Event,
        delay: float = 0.0,
        train_steps: int = 0,
        log_interval: int = 50,
):
    """Run the main training loop for N iterations, each iteration contains M updates.
    This controls the 'pace' of the pipeline, including when should the other parties to stop.

    Args:
        network: the neural network we want to optimize.
        optimizer: neural network optimizer.
        lr_scheduler: learning rate annealing scheduler.
        device: torch runtime device.
        actor_network: the neural network actors running self-play, for the case AlphaZero pipeline without evaluation.
        replay: a simple uniform experience replay.
        data_queue: a multiprocessing.SimpleQueue instance, only used to signal data collector to stop.
        batch_size: sample batch size during training.
        num_train_steps: total number of training steps to run.
        checkpoint_frequency: the frequency to create new checkpoint.
        checkpoint_dir: create new checkpoint save directory.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint files.
        csv_file: a csv file contains the training statistics.
        stop_event: a multiprocessing.Event signaling other parties to stop running pipeline.
        delay: wait time (in seconds) before start training on next batch samples, default 0.
        train_steps: already trained steps, used when resume training, default 0.
        log_interval: how often to log training statistics, default 1000.

    Raises:
        ValueError:
            if `min_replay_size` less than `batch_size`.
            if `checkpoint_dir` is invalid.
    """

    if not isinstance(checkpoint_dir, str) or checkpoint_dir == '':
        raise ValueError(f'Expect checkpoint_dir to be valid path, got {checkpoint_dir}')

    writer = CsvWriter(csv_file)
    logging.info('Start training thread')
    start = None
    last_train_step = train_steps  # Store train step from last session incase resume training
    disable_auto_grad(actor_network)

    network = network.to(device=device)
    network.train()
    actor_network.eval()

    ckpt_dir = fs.Path(checkpoint_dir)
    if checkpoint_dir is not None and checkpoint_dir != '' and not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def get_state_to_save():
        return {
            'network': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_steps': train_steps,
        }

    while True:
        if replay.num_games_added < 200:
            time.sleep(30)
            continue

        if start is None:
            start = timeit.default_timer()

        # Signaling other parties to stop running pipeline.
        if train_steps >= num_train_steps:
            break

        transitions = replay.sample(batch_size)

        if transitions is None:
            continue

        optimizer.zero_grad()
        policy_loss, value_loss = calc_loss(network, device, transitions, False)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_steps += 1

        if train_steps % checkpoint_frequency == 0:
            state_to_save = get_state_to_save()
            ckpt_file = ckpt_dir / f'train_steps_{train_steps}'
            create_checkpoint(state_to_save, ckpt_file)
            checkpoint_files.append(ckpt_file)
            train_rate = ((train_steps - last_train_step) * batch_size) / (timeit.default_timer() - start)
            logging.info(f'Train step {train_steps}, train sample rate {train_rate:.2f}')

        if train_steps % 10 == 0:
            actor_network.load_state_dict(network.state_dict())
            actor_network.eval()

        if train_steps % log_interval == 0:
            log_output = [
                ('timestamp', get_time_stamp(), '%1s'),
                ('train_steps', train_steps, '%3d'),
                ('policy_loss', policy_loss.detach().item(), '%.4f'),
                ('value_loss', value_loss.detach().item(), '%.4f'),
                ('learning_rate', lr_scheduler.get_last_lr()[0], '%.2f'),
            ]
            write_to_csv(writer, log_output)

        # Wait for sometime before start training on next batch.
        if delay > 0:
            time.sleep(delay)

    stop_event.set()
    time.sleep(60)
    data_queue.put('STOP')


def run_data_collector(
        data_queue: mp.SimpleQueue,
        replay: UniformReplay,
        log_interval: int = 50,  # every 1000 games
) -> None:
    """Collect samples from self-play,
    this runs on the same process as the training loop,
    but with a separate thread.

    Args:
        data_queue: a multiprocessing.SimpleQueue to receive samples from self-play processes.
        replay: a simple uniform random experience replay.
        log_interval: how often to log the statistic, measured in number of games received.

    """
    logging.info('Start data collector thread')
    start = timeit.default_timer()

    game_steps = collections.deque(maxlen=1000)

    while True:
        try:
            item = data_queue.get()
            if item == 'STOP':
                break

            replay.add_game(item)
            game_steps.append(len(item))

            if replay.num_games_added % log_interval == 0:
                sample_gen_rate = replay.num_samples_added / (timeit.default_timer() - start)
                logging.info(
                    f'Collected {replay.num_games_added} self-play games, sample generation rate {sample_gen_rate:.2f}'
                )

        except queue.Empty:
            pass
        except EOFError:
            pass


def run_evaluation(
        old_network: torch.nn.Module,
        new_network: torch.nn.Module,
        device: torch.device,
        env: RackArrangementEnv,
        c_puct_base: float,
        c_puct_init: float,
        temperature: float,
        num_simulations: int,
        num_parallel: int,
        checkpoint_files: List,
        curriculum_info: Dict,
        csv_file: str,
        stop_event: mp.Event,
        eval_num: int,
        reset_num: int,
) -> None:
    """Monitoring training progress by play a single game with new checkpoint against last checkpoint.

    Args:
        old_network: the last checkpoint network.
        new_network: new checkpoint network we want to evaluate.
        device: torch runtime device.
        env: a BoardGameEnv type environment.
        c_puct: a constant controls the level of exploration during MCTS search.
        temperature: the temperature exploration rate after MCTS search
            to generate play policy.
        num_simulations: number of simulations for each MCTS search.
        num_parallel: Number of parallel leaves for MCTS search.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint.
        csv_file: a csv file contains the statistics for the best checkpoint.
        stop_event: a multiprocessing.Event signaling to stop running pipeline.
        initial_elo: initial elo ratings for the players, default 0.

     Raises:
        ValueError:
            if `env` is not a valid BoardGameEnv instance.
    """
    if not isinstance(env, RackArrangementEnv):
        raise ValueError(f'Expect env to be a valid BoardGameEnv instance, got {env}')

    # init_absl_logging()
    handle_exit_signal()
    writer = CsvWriter(csv_file)
    logging.info('Start evaluator')

    disable_auto_grad(old_network)
    disable_auto_grad(new_network)

    old_network = old_network.to(device=device)
    new_network = new_network.to(device=device)

    while True:
        if stop_event.is_set() and len(checkpoint_files) == 0:
            break
        if len(checkpoint_files) == 0:
            continue

        # Remove the checkpoint file path from the shared list.
        while len(checkpoint_files) > 0:
            ckpt_file = checkpoint_files.pop(0)
        loaded_state = load_checkpoint(ckpt_file, device)
        new_network.load_state_dict(loaded_state['network'])
        train_steps = loaded_state['train_steps']

        new_network.eval()
        old_network.eval()

        # Black is the new checkpoint, white is last checkpoint.
        mcts_player = create_mcts_player(
            network=new_network,
            device=device,
            num_simulations=num_simulations,
            num_parallel=num_parallel,
            root_noise=False,
            deterministic=True,
        )

        # TEST
        n_success = 0
        total_len = 0
        cl = curriculum_info['level']
        env.scheduler.set_training_level(cl)
        r = []
        for i in tqdm(range(eval_num)):
            path = []
            env.reset()
            done = False
            n_step = 0
            rewards = []
            while not done:
                n_step += 1
                move, search_pi, root_node = mcts_player(env, None, c_puct_base, c_puct_init, temperature)

                # player_ids.append(0)
                obs_s, reward, done, _ = env.step(move)
                rewards.append(reward)
                path.append(obs_s)
                # steps are out of the limit
                if cl > 3:
                    if n_step >= reset_num:
                        break
                else:
                    if n_step >= cl:
                        break
                # obs = env.observation()
            if done:
                n_success += 1
                total_len += n_step
                r.append(path)
        avg_len = total_len / n_success if n_success != 0 else 0
        logging.info(f"[Evaluation] Number of success "
                     f"{n_success}/{eval_num}, "
                     f"Average Length: "
                     f"{avg_len}, "
                     f"Training Level: {cl}")

        if n_success >= eval_num and total_len / n_success <= (env.scheduler.state_level * 2):
            logging.info(f"[Evaluation] Update Level Training {cl} -> {cl + 1}")
            curriculum_info['level'] += 1
            env.scheduler.set_training_level(curriculum_info['level'])
            fs.dump_pickle(r, "evaluation_r.pkl", reminder=False)

        log_output = [
            ('timestamp', get_time_stamp(), '%1s'),
            ('train_steps', train_steps, '%3d'),
            ('episode_steps', n_step, '%1d'),
            ('training_level', curriculum_info['level'], '%1d'),
        ]
        write_to_csv(writer, log_output)

        # Unlike in AlphaGo Zero, here we always use the latest checkpoint for next evaluation.
        old_network.load_state_dict(new_network.state_dict())
