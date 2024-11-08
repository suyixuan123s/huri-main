import os
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
import ray
from ray.util.queue import Empty
import copy
from huri.learning.utils import select_device


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint):
        self.current_checkpoint = copy.deepcopy(checkpoint)

    # def save_checkpoint(self, path=None):
    #     if not path:
    #         path = self.config.results_path / "model.checkpoint"
    #
    #     torch.save(self.current_checkpoint, path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError


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


@ray.remote
class CheckpointFile:
    def __init__(self):
        self._checkpoint = None

    def set(self, v):
        self._checkpoint = v

    def get(self):
        return self._checkpoint


@ray.remote
class CurriculumInfo:
    def __init__(self):
        self._level = None

    def set(self, v):
        self._level = v

    def get(self):
        return self._level


@ray.remote
def run_self_play_remote(
        rank: int,
        network: torch.nn.Module,
        device: torch.device,
        env,
        data_queue: ray.util.queue.Queue,
        c_puct_base: float,
        c_puct_init: float,
        warm_up_steps: int,
        num_simulations: int,
        num_parallel: int,
        gamma: float,
        reset_num: int,
        shared_storage: ray.actor.ActorClass,
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
    # breakpoint()
    print(f'Start self-play actor {rank}')
    writer = CsvWriter(
        fr'E:\huri_shared\huri\learning\method\AlphaZero\run/log/self_play_{mp.current_process().pid}.csv')
    disable_auto_grad(network)
    # if device == 'gpu':
    #     devices = ray.get_gpu_ids()
    #     if rank < 20:
    #         device = select_device(devices[0])
    #     else:
    #         if len(devices) > 0:
    #             device = select_device(devices[1])

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
    train_steps = -1
    while steps < 10000000000000:
        # For each new game.
        train_steps_tmp = ray.get(shared_storage.get_info.remote('train_steps'))
        if train_steps_tmp != train_steps:
            loaded_state = ray.get(shared_storage.get_checkpoint.remote())
            train_steps = loaded_state['train_steps']
            network.load_state_dict(loaded_state['weights'])
            env.scheduler.set_training_level(
                loaded_state['curriculum_level']
            )

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
            ('train_steps', train_steps, '%1d'),
            ('episode_lens', len(episode_values), '%1d'),
        ])
    logging.info(f'Stop self-play actor {rank}')


@ray.remote(num_gpus=1)
def run_training_remote(
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
        device: torch.device,
        replay: UniformReplay,
        data_queue: ray.util.queue.Queue,
        batch_size: int,
        num_train_steps: int,
        checkpoint_frequency: int,
        checkpoint_dir: str,
        shared_storage: ray.actor.ActorClass,
        csv_file: str,
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
    print('Start training thread')
    start = None
    last_train_step = train_steps  # Store train step from last session incase resume training

    network = network.to(device=device)
    network.train()
    # breakpoint()
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
        # breakpoint()
        num_games = ray.get(replay.num_games.remote())
        if num_games < 200:
            time.sleep(30)
            continue
        if start is None:
            start = timeit.default_timer()

        # Signaling other parties to stop running pipeline.
        if train_steps >= num_train_steps:
            break

        transitions = copy.deepcopy(ray.get(replay.sample.remote(batch_size)))

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
            train_rate = ((train_steps - last_train_step) * batch_size) / (timeit.default_timer() - start)
            print(f'Train step {train_steps}, train sample rate {train_rate:.2f}')

        if train_steps % log_interval == 0:
            shared_storage.set_info.remote({
                "weights": {k: v.cpu() for k, v in network.state_dict().items()},
                "train_steps": train_steps,
            })
            # breakpoint()
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

    time.sleep(60)
    data_queue.put('STOP')


@ray.remote
def run_data_collector(
        data_queue: ray.util.queue.Queue,
        replay: UniformReplay,
        log_interval: int = 1000,  # every 1000 games
) -> None:
    """Collect samples from self-play,
    this runs on the same process as the training loop,
    but with a separate thread.

    Args:
        data_queue: a multiprocessing.SimpleQueue to receive samples from self-play processes.
        replay: a simple uniform random experience replay.
        log_interval: how often to log the statistic, measured in number of games received.

    """
    print('Start data collector thread')
    start = timeit.default_timer()

    game_steps = collections.deque(maxlen=1000)

    while True:
        try:
            item = data_queue.get()
            if item == 'STOP':
                break
            # breakpoint()
            num_games = ray.get(replay.add_game.remote(item))
            game_steps.append(len(item))
            if num_games % log_interval == 0:
                sample_gen_rate = num_games / (timeit.default_timer() - start)
                print(
                    f'Collected {num_games} self-play games, sample generation rate {sample_gen_rate:.2f}'
                )
        except Empty:
            pass
        except EOFError:
            pass


@ray.remote
def run_evaluation_remote(
        old_network: torch.nn.Module,
        new_network: torch.nn.Module,
        device: torch.device,
        env: RackArrangementEnv,
        c_puct_base: float,
        c_puct_init: float,
        temperature: float,
        num_simulations: int,
        num_parallel: int,
        shared_storage: ray.actor.ActorClass,
        csv_file: str,
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
    # handle_exit_signal()
    writer = CsvWriter(csv_file)
    logging.info('Start evaluator')

    disable_auto_grad(old_network)
    disable_auto_grad(new_network)

    old_network = old_network.to(device=device)
    new_network = new_network.to(device=device)
    train_steps = 0
    while True:
        train_steps_tmp = ray.get(shared_storage.get_info.remote('train_steps'))
        if train_steps_tmp == train_steps:
            continue

        # Remove the checkpoint file path from the shared list.
        loaded_state = ray.get(shared_storage.get_checkpoint.remote())
        new_network.load_state_dict(loaded_state['weights'])
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
        cl = loaded_state['curriculum_level']
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
        print(f"[Evaluation] Number of success "
              f"{n_success}/{eval_num}, "
              f"Average Length: "
              f"{total_len / n_success}, "
              f"Training Level: {cl}")

        if n_success >= eval_num and total_len / n_success <= (env.scheduler.state_level * 2):
            logging.info(f"[Evaluation] Update Training Level {cl} -> {cl + 1}")
            shared_storage.set_info.remote('curriculum_level', cl + 1)
            env.scheduler.set_training_level(
                ray.get(shared_storage.get_info.remote('curriculum_level'))
            )
            # fs.dump_pickle(r, "evaluation_r.pkl", reminder=False)

        log_output = [
            ('timestamp', get_time_stamp(), '%1s'),
            ('train_steps', train_steps, '%3d'),
            ('episode_steps', n_step, '%1d'),
        ]
        write_to_csv(writer, log_output)

        # Unlike in AlphaGo Zero, here we always use the latest checkpoint for next evaluation.
        old_network.load_state_dict(new_network.state_dict())
