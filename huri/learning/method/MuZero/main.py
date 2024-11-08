import datetime
import time

import ray
import torch
import huri.core.file_sys as fs

import trainer
import shared_storage
import self_play
from replay_buffer import ReplayBuffer
from huri.learning.method.MuZero.game.tubeswap import MuZeroConfig, Game

checkpoint = {
    "weights": None,
    "optimizer_state": None,
    "total_reward": 0,
    "muzero_reward": 0,
    "opponent_reward": 0,
    "episode_length": 0,
    "mean_value": 0,
    "training_step": 0,
    "lr": 0,
    "total_loss": 0,
    "value_loss": 0,
    "reward_loss": 0,
    "policy_loss": 0,
    "num_played_games": 0,
    "num_played_steps": 0,
    "num_reanalysed_games": 0,
    "terminate": False,
}

replay_buffer = {}

if __name__ == "__main__":
    ray.init(num_gpus=torch.cuda.device_count())
    config = MuZeroConfig()
    if config.save_model:
        data_path = fs.workdir_learning / "run" / f"alphazero_{datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')}"
        data_path = fs.workdir_learning / "run" / f"alphazero_debug"
        datasets = []
        if not data_path.is_dir():
            print("Create a new path")
            data_path.mkdir()
        data_path_config = data_path.joinpath('config.json')
        data_path_model_best = data_path.joinpath('model_best.pt')
        data_path_model_last = data_path.joinpath('model_last.pt')
        fs.dump_json(config.to_dict, data_path_config, reminder=False)

    # TODO: Manage GPUs
    pass

    # Initialize workers
    # training worker
    training_worker = trainer.Trainer.options(
        num_cpus=0,
        num_gpus=.5,
    ).remote(checkpoint, config)
    # store network weights and some information
    shared_storage_worker = shared_storage.SharedStorage.remote(
        checkpoint,
        config,
    )
    shared_storage_worker.set_info.remote("terminate", False)
    # replay buffer
    replay_buffer_worker = ReplayBuffer.remote(
        checkpoint, replay_buffer, config
    )
    # reanalyse_worker
    pass
    # init actors
    self_play_workers = [
        self_play.SelfPlay.options(
            num_cpus=1,
            num_gpus=.5,
        ).remote(
            checkpoint,
            Game,
            config,
            config.seed + seed,
        )
        for seed in range(config.num_workers)
    ]

    # actor plays
    [
        self_play_worker.remote(
            shared_storage_worker, replay_buffer_worker
        )
        for self_play_worker in self_play_workers
    ]

    # learner learns
    training_worker.continuous_update_weights.remote(
        replay_buffer_worker, shared_storage_worker
    )

    while 1:
        time.sleep(1)
