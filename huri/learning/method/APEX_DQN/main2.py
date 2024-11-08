import copy
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

#
from huri.learning.env.rack_v1.env import RackArrangementEnv, GOAL_PATTERN_5x10
from huri.learning.network.d3qn_attention import DuelingDQN3
from learner import Learner
from actor import Actor, Eval
from replay import SharedReplayBuffer, SharedPERBuffer
from huri.learning.utils import select_device, LOGGER

import huri.core.file_sys as fs
import datetime


#
# import os
#
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def add_experience_to_replay_mem(shared_mem, replay_mem, batch_data_len, toggle_data=False):
    while 1:
        if toggle_data:
            print(replay_mem.size)
            print(shared_mem.qsize())
        if shared_mem.qsize() >= batch_data_len:
            for _ in range(batch_data_len):
                batch = shared_mem.get()
                replay_mem.store(*batch)
            # while shared_mem.qsize() or not shared_mem.empty():
            #     batch = shared_mem.get()
            #     replay_mem.store(*batch)


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument("--step-num", type=int, default=10 ** 8, help="Step number for learning")
    parser.add_argument("--num-actors", type=int, default=3, help="Number of actors")
    parser.add_argument("--reset-num", type=int, default=100, help="Reset number for each episode")
    parser.add_argument("--start-step", type=int, default=2000, help="Start step of training")
    parser.add_argument("--eval-interval", type=int, default=60, help="Evaluation interval")
    parser.add_argument("--batch-size", type=int, default=64, help="training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=.95, help="decay factor")
    parser.add_argument("--max-eps", type=float, default=1, help="epsilon greedy upper bound")
    parser.add_argument("--min-eps", type=float, default=.2, help="epsilon greedy lower bound")
    parser.add_argument("--eps-decay", type=float, default=3e-6, help="epsilon greedy decay rate")
    parser.add_argument("--seed", type=int, default=888, help="seed number")
    parser.add_argument("--replay-size", type=int, default=10 ** 7, help="replay buffer size")
    parser.add_argument("--update-freq", type=int, default=100, help="target network update frequency")
    parser.add_argument("--env-difficulty", type=int, default=8, help="difficulty of the environment")
    parser.add_argument("--net-init", type=str, default="", help="the initial model for the net")
    parser.add_argument("--buffer-data-size", type=int, default=1000, help="Buffer data size")
    args = parser.parse_args()
    # args.net_init = 'E:\chen\huri_shared\huri\learning\method\APEX_DQN\transfer_learning_weight.pt'

    mp_manager = mp.Manager()
    # https://pytorch.org/docs/stable/notes/multiprocessing.html
    shared_state = mp_manager.dict()
    shared_state['state_level'] = None
    shared_state['class_level'] = None

    shared_mem = mp.Queue()

    # device
    device = select_device()

    # initialize the environment
    num_tube_classes = 2
    rack_size = (5, 10)
    action_space_dim = np.prod(rack_size) ** 2
    observation_space_dim = (2 * rack_size[0], rack_size[1])
    observation_space_dim_nn = (1, *rack_size)
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_tube_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=True,
                             is_goalpattern_fixed=True,
                             seed=args.seed)
    env.set_goal_pattern(GOAL_PATTERN_5x10)

    shared_state['state_level'] = env.scheduler.state_level
    shared_state['class_level'] = env.scheduler.class_level

    # env.goal_pattern = np.array([[1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    #                              [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    #                              [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    #                              [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    #                              [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]])

    # shared replay memory
    # shared_data = gen_replay_buffer_shared_memo(obs_dim=env.observation_space_dim, size=args.replay_size)
    # replay_mem = ReplayBuffer(shared_data)
    # replay_mem = SharedReplayBuffer(obs_dim=env.observation_space_dim,
    #                                 size=args.replay_size,
    #                                 batch_size=args.batch_size,
    #                                 device=device)
    replay_mem = SharedPERBuffer(obs_dim=env.observation_space_dim,
                                 size=args.replay_size,
                                 batch_size=args.batch_size,
                                 device=device)

    data_path = fs.workdir_learning / "run" / f"dqn_{datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')}"
    data_path = fs.workdir_learning / "run" / f"dqn_debug"
    datasets = []
    if not data_path.is_dir():
        print("Create a new path")
        data_path.mkdir()
    data_path_config = data_path.joinpath('config.json')
    data_path_model_best = data_path.joinpath('model_best.pt')
    data_path_model_last = data_path.joinpath('model_last.pt')
    fs.dump_json(args.__dict__, data_path_config, reminder=False)

    # net
    net = DuelingDQN3(obs_dim=observation_space_dim_nn, action_dim=action_space_dim * 2,
                      num_classes=num_tube_classes).to(
        device)
    # TODO bug: When using the multirpocessing, copy the weight to the gpu is a zero operation ( all weights are set to 0)

    shared_net = copy.deepcopy(net)
    shared_net.eval()
    shared_net.share_memory()

    # A learner is started before the Actors so that the shared_state is populated with a Q_state_dict
    learner = Learner(net=copy.deepcopy(net),
                      shared_net=shared_net,
                      shared_state=shared_state,
                      shared_replay_memory=replay_mem,
                      lr=args.lr,
                      gamma=args.gamma,
                      env_act_dim=env.action_space_dim,
                      env_to_state_func=env.to_state,
                      target_update_freq=args.update_freq,
                      min_replay_mem_size=args.start_step,
                      device=device, )
    learner_proc = mp.Process(target=learner.learn, args=(args.step_num, args.net_init))
    learner_proc.start()
    LOGGER.info(f"Learner Process: started")
    time.sleep(.5)
    #  TODO: Test with multiple actors
    actor_procs = []
    for i in range(args.num_actors):
        env_actor = env.copy()
        env_actor.set_seed(i * args.seed)
        actor_proc = Actor(actor_id=i,
                           net=copy.deepcopy(net),
                           env=env_actor,
                           batch_size=args.batch_size,
                           epsilon_decay=args.eps_decay,
                           max_epsilon=args.max_eps,
                           min_epsilon=args.min_eps,
                           reset_num=args.reset_num,
                           target_update_freq=args.update_freq,
                           shared_net=shared_net,
                           shared_state=shared_state,
                           shared_mem=shared_mem,
                           device=device,
                           toggle_visual=False)
        actor_proc.start()
        LOGGER.info(f"Actor {i} Process: started")
        actor_procs.append(actor_proc)

        # Run a routine in a separate proc to fetch/pre-fetch shared_replay_mem onto the ReplayBuffer for learner's use
    replay_mem_proc = mp.Process(target=add_experience_to_replay_mem,
                                 args=(shared_mem, replay_mem, args.buffer_data_size * args.num_actors, False))
    replay_mem_proc.start()

    env_test = env.copy()
    env_test.seed(2 ** 31 - args.seed)
    eval_proc = Eval(net=copy.deepcopy(net),
                     env_test=env_test,
                     reset_num=5,
                     eval_num=300,
                     eval_interval=args.eval_interval,
                     shared_net=shared_net,
                     shared_state=shared_state,
                     device=device,
                     inc_diff_threshold=48,
                     toggle_visual=False,
                     save_model_path_best=data_path_model_best,
                     save_model_path_last=data_path_model_last)
    eval_proc.start()
    LOGGER.info(f"Evaluation Process: started")

    # learner.learn(args.step_num)

    # learner_proc.join()
    [actor_proc.join() for actor_proc in actor_procs]
    replay_mem_proc.join()
    eval_proc.join()
