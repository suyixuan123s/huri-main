import copy
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

#
from huri.learning.env.rack_v2.env import RackArrangementEnv, GOAL_PATTERN_5x10
from huri.learning.method.APEX_DQN.env2.min_tst2.dqn_model import DuelingDQN, DuelingDQNAttentionMini, DuelingDQNMini
from huri.learning.method.APEX_DQN.learner import Learner, HERLearner2
from huri.learning.method.APEX_DQN.actor import Actor, Eval, HERActor
from huri.learning.method.APEX_DQN.replay import SharedReplayBuffer, SharedPERBuffer, HERSharedReplayBuffer, \
    SharedHERPERBuffer
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


def create_agent(obs_dim, act_dim, n_classes, device):
    return DuelingDQNMini(obs_dim=obs_dim, action_dim=act_dim, num_classes=n_classes).to(device)


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument("--step-num", type=int, default=10 ** 8, help="Step number for learning")
    parser.add_argument("--num-actors", type=int, default=7, help="Number of actors")
    parser.add_argument("--reset-num", type=int, default=100, help="Reset number for each episode")
    parser.add_argument("--start-step", type=int, default=2000, help="Start step of training")
    parser.add_argument("--eval-interval", type=int, default=60, help="Evaluation interval")
    parser.add_argument("--batch-size", type=int, default=64, help="training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=.95, help="decay factor")
    parser.add_argument("--max-eps", type=float, default=1, help="epsilon greedy upper bound")
    parser.add_argument("--min-eps", type=float, default=.2, help="epsilon greedy lower bound")
    parser.add_argument("--eps-decay", type=float, default=1e-5, help="epsilon greedy decay rate")
    parser.add_argument("--seed", type=int, default=888, help="seed number")
    parser.add_argument("--replay-size", type=int, default=2 * 10 ** 7, help="replay buffer size")
    parser.add_argument("--update-freq", type=int, default=100, help="target network update frequency")
    parser.add_argument("--env-difficulty", type=int, default=8, help="difficulty of the environment")
    parser.add_argument("--net-init", type=str, default="", help="the initial model for the net")
    parser.add_argument("--buffer-data-size", type=int, default=100, help="Buffer data size")
    args = parser.parse_args()
    # args.net_init = "E:\huri_shared\huri\learning\method\APEX_DQN\env2\min_tst\\transfer_learning_weight.pt"

    mp_manager = mp.Manager()
    # https://pytorch.org/docs/stable/notes/multiprocessing.html
    shared_state = mp_manager.dict()
    shared_state['state_level'] = None
    shared_state['class_level'] = None

    shared_mem = mp.Queue()

    # device
    device = select_device()

    # initialize the environment
    from env_tst import env, num_tube_classes, rack_size, action_space_dim, observation_space_dim

    shared_state['state_level'] = env.scheduler.state_level
    # shared_state['state_level'] = 2
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
    replay_mem = SharedHERPERBuffer(obs_dim=rack_size,
                                    size=args.replay_size,
                                    batch_size=args.batch_size,
                                    device=device,
                                    beta_decay_steps=2000000)

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

    net = create_agent(observation_space_dim, action_space_dim, num_tube_classes, device)
    # TODO bug: When using the multirpocessing, copy the weight to the gpu is a zero operation ( all weights are set to 0)

    shared_net = copy.deepcopy(net)
    shared_net.eval()
    shared_net.share_memory()

    # A learner is started before the Actors so that the shared_state is populated with a Q_state_dict
    learner = HERLearner2(shared_net=shared_net,
                          shared_state=shared_state,
                          shared_replay_memory=replay_mem,
                          lr=args.lr,
                          gamma=args.gamma,
                          env_obs_dim=env.observation_space_dim,
                          env_act_dim=env.action_space_dim,
                          env_n_classes=env.num_classes,
                          env_abs_state_slice=env.abs_state_slice,
                          env_abs_goal_slice=env.abs_goal_slice,
                          env_to_state_func=env.to_state,
                          target_update_freq=args.update_freq,
                          min_replay_mem_size=args.start_step,
                          device=device, )
    learner_proc = mp.Process(target=learner.learn, args=(args.step_num, create_agent, args.net_init))
    learner_proc.start()
    LOGGER.info(f"Learner Process: started")
    time.sleep(.5)
    #  TODO: Test with multiple actors
    actor_procs = []
    actor_params = [(1, 1),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                    (2, 1),
                    (2, 1),
                    (2, 1),
                    (3, 1),
                    (2, 2),
                    (3, 3),
                    (4, 4), ]
    for i in range(args.num_actors):
        env_actor = env.copy()
        env_actor.set_seed(i * args.seed)
        # env_actor.scheduler.state_level = actor_params[i][0]
        # env_actor.scheduler.class_level = actor_params[i][1]
        actor_proc = HERActor(actor_id=i,
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
