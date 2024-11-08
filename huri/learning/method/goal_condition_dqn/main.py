import numpy as np
from huri.learning.method.goal_condition_dqn.dqn_r import DQNAgent_r as DQNAgent
from huri.learning.network.d3qn_goal_condition import DuelingDQN
from huri.learning.env.arrangement_planning_rack_goal_condition.env import RackArrangementEnv
from argparse import ArgumentParser
import logging
import huri.core.file_sys as fs
import datetime
from pathlib import Path
import torch

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

if __name__ == "__main__":
    # DQN agent parameters
    parser = ArgumentParser()
    parser.add_argument("--step-num", type=int, default=10 ** 8, help="Step number for learning")
    parser.add_argument("--reset-num", type=int, default=200, help="Reset number for each episode")
    parser.add_argument("--start-step", type=int, default=2000, help="Start step of training")
    parser.add_argument("--eval-interval", type=int, default=70, help="Evaluation interval")
    parser.add_argument("--batch-size", type=int, default=32, help="training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=.95, help="decay factor")
    parser.add_argument("--max-eps", type=float, default=1, help="epsilon greedy upper bound")
    parser.add_argument("--min-eps", type=float, default=.1, help="epsilon greedy lower bound")
    parser.add_argument("--eps-decay", type=float, default=1e-5, help="epsilon greedy decay rate")
    parser.add_argument("--replay-size", type=int, default=10 ** 7, help="replay buffer size")
    parser.add_argument("--update-freq", type=int, default=10000, help="target network update frequency")
    parser.add_argument("--env-difficulty", type=int, default=8, help="difficulty of the environment")
    parser.add_argument("--net-init", type=str, default="", help="the initial model for the net")
    args = parser.parse_args()

    # initialize the environment
    num_tube_classes = 5
    rack_size = (5, 10)
    action_space_dim = np.prod(rack_size) ** 2
    observation_space_dim = (1, 5 * 4, 10)
    observation_space_dim_nn = (1, *rack_size)
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_tube_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=True,
                             is_goalpattern_fixed=True,
                             difficulty=args.env_difficulty, )
    env.goal_pattern = np.array([[1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
                                 [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
                                 [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
                                 [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
                                 [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]])

    # initialize the main pipeline
    # setup the file to dump the data
    data_path = fs.workdir_learning / "run" / f"dqn_{datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')}"
    datasets = []
    if not data_path.is_dir():
        data_path.mkdir()

    # initilize the neural network
    net = DuelingDQN(obs_dim=observation_space_dim_nn, action_dim=action_space_dim)
    if Path(args.net_init).is_file():
        print(f"Load model file from {args.net_init}")
        net.load_state_dict(torch.load(args.net_init))

    dqn_agent = DQNAgent(env=env,
                         net=net,
                         replay_size=args.replay_size,
                         batch_size=args.batch_size,
                         target_update_freq=args.update_freq,
                         lr=args.lr,
                         epsilon_decay=args.eps_decay,
                         max_epsilon=args.max_eps,
                         min_epsilon=args.min_eps,
                         gamma=args.gamma,
                         alpha=.6,
                         path=data_path, )

    dqn_agent.train(num_steps=args.step_num,
                    reset_num=args.reset_num,
                    replay_init_steps=args.start_step,
                    eval_interval=args.eval_interval)
