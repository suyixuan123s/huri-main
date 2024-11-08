import gym
import gym.wrappers
import numpy as np
import torch
from torch import nn
import pfrl
from pfrl.agents.ddpg import DDPG
from pfrl import experiments, explorers, replay_buffers, utils
from huri.learning.env.yumi_ik_solver.env_cpu import YuMiEnv
from pfrl.nn import BoundByTanh, ConcatObsAndAction
from pfrl.policies import DeterministicHead


# Set a random seed used in PFRL
def main(seed=888,
         batch_size=32,
         replay_start_size=10000,
         steps=10e10,
         eval_n_episodes=10,
         train_max_episode_len=30,
         eval_interval=1000,
         outdir=".",
         ):
    utils.set_random_seed(seed=seed)

    def make_env(test=False):
        env = YuMiEnv()
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - seed if test else seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        return env

    env = make_env(test=False)
    obs_space = env.observation_space
    action_space = env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    q_func = nn.Sequential(
        ConcatObsAndAction(),
        nn.Linear(obs_size + action_size, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, 1),
    )
    policy = nn.Sequential(
        nn.Linear(obs_size, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, action_size),
        BoundByTanh(low=action_space.low, high=action_space.high),
        DeterministicHead(),
    )

    actor_optimizer_static_dict = torch.load("result/best/actor_optimizer.pt")
    critic_optimizer_static_dict = torch.load("result/best/actor_optimizer.pt")

    opt_a = torch.optim.Adam(policy.parameters())
    opt_c = torch.optim.Adam(q_func.parameters())

    rbuf = replay_buffers.ReplayBuffer(10 ** 6)

    explorer = explorers.AdditiveGaussian(
        scale=0.1, low=action_space.low, high=action_space.high
    )

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = DDPG(
        policy,
        q_func,
        opt_a,
        opt_c,
        rbuf,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=replay_start_size,
        target_update_method="soft",
        target_update_interval=1,
        update_interval=1,
        soft_update_tau=5e-3,
        n_times_update=1,
        gpu=0,
        minibatch_size=batch_size,
        burnin_action_func=burnin_action_func,
    )

    agent.load("result/best")

    # import logging
    # import sys
    # logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
    import time
    eval_env = make_env(test=True)
    eval_stats = experiments.train_agent_with_evaluation(
        env=eval_env,
        agent=agent,
        steps=steps,  # Train the agent for 2000 steps
        eval_n_steps=None,  # We evaluate for episodes, not time
        eval_n_episodes=eval_n_episodes,  # 10 episodes are sampled for each evaluation
        train_max_episode_len=train_max_episode_len,  # Maximum length of each episode
        eval_interval=eval_interval,  # Evaluate the agent after every 1000 steps
        outdir=f"result{time.strftime('%Y%m%d-%H%M%S')}",
        use_tensorboard=True,
    )


if __name__ == "__main__":
    main()
