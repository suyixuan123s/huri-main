import time

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
from huri.core.common_import import wd, gm


# Set a random seed used in PFRL
def main(seed=888,
         batch_size=100,
         replay_start_size=10000,
         weight_dir=".",
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

    env = make_env(test=True)
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

    agent.load(weight_dir)
    print(agent)

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    obs = env.reset()
    e = obs.copy()
    a = time.time()
    for i in range(10):
        # rbt_mdl, tgt_pt_mdl = env.render_current_state()
        # rbt_mdl.attach_to(base)
        # tgt_pt_mdl.attach_to(base)

        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)

        # print(obs, reward)
    b = time.time()
    print(b - a)
    rbt_mdl, tgt_pt_mdl = env.render_current_state()
    rbt_mdl.attach_to(base)
    tgt_pt_mdl.attach_to(base)

    r = env.obs_2_tgt_homomat(e)
    a = time.time()
    print(env.env._yumi.ik("rgt_arm", r[:3, 3], r[:3, :3], e[:7]))
    b = time.time()
    print(b - a)

    base.run()


if __name__ == "__main__":
    main(weight_dir="./result20220701-142859/best")
