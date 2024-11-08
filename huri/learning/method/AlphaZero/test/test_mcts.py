import numpy as np
import unittest
from huri.learning.method.AlphaZero import mcts
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackState


def mock_eval_func(state_tensor, batched=False):
    # Mock network output
    num_actions = state_tensor.shape[-1] ** 2
    if not batched:
        prior_shape = (num_actions,)
        value_shape = (1,)
    else:
        batch_size = state_tensor.shape[0]
        prior_shape = (
            batch_size,
            num_actions,
        )
        value_shape = (
            batch_size,
            1,
        )

    prior_prob = np.random.uniform(size=prior_shape)
    v = np.random.uniform(-1, 1, size=value_shape)

    if not batched:
        v = v.item()

    return (prior_prob, v)


class UCTSearchTest(unittest.TestCase):
    def test_run_uct_search(self):
        rack_sz = (5, 10)
        obs_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_sz)
        env = RackArrangementEnv(rack_size=(5, 10),
                                 num_classes=3,
                                 observation_space_dim=obs_dim,
                                 action_space_dim=act_dim,
                                 is_curriculum_lr=False,
                                 is_goalpattern_fixed=False)
        obs = env.reset()
        root_node = None
        steps = 0
        while steps < 10:
            action, pi_prob, root_node = mcts.uct_search(env, mock_eval_func, root_node, 19652, 1.25, 1.0, 100)
            obs, reward, done, info = env.step(action)
            if done:
                break
            steps += 1
