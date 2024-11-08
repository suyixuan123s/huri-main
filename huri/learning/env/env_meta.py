import gym
from gym.utils import seeding
from gym import spaces


class Gym_Proto(gym.Env):

    def __init__(self, *args, **kwargs):
        super(Gym_Proto, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects

    def step(self, action):
        # Execute one time step within the environment
        raise NotImplementedError

    def reset(self):
        # Reset the state of the environment to an initial state
        raise NotImplementedError

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        raise NotImplementedError

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]
