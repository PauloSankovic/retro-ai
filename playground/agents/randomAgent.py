import numpy as np

from gym.core import Env
from gym.spaces.discrete import Discrete


class RandomAgent:
    def __init__(self, env: Env, name: str = "RandomAgent"):
        self.name = name
        self.is_discrete = type(env.action_space) == Discrete

        if self.is_discrete:
            self.action_size = env.action_space.n
        if not self.is_discrete:
            self.action_high = env.action_space.high
            self.action_low = env.action_space.low
            self.action_shape = env.action_space.shape

    def get_action(self, state, train: bool = False):
        if self.is_discrete:
            return np.random.choice(self.action_size)
        else:
            return np.random.uniform(self.action_low, self.action_high, self.action_shape)

    def train(self, state, action, next_state, reward, done):
        raise NotImplementedError()
