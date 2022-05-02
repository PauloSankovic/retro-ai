import numpy as np
from gym.core import Env

from randomAgent import RandomAgent


class BasicQAgent(RandomAgent):
    def __init__(self, env: Env, **kvargs):
        super().__init__(env, "BasicQAgent")
        self.state_size = env.observation_space.n
        self.q_table = 1e-4 * np.random.random((self.state_size, self.action_size))

        self.eps = kvargs.get('eps', 1)
        self.discount_rate = kvargs.get('discount_rate', 1)
        self.learning_rate = kvargs.get('learning_rate', 1)

    def get_action(self, state, train: bool = False):
        q_state = self.q_table[state]
        action_greedy = np.argmax(q_state)
        if not train:
            return action_greedy
        action_random = super().get_action(state)
        return action_random if np.random.random() < self.eps else action_greedy

    def train(self, state, action, next_state, reward, done):
        q_next = np.zeros(self.action_size) if done else self.q_table[next_state]
        q_target = reward + self.discount_rate * np.max(q_next)

        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update

        if done:
            self.eps *= 0.99
