import logging
import time

import gym
from gym.core import Env


def instantiate(environment_id: str, **kwargs) -> Env:
    return gym.make(environment_id, **kwargs)


def run(env: Env, agent, timeout: float = 0) -> float:
    state = env.reset()

    total_reward = 0.0
    done = False
    while not done:
        time.sleep(timeout)
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)

        total_reward += reward
        env.render()

    return total_reward


def init_logging(environment_id: str):
    logging.basicConfig(format=f'%(asctime)s - %{environment_id} - %(message)s', datefmt='%H:%M:%S')
