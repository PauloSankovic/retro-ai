import logging
import time

import gym
from gym.core import Env


def instantiate(environment_id: str, **kwargs) -> Env:
    # init_logging(environment_id)
    return gym.make(environment_id, **kwargs)


def run(env: Env, agent, timeout: float = 0) -> float:
    state = env.reset()
    # logger.info('Initial state: %s', state)

    total_reward = 0.0
    done = False
    while not done:
        time.sleep(timeout)
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)

        # if logger.isEnabledFor(logging.DEBUG):
        #     logger.debug('Agent action: %s; Current state: %s; Reward: %s; Done: %s', action, state, reward, done)

        total_reward += reward
        env.render()

    # logger.info('Total reward: %f', total_reward)
    return total_reward


def init_logging(environment_id: str):
    logging.basicConfig(format=f'%(asctime)s - %{environment_id} - %(message)s', datefmt='%H:%M:%S')
