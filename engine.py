import logging
import time

import gym
from gym.core import Env

from IPython.display import clear_output

logger = logging.getLogger(__name__)


def instantiate(environment_id: str, **kwargs) -> Env:
    logger.info(f"New environment created - {environment_id} {kwargs if len(kwargs) > 0 else ''}")
    return gym.make(environment_id, **kwargs)


def run(env: Env, agent, stop_callback, verbose: bool = False, **kwargs):
    _clear_output = kwargs.get('clear_output', False)

    cumulative_reward = 0
    episode = 0
    while True:
        if _clear_output:
            clear_output(wait=True)

        reward = run_episode(env, agent, kwargs.get('train', False), kwargs.get('render', True),
                             kwargs.get('timeout', 0))
        cumulative_reward += reward

        if verbose and (episode % 10 == 0 or episode == 999):
            agent.snapshot(episode, reward, cumulative_reward)
            logger.info(f"Episode {episode} -> Reward {reward}")

        episode += 1
        if episode >= 1000:
            break
    logger.info("Terminating")


def run_episode(env: Env, agent, train: bool = False, render: bool = True, timeout: float = 0) -> float:
    state = env.reset()

    total_reward = 0.0
    done = False
    while not done:
        if train:
            action = agent.get_action(state, train)
            next_state, reward, done, info = env.step(action)
            agent.train(state, action, next_state, reward, info, done)
        else:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state
        if render:
            env.render()
        time.sleep(timeout)

    return total_reward
