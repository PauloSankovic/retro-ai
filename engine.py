import logging
import time

import gym
from gym.core import Env

logger = logging.getLogger(__name__)


def instantiate(environment_id: str, **kwargs) -> Env:
    logger.info(f"New environment created - {environment_id} {kwargs if len(kwargs) > 0 else ''}")
    return gym.make(environment_id, **kwargs)


def run(env: Env, agent, stop_callback, **kwargs):
    cumulative_reward = 0
    episode = 1
    while True:
        reward = run_episode(env, agent, render=True)
        cumulative_reward += reward
        logger.info(f"Episode {episode} -> Reward {reward}")

        episode += 1
        if stop_callback:
            break


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
