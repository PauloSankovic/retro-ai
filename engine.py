import logging
import time

import numpy as np
import gym
from IPython.display import clear_output
from gym.core import Env
from torch.utils.tensorboard import SummaryWriter

from utils import parts_to_string

logger = logging.getLogger(__name__)

LOG_INTERVAL = 1_000
LOG_DIR = '../summary/cartpole/' + parts_to_string(net='a2c', lr='5e-4')

def instantiate(environment_id: str, **kwargs) -> Env:
    logger.info(f"New environment created - {environment_id} {kwargs if len(kwargs) > 0 else ''}")
    return gym.make(environment_id, **kwargs)


def run(env: Env, agent, optimizer=None, stop_callback=None, verbose: bool = False, **kwargs):
    _clear_output = kwargs.get('clear_output', False)

    summary_writer = SummaryWriter(LOG_DIR)

    running_reward = env.spec.reward_threshold * 0.01
    episode = 0
    losses = []
    iters = 0
    while True:
        if _clear_output:
            clear_output(wait=True)

        reward, iters = run_episode(env, agent, iters, kwargs.get('train', False), kwargs.get('render', True),
                             kwargs.get('remember_rewards', False), kwargs.get('timeout', 0))
        running_reward = 0.05 * reward + 0.95 * running_reward

        if optimizer:
            loss = agent.evaluate(optimizer)
            losses.append(loss)

        summary_writer.add_scalar("Cumulative reward", running_reward, global_step=iters)
        summary_writer.add_scalar("Loss", np.mean(losses), global_step=iters)
        summary_writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], global_step=iters)

        if verbose and episode % 10 == 0:
            print(f"Episode {episode} -> Reward {reward} (running reward: {running_reward:.3f})")

        episode += 1
        if running_reward > env.spec.reward_threshold:
            print(f"Solved after {episode} episodes -> {running_reward}")
            break
    logger.info("Terminating")


def run_episode(env: Env, agent, iters, train: bool = False, render: bool = True, remember_rewards: bool = False,
                timeout: float = 0) -> float:
    state = env.reset()

    total_reward = 0.0
    done = False
    iters = iters
    while not done:
        if train:
            action = agent.get_action(state, True)
            next_state, reward, done, info = env.step(action)
            # agent.train(state, action, next_state, reward, info, done)
        else:
            action = agent.get_action(state, False)
            next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state
        if render:
            env.render()
        if remember_rewards:
            agent.rewards.append(reward)

        iters += 1

        time.sleep(timeout)

    return total_reward, iters
