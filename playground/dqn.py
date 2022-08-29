import torch
import gym
import itertools
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter

from playground.agents import ReplayMemory, DeepQNetworkAgent, DoubleDeepQNetworkAgent
from networks import fc

from utils import save_state_dict, parts_to_string

# learning rate
ALPHA = 5e-4
# discount rate for computing our temporal difference target
GAMMA = 0.99
BATCH_SIZE = 32
# maximum number of transitions we store before overwriting old transitions
BUFFER_SIZE = 50_000
# how many transitions we want in replay buffer
# before we start calculating gradients and training
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
# decay period
EPSILON_DECAY = 10_000
# number of steps where we set the target parameters equal to the online parameters
TARGET_UPDATE_FREQ = 1000
# summary writer directory
LOG_DIR = '../summary/cartpole/' + parts_to_string(net='ddqn', lr='5e-4', bs='32', es='1', ee='0.02', ed='10000')
# logging interval
LOG_INTERVAL = 1_000
# model parameters saving interval
SAVE_INTERVAL = 20_000


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    replay_memory = ReplayMemory(BUFFER_SIZE)
    summary_writer = SummaryWriter(LOG_DIR)

    cumulative_reward = 0
    episode_reward = 0

    net = fc([env.observation_space.shape[0], 128, 64, env.action_space.n])

    online_net = DeepQNetworkAgent(env, net)
    target_net = DeepQNetworkAgent(env, net)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

    # Initialize replay buffer
    state = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        replay_memory.push(state, action, done, next_state, reward)
        state = next_state

        if done:
            state = env.reset()

    # Main Training Loop
    state = env.reset()
    losses = []
    for step in itertools.count():
        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = online_net.get_action(state, train=True)

        next_state, reward, done, _ = env.step(action)
        replay_memory.push(state, action, done, next_state, reward)
        state = next_state

        episode_reward += reward

        if done:
            state = env.reset()

            cumulative_reward = 0.05 * episode_reward + 0.95 * cumulative_reward
            episode_reward = 0

        if cumulative_reward > 1000:
            print(f"Solved after {step} steps -> {cumulative_reward}")
            save_state_dict(online_net, env='breakout-ram', step=str(step), lr='5e-4', bs='32', es='1', ee='0.02',
                            ed='10000')
            break

        transitions = replay_memory.sample(BATCH_SIZE)
        loss = online_net.loss(transitions, target_net, GAMMA)
        losses.append(loss.item())

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Target Net
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % LOG_INTERVAL == 0:
            print()
            print('Step:', step)
            print('Cumulative reward:', cumulative_reward)

            summary_writer.add_scalar("Cumulative reward", cumulative_reward, global_step=step)
            summary_writer.add_scalar("Exploration", epsilon, global_step=step)
            summary_writer.add_scalar("Loss", np.mean(losses), global_step=step)
            summary_writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], global_step=step)

            losses = []

        # if step % SAVE_INTERVAL == 0 and step != 0:
        #     print("Saving")
        #     save_state_dict(online_net, env='cartpole', step=str(step), lr='5e-4', bs='32', es='1', ee='0.02',
        #                     ed='10000')
