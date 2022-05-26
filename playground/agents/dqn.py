import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque, namedtuple
import random

import gym

# learning rate
ALPHA = 5e-4
# discount rate for computing our temporal difference target
GAMMA = 0.99
BATCH_SIZE = 32
# maximum number of transitions we store before overwriting old transitions
BUFFER_SIZE = 50_000
# how many transitions we want in replay buffer
# before we start calculating gradients and training
MIN_REPLAY_SIZE = 1_000
EPSILON_START = 1.0
EPSILON_END = 0.02
# decay period
EPSILON_DECAY = 10_000
# number of steps where we set the target parameters equal to the online parameters
TARGET_UPDATE_FREQ = 1_000

Transition = namedtuple('Transition', ('state', 'action', 'done', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# only with discrete action space
class DeepQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = env.observation_space.shape[0]
        out_features = env.action_space.n

        self.fc1 = nn.Linear(in_features=in_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=out_features)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_action(self, observation):
        observation = torch.tensor(observation, device=self.device).unsqueeze(0)
        q_values = self.forward(observation)

        max_action = torch.argmax(q_values, dim=1)
        return max_action.item()


replay_memory = ReplayMemory(BUFFER_SIZE)

env = gym.make('CartPole-v1')
online_net = DeepQNetwork(env)
target_net = DeepQNetwork(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = optim.Adam(online_net.parameters(), lr=ALPHA)

state = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    replay_memory.push(state, action, done, next_state, reward)
    state = next_state

    if done:
        state = env.reset()

# Main training loop
state = env.reset()

running_reward = env.spec.reward_threshold * 0.01
episode_reward = 0
for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    if random.random() <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.get_action(state)

    next_state, reward, done, _ = env.step(action)
    replay_memory.push(state, action, done, next_state, reward)
    state = next_state

    episode_reward += 1

    if done:
        state = env.reset()

        running_reward = 0.05 * episode_reward + 0.95 * running_reward
        if running_reward > env.spec.reward_threshold:
            print("Running reward", running_reward, "with latest episode reward", episode_reward)
        episode_reward = 0

    # Start Gradient Step
    transitions = replay_memory.sample(BATCH_SIZE)

    states = np.asarray([t.state for t in transitions])
    actions = np.asarray([t.action for t in transitions])
    dones = np.asarray([t.done for t in transitions])
    next_states = np.asarray([t.next_state for t in transitions])
    rewards = np.asarray([t.reward for t in transitions])

    states = torch.tensor(states, device=online_net.device)
    actions = torch.tensor(actions, device=online_net.device, dtype=torch.int64).unsqueeze(-1)
    dones = torch.tensor(dones, device=online_net.device, dtype=torch.int64).unsqueeze(-1)
    next_states = torch.tensor(next_states, device=online_net.device)
    rewards = torch.tensor(rewards, device=online_net.device, dtype=torch.float32).unsqueeze(-1)

    # Compute targets
    target_q_values = target_net.forward(next_states)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rewards + GAMMA * (1 - dones) * max_target_q_values

    # Compute Loss
    # we got a set of q values for each state
    q_values = online_net.forward(states)

    # we need to get a q value for the actual action we took in that transition
    # gather applies actions index and dim to all q_values, so we got value -> action mapping
    action_q_values = torch.gather(input=q_values, dim=1, index=actions)

    loss = F.huber_loss(action_q_values, targets)

    # Gradient Decent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 1000 == 0:
        print('Step', step, 'Running reward', running_reward)
