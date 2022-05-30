import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import deque, namedtuple
import random

from pytorch_wrappers import PytorchLazyFrames

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


class DeepQNetworkAgent(nn.Module):
    def __init__(self, env, net):
        super().__init__()
        self.name = 'DeepQNetworkAgent'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_actions = env.action_space.n
        self.net = net.to(self.device)
        self.to(self.device)

    def forward(self, x):
        return self.net(x.float())

    def get_action(self, observation, train: bool = False):
        observation = torch.tensor(observation, device=self.device).unsqueeze(0)
        q_values = self(observation)

        max_action = torch.argmax(q_values, dim=1)
        return max_action.item()

    def get_actions(self, observations, epsilon, train: bool = False):
        observations = torch.tensor(observations, device=self.device)
        q_values = self(observations)

        actions = torch.argmax(q_values, dim=1).tolist()
        if not train:
            return actions

        for i in range(len(actions)):
            if random.random() <= epsilon:
                actions[i] = random.randint(0, self.n_actions - 1)

        return actions

    def loss(self, transitions, target_net, gamma):
        states = [t.state for t in transitions]
        actions = np.asarray([t.action for t in transitions])
        dones = np.asarray([t.done for t in transitions])
        next_states = [t.next_state for t in transitions]
        rewards = np.asarray([t.reward for t in transitions])

        if isinstance(states[0], PytorchLazyFrames):
            states = np.stack([s.get_frames() for s in states])
            next_states = np.stack([s.get_frames() for s in next_states])
        else:
            states = np.asarray(states)
            next_states = np.asarray(next_states)


        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(-1)
        dones = torch.tensor(dones, device=self.device, dtype=torch.int64).unsqueeze(-1)
        next_states = torch.tensor(next_states, device=self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(-1)

        # Compute targets
        target_q_values = target_net(next_states)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards + gamma * (1 - dones) * max_target_q_values

        # Compute Loss
        # we got a set of q values for each state
        q_values = self(states)

        # we need to get a q value for the actual action we took in that transition
        # gather applies actions index and dim to all q_values, so we got value -> action mapping
        action_q_values = torch.gather(input=q_values, dim=1, index=actions)

        loss = F.huber_loss(action_q_values, targets)
        return loss