import torch
import torch.nn as nn

from collections import deque, namedtuple
import random

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
    def __init__(self, net):
        super().__init__()
        self.name = 'DeepQNetworkAgent'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = net.to(self.device)
        self.to(self.device)

    def forward(self, x):
        return self.net(x)

    def get_action(self, observation, train: bool = False):
        observation = torch.tensor(observation, device=self.device).unsqueeze(0)
        q_values = self(observation)

        max_action = torch.argmax(q_values, dim=1)
        return max_action.item()
