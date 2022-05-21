import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.core import Env
from torch.distributions import Categorical


class ActorCriticAgent(nn.Module):
    def __init__(self, env: Env, **kwargs):  # TODO dodati learning rate i discount factor
        super().__init__()
        self.name = 'ActorCriticAgent'
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n
        sizes = kwargs.pop('hidden_layers', [128])
        sizes.insert(0, input_size)

        self.gamma = kwargs.pop('gamma', 0.99)

        self.fc1 = nn.Linear(in_features=input_size, out_features=128, bias=True)
        self.actor = nn.Linear(in_features=128, out_features=output_size, bias=True)
        self.critic = nn.Linear(in_features=128, out_features=1, bias=True)

        self.actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))

        action_probability = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probability, state_value

    def get_action(self, state, train):
        state = torch.from_numpy(state).float()

        with torch.set_grad_enabled(train):
            action_probability, state_value = self.forward(state)

        m = Categorical(action_probability)
        action = m.sample()

        if train:
            self.actions.append((m.log_prob(action), state_value))
        return action.item()

    def evaluate(self, optimizer):
        cum_reward = 0
        expected_returns = []
        for reward in self.rewards[::-1]:
            cum_reward = reward + self.gamma * cum_reward
            expected_returns.insert(0, cum_reward)

        expected_returns = torch.tensor(expected_returns)
        expected_returns = (expected_returns - expected_returns.mean()) / (expected_returns.std() + np.finfo(np.float32).eps.item())

        actor_losses = []
        critic_losses = []
        for (action_prob, state_value), reward in zip(self.actions, expected_returns):
            advantage = reward - state_value.item()

            actor_losses.append(-action_prob * advantage)
            critic_losses.append(F.smooth_l1_loss(state_value, torch.tensor([reward])))

        optimizer.zero_grad()
        loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()

        loss.backward()
        optimizer.step()

        del self.rewards[:]
        del self.actions[:]
