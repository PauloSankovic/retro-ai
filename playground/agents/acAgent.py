import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.core import Env


class ActorCriticAgent(nn.Module):
    def __init__(self, env: Env, **kwargs):  # TODO dodati learning rate i discount factor
        super().__init__()
        self.name = 'ActorCriticAgent'
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n
        sizes = kwargs.pop('hidden_layers', [128])
        sizes.insert(0, input_size)

        self.gamma = kwargs.pop('gamma', 0.99)

        self.layers = []
        for i in range(1, len(sizes)):
            self.layers.append(nn.Linear(in_features=sizes[i - 1], out_features=sizes[i], bias=True))
        self.actor = nn.Linear(in_features=sizes[-1], out_features=output_size, bias=True)
        self.critic = nn.Linear(in_features=sizes[-1], out_features=1, bias=True)

        self.actions = []
        self.rewards = []

    def forward(self, x, ):
        for layer in self.layers:
            x = F.relu(layer(x))

        action_probability = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probability, state_value

    def get_action(self, state, train):
        state = torch.from_numpy(state).float()

        with torch.set_grad_enabled(train):
            action_probability, state_value = self.forward(state)

        action = torch.argmax(action_probability)
        if train:
            self.actions.append((action_probability[action], state_value))
        return action.item()

    def evaluate(self):
        cum_reward = 0
        expected_returns = []
        for reward in self.rewards[::-1]:
            cum_reward = reward + self.gamma * cum_reward
            expected_returns.insert(0, cum_reward)

        expected_returns = torch.tensor(expected_returns)
        expected_returns = (expected_returns - expected_returns.mean()) / (expected_returns.std() - np.finfo(np.float32).eps.item())

        actor_losses = []
        critic_losses = []
        for (action_prob, state_value), reward in zip(self.actions, expected_returns):
            advantage = reward - state_value.item()

            actor_losses.append(-torch.log(action_prob) * advantage)
            critic_losses.append(F.smooth_l1_loss(state_value, torch.tensor(reward)))

        optimizer.zero_grad()
        loss = sum(actor_losses) + sum(critic_losses)

        loss.backward()
        optimizer.step()

        del self.rewards[:]
        del self.actions[:]


import engine
import envs
import logging
import torch.optim as optim
import numpy as np

logger = logging.getLogger(__name__)

env = engine.instantiate(envs.CART_POLE)

agent = ActorCriticAgent(env)

optimizer = optim.Adam(agent.parameters(), lr=3e-2)

env.seed(42)
torch.manual_seed(42)


def run_epoch(render):
    state = env.reset()

    epoch_reward = 0.0
    done = False
    while not done:
        action = agent.get_action(state, True)
        next_state, reward, done, info = env.step(action)
        agent.rewards.append(reward)
        epoch_reward += reward
        state = next_state
        if render:
            env.render()
    return epoch_reward


cumulative_reward = 0
epoch = 0
while True:
    epoch_reward = run_epoch(render=False)
    cumulative_reward = 0.05 * epoch_reward + (1 - 0.05) * cumulative_reward
    print(f"Episode {epoch} -> Reward {epoch_reward}")
    epoch += 1
    agent.evaluate()
    if cumulative_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and the last episode runs to {} reward!".format(cumulative_reward, epoch_reward))
        break

logger.info("Terminating")
