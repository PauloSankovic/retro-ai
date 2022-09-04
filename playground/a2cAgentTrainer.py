import gym
import torch.optim as optim

import engine
from agents import AdvantageActorCriticAgent
from utils import save_state_dict

env = gym.make('CartPole-v1')

agent = AdvantageActorCriticAgent(env)
optimizer = optim.Adam(agent.parameters(), lr=3e-2)

engine.run(env, agent, optimizer=optimizer, train=True, verbose=True, remember_rewards=True, clear_output=True, render=False)

save_state_dict(agent, env='a2c', lr='3e-2')