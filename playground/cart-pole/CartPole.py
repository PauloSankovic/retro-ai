import time

import gym
from RandomAgent import *
from DummyAgent import *

# Create an instance of environment
env = gym.make("CartPole-v1")

def play(agent):
    # Initialize the environment state (uniform random value between ±0.05)
    state = env.reset()

    for _ in range(500):
        # time.sleep(0.5)
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        print(action, "->", state, reward, done, info)
        env.render()
        if done:
            env.reset()
            return


agent = DummyAgent(env)
play(agent)
