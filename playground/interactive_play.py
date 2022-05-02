import gym
from gym.utils import play

import envs

env = gym.make(envs.CAR_RACING)
play.play(env, zoom=3)
