import engine
import envs
from agents import RandomAgent

env = engine.instantiate(envs.CAR_RACING)

agent = RandomAgent(env)

for _ in range(10):
    engine.run(env, agent)
