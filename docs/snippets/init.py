import gym

env = gym.make("MountainCar-v0")

observation = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    env.render()

env.close()
