from stable_baselines3 import A2C

env = A2C('MlpPolicy', 'CartPole-v1', verbose=1)
model = env.learn(total_timesteps=100_000)
