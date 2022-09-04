import gym
from gym.wrappers import RecordVideo

from agents import DeepQNetworkAgent
from networks import fc
from playground.agents import DoubleDeepQNetworkAgent, AdvantageActorCriticAgent
from utils import load_state_dict

env = gym.make('CartPole-v1')

# agent = DeepQNetworkAgent(env, fc([env.observation_space.shape[0], 128, 64, env.action_space.n]))
# agent.load_state_dict(load_state_dict('DeepQNetworkAgent', env='dqn', v='2', step='122368', lr='5e-4', bs='32', es='1', ee='0.02', ed='10000'))
# env = RecordVideo(env, './video/cart-pole-dqn')

# agent = DoubleDeepQNetworkAgent(env, fc([env.observation_space.shape[0], 128, 64, env.action_space.n]))
# agent.load_state_dict(load_state_dict('DoubleDeepQNetworkAgent', env='ddqn', v='2', step='140144', lr='5e-4', bs='32', es='1', ee='0.02', ed='10000'))
# env = RecordVideo(env, './video/cart-pole-ddqn')

agent = AdvantageActorCriticAgent(env)
agent.load_state_dict(load_state_dict('ActorCriticAgent', env='a2c', lr='3e-2'))
env = RecordVideo(env, './video/cart-pole-a2c')


env.env.spec.max_episode_steps = 2000

done = False
state = env.reset()
while not done:
    action = agent.get_action(state, False)
    next_state, reward, done, info = env.step(action)
    state = next_state
    env.render()