import itertools

import numpy as np
import torch
from gym.wrappers import RecordVideo

from networks import cnn, CnnStructure
from playground.agents import DoubleDeepQNetworkAgent
from utils import load_state_dict
from wrappers import DummyVecEnv, Monitor
from wrappers.pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

make_env = lambda: Monitor(make_atari_deepmind('ALE/Breakout-v5', render_mode='human', scale_values=True),
                           allow_early_resets=True)

vec_env = DummyVecEnv([make_env for _ in range(1)])

env = RecordVideo(BatchedPytorchFrameStack(vec_env, k=4), './video/breakout-ddqn',
                  episode_trigger=lambda episode_id: True)
# env = BatchedPytorchFrameStack(vec_env, k=4)

cnn_layers = [
    CnnStructure(in_channels=env.observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
    CnnStructure(in_channels=32, out_channels=64, kernel_size=4, stride=2),
    CnnStructure(in_channels=64, out_channels=64, kernel_size=3, stride=1)
]
net = cnn(env.observation_space, cnn_layers, [512], env.action_space.n)

agent = DoubleDeepQNetworkAgent(env, net)

# state = load_state_dict('DeepQNetworkAgent', env='breakout', step='1080000')
state = load_state_dict('DoubleDeepQNetworkAgent', env='breakout', step='1400000', v='2')
agent.load_state_dict(state)

obs = env.reset()
beginning_episode = True
iter = 0
for t in itertools.count():
    if isinstance(obs[0], PytorchLazyFrames):
        act_obs = np.stack([o.get_frames() for o in obs])
        action = agent.get_actions(act_obs, 0.0)
    else:
        action = agent.get_actions(obs, 0.0)

    if beginning_episode:
        action = [1]
        beginning_episode = False

    obs, rew, done, info = env.step(action)

    if done[0] and info[0]['lives'] == 0:
        print(iter, "->", info[0]['episode']['r'])
        iter = 0

        obs = env.reset()
        beginning_episode = True
        break
    iter += 1
