import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import itertools

from wrappers import SubprocVecEnv, Monitor
from playground.agents.ddqnAgent import DoubleDeepQNetworkAgent, ReplayMemory
from networks import cnn, CnnStructure
from wrappers.pytorch_wrappers import make_atari_deepmind, PytorchLazyFrames, BatchedPytorchFrameStack

from utils import save_state_dict

# learning rate
ALPHA = 5e-5
# discount rate for computing our temporal difference target
GAMMA = 0.99
BATCH_SIZE = 32
# maximum number of transitions we store before overwriting old transitions
BUFFER_SIZE = int(1e6)
# how many transitions we want in replay buffer
# before we start calculating gradients and training
MIN_REPLAY_SIZE = 50_000
EPSILON_START = 1.0
EPSILON_END = 0.1
# decay period
EPSILON_DECAY = int(1e6)
# number of environments
NUM_ENVS = 4
# number of steps where we set the target parameters equal to the online parameters
TARGET_UPDATE_FREQ = 10_000 // NUM_ENVS
# model parameters saving interval
SAVE_INTERVAL = 10_000
# summary writer directory
LOG_DIR = '../summary/breakout/DDQN'
# logging interval
LOG_INTERVAL = 1_000

if __name__ == '__main__':
    replay_memory = ReplayMemory(BUFFER_SIZE)
    summary_writer = SummaryWriter(LOG_DIR)

    env = lambda: Monitor(make_atari_deepmind('ALE/Breakout-v5', scale_values=True), allow_early_resets=True)

    # switched to the batched env ->
    # everything returned from batched environment has batched dimension
    # these envs both reset the env when it's done for us
    # env = DummyVecEnv([env for _ in range(NUM_ENVS)])
    env = SubprocVecEnv([env for _ in range(NUM_ENVS)])

    env = BatchedPytorchFrameStack(env, k=NUM_ENVS)

    cnn_layers = [
        CnnStructure(in_channels=env.observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
        CnnStructure(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        CnnStructure(in_channels=64, out_channels=64, kernel_size=3, stride=1)
    ]
    net = cnn(env.observation_space, cnn_layers, [512], env.action_space.n)

    online_net = DoubleDeepQNetworkAgent(env, net)
    target_net = DoubleDeepQNetworkAgent(env, net)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.Adam(online_net.parameters(), lr=ALPHA)

    states = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        actions = [env.action_space.sample() for _ in range(NUM_ENVS)]
        next_states, rewards, dones, _ = env.step(actions)
        for state, action, done, next_state, reward in zip(states, actions, dones, next_states, rewards):
            replay_memory.push(state, action, done, next_state, reward)
        states = next_states

    # Main training loop
    states = env.reset()

    ep_infos = []
    episode_count = 0
    for step in itertools.count():
        epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        if isinstance(states[0], PytorchLazyFrames):
            act_states = np.stack([s.get_frames() for s in states])
            actions = online_net.get_actions(act_states, epsilon)
        else:
            actions = online_net.get_actions(states, epsilon)

        next_states, rewards, dones, infos = env.step(actions)
        for state, action, done, next_state, reward, info in zip(states, actions, dones, next_states, rewards, infos):
            replay_memory.push(state, action, done, next_state, reward)

            if done:
                ep_infos.append(info['episode'])
                episode_count += 1

        states = next_states

        # Start Gradient Step
        transitions = replay_memory.sample(BATCH_SIZE)
        loss = online_net.loss(transitions, target_net, GAMMA)

        # Gradient Decent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % LOG_INTERVAL == 0:
            rew_mean = np.mean([e['r'] for e in ep_infos]) or 0
            len_mean = np.mean([e['l'] for e in ep_infos]) or 0

            print()
            print('Step', step)
            print('Average reward', rew_mean)
            print('Average episode length', len_mean)
            print('Episodes', episode_count)
            summary_writer.add_scalar('AvgRew', rew_mean, global_step=step)
            summary_writer.add_scalar('AvgEpLen', len_mean, global_step=step)
            summary_writer.add_scalar('Episodes', episode_count, global_step=step)

        if step % SAVE_INTERVAL == 0 and step != 0:
            print()
            print("Saving...")
            save_state_dict(online_net, env='breakout', step=str(step), v='2')
