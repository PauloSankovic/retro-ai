# Technology Research

## PyGame
*Free and Open Source python programming language library for making multimedia applications like games built on top of the SDL (Simple DirectMedia Layer) library.*

[News](https://www.pygame.org/news)  
[GitHub Repository](https://github.com/pygame/)

**Pros:**
- beginner friendly - simple to use
- comprehensive docs
- lots of tutorials and already finished games

**Cons:**
- not suitable for complex games
- slower than modern game engines

**Setup:** 
`pip install pygame`

**Tutorials and other useful links:**
- [Creating Space Invaders in Pygame/Python](https://youtu.be/o-6pADy5Mdg)
- [Gentle Intro To Neural Nets - Flappy and PyGame](https://youtu.be/ra2o2bPZlwk)
- [PyGame and OpenAI implementation](https://stackoverflow.com/questions/58974034/pygame-and-open-ai-implementation)
- [Teach AI To Play Snake - Reinforcement Learning Tutorial With PyTorch And Pygame](https://youtu.be/VGkcmBaeAGM)

![All PyGame created and published games](https://i.ibb.co/DzSK90G/Snimka-zaslona-2021-11-20-042044.png)

# Gym
*Gym is a toolkit for developing and comparing reinforcement learning algorithms. It makes no assumptions about the structure of your agent, and is compatible with any numerical computation library, such as TensorFlow or Theano.*

*The gym library is a collection of test problems — **environments** — that you can use to work out your reinforcement learning algorithms. These environments have a shared interface, allowing you to write general algorithms.*

[Website](https://gym.openai.com/)  
[>> Documentation <<](https://retro.readthedocs.io/en/latest/)  
[>> GitHub Repository <<](https://github.com/openai/gym)  

[>> Environments <<](https://gym.openai.com/envs/#classic_control)  

**Setup:**
- *base library:* `pip install gym`
- *dependency for one family:* `pip install gym[atari]`
- *all dependencies:* `pip install gym[all]`

*Minimum example of getting something running  ([cart - pole problem](https://youtu.be/J7E6_my3CHk)):*
```
import gym

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
	env.render()
	
	# the agent chooses an action, 
	# and the environment returns an observation and a reward
	
	action = env.action_space.sample()  # random action
	observation, reward, done, info = env.step(action)

	if done:
		# ...

env.close()
```

![Agent Environment Interaction](https://i.ibb.co/k0LZjPt/Snimka-zaslona-2021-11-20-042428.png)

**Tutorials and useful links**
- [Notable Related Libraries](https://github.com/openai/gym#notable-related-libraries)

---
- https://stable-baselines.readthedocs.io/en/master/
---
- https://deepmind.com/research
---
- https://www.py4u.net/discuss/232590
- https://github.com/openai/gym/tree/master/gym/envs
- https://stackoverflow.com/questions/49346051/openai-tensorflow-custom-game-environment-instead-of-using-gym-make
- https://github.com/ClarityCoders/SpaceShooter-Gym
- https://github.com/AGKhalil/BlockDude_CL/blob/master/gym_blockdude/envs/blockdude_hard_env.py
- [Training a neural network to play a game with TensorFlow and Open AI](https://youtu.be/3zeg7H6cAJw)
- https://www.youtube.com/watch?v=3zeg7H6cAJw
- https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a
- https://medium.com/analytics-vidhya/how-i-built-an-algorithm-to--takedown-atari-games-a13d3b3def69
- https://www.youtube.com/watch?v=osbmLJb2Tkc
