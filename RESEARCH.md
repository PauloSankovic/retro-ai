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

## Gym
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


## Gym Retro
*OpenAi Gym extension.   
Gym Retro lets you turn classic video games into [Gym](https://gym.openai.com/) environments for reinforcement learning and comes with integrations for ~1000 games. It uses various emulators that support the [Libretro API](https://www.libretro.com/index.php/api/), making it fairly easy to add new emulators.*

[>> GitHub Repository <<](https://github.com/openai/retro)  
[>> Documentation <<](https://retro.readthedocs.io/en/latest/)  

**Setup:**
`pip install gym-retro`

*Minimum example of getting something running* (same as gym) *:*
```
import retro

def main():
    env = retro.make(game='Airstriker-Genesis')
    obs = env.reset()
    while True:
        observation, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()
```
`observation (numpy.array)` `[IMAGE, RAM]` &rarr; current state; RGB values of game pixels / RAM values ?!?! array of `65536` values  
`reward (float)` &rarr; the reward gained by the last performed action/during the previous step   
`done (boolean)` &rarr; flag that turns to `true` when the environment reaches an end point; no more actions can be performed and the environment needs to be reset  
 `info (dict)` &rarr; data for additional or debugging purposes; the data is not visible to the agent  

**PROS:**
- decent documentation and stack-overflow support (for `gym` in general)
- bunch of `Atari`, `Sega`, `NEC` & `Nintendo` games
- multiplayer environments
- replay files
- gameplay record
- playback
- render to video

**CONS:**
- observation array is somehow useful ?!?!

**The Integration UI:** *?! setup &rarr; how ?!*    
![Retro Gym Integration UI](https://user-images.githubusercontent.com/31688036/60402086-ca19ee80-9b8a-11e9-9eb0-662c9b4466dd.png)


**Tutorials and useful links:**
- [Gym Retro Template - *GitHub*](https://github.com/floydhub/gym-retro-template)
- [Integration UI google storage isn't working?](https://github.com/openai/retro/issues/227)
- [Retro Gym with Baselines](https://medium.com/aureliantactics/retro-gym-with-baselines-4-basic-usage-tips-1842d9aeff5)
- [Setup an openAI training environment with gym-retro and NEAT in Windows 7](https://youtu.be/j3eHWG2CtqU)
- [Ultimate Guide to Reinforcement Learning Part 1 — Creating a Game](https://towardsdatascience.com/ultimate-guide-for-reinforced-learning-part-1-creating-a-game-956f1f2b0a91)
- [Retro Contest](https://openai.com/blog/retro-contest/)
- [Player Games in OpenAI Retro](https://ai.stackexchange.com/questions/11174/2-player-games-in-openai-retro)
- [Is it possible to modify OpenAI environments?](https://stackoverflow.com/questions/53194107/is-it-possible-to-modify-openai-environments)

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
