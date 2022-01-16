# Useful Algorithms

## Q-learning

**Q** - _quality_ - how useful a given action is in gaining some future reward

Reinforcement learning algorithm that seeks to find the best action to take given the current state.
It does not require a model of the environment (hence "model-free"), and it can handle problems with **stochastic transitions** and reward without requiring adaptations.

For any finite _Markov Decision Process_ (FMDP), Q-learning finds an optimal policy in the sense of maximizing the expected value of the total reward over any and all successive steps, starting from the current state. Q-learning can identify an optimal action-selection policy for any given FMDP, given infinite exploration time and a partly-random policy.

### 1. Create a Q-table and initialize it

Matrix with shape of `[state_size, action_size]`, initialized to 0.
We then update and store our _q-values_ after an episode.
Q-table is a reference table for our agent to select the best action based on the _q-value_.

```python
import numpy as np

qtable = np.zeros((state_size, action_size))
```

### 2. Q-learning

The next step is simply for the agent to interact with the environment and make updates to the state action pairs in our _q-table_.

An agent interacts with the environment in 1 of 2 ways:

The first is to use the _q-table_ as a reference and view all possible actions for a given state. 
The agent then selects the action based on the max value of those actions.
This is known as **_exploiting_** since we use the information we have available to us to make a decision.

The second way to take action is to act randomly. This is called _**exploring**_.
Instead of selecting actions based on the max future reward, we select an action at random. 
Acting randomly is important because it allows the agent to explore and discover new states that otherwise may not be selected during the exploitation process.

You can balance this two options (_exploration/exploitation_) using epsilon (ε) and setting the value of how often you want to explore vs exploit.

```python
import random

# Set the percent you want to explore
epsilon = 0.2

if random.uniform(0, 1) < epsilon:
    # Explore: select a random action

else:
    # Exploit: select the action with max value (future reward)
```

### 3. Updating the Q-table

The updates occur after each step or action and ends when an episode is done.
The agent will nt learn much after a single episode, but eventually with enough exploring (steps and episodes) it will converge and learn the optimal q-values (_q-star_, _Q*_).

```python
Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])
```

- [Simple Reinforcement Learning: Q-learning](https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56)
- [FrozenLake example](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb)
- [Great RL and q-learning example using the OpenAI Gym taxi environment](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
- [Reinforcement Learning: An Introduction (free book by Sutton)](http://www.incompleteideas.net/book/RLbook2018trimmed.pdf)
- [Diving deeper into Reinforcement Learning with Q-Learning](https://www.freecodecamp.org/news/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe/)
- [An introduction to Reinforcement Learning](https://www.freecodecamp.org/news/an-introduction-to-reinforcement-learning-4339519de419/)

## Deep Q-learning

## Double Q-learning

## Monte Carlo strategy

## Genetic algorithms (Evolution strategy)
