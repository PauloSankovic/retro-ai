class RandomAgent:
    def __init__(self, env):
        self.action_space = env.action_space

    def get_action(self):
        return self.action_space.np_random.choice(self.action_space.n)
