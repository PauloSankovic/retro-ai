class DummyAgent(object):
    def __init__(self, env):
        self.n = env.action_space.n

    def get_action(self, state):
        return 0 if state[2] < 0 else 1
