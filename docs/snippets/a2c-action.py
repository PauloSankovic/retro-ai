action_probability, state_value = self.forward(state)

m = Categorical(action_probability)
action = m.sample()

self.actions_history.append((m.log_prob(action), state_value))