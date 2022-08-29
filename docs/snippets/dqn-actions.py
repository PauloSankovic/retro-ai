q_values = self(observations)       # online_net unaprijedni prolaz
actions = torch.argmax(q_values, dim=1).tolist()

for i in range(len(actions)):
    if random.random() <= epsilon:
        actions[i] = random.randint(0, self.n_actions - 1)