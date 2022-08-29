cum_reward = 0
expected_returns = []
for reward in self.rewards[::-1]:
    cum_reward = reward + self.gamma * cum_reward
    expected_returns.insert(0, cum_reward)

...

actor_losses = []
critic_losses = []
for (action_prob, state_value), reward in zip(self.actions_history, expected_returns):
    advantage = reward - state_value.item()

    actor_losses.append(-action_prob * advantage)
    critic_losses.append(F.huber_loss(state_value, torch.tensor([reward])))

optimizer.zero_grad()
loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()

loss.backward()
optimizer.step()