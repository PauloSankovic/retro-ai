with torch.no_grad():
    target_q_values = target_net(next_states)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rewards + gamma * (1 - dones) * max_target_q_values

q_values = self(states)

action_q_values = torch.gather(input=q_values, dim=1, index=actions)
loss = F.huber_loss(action_q_values, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()