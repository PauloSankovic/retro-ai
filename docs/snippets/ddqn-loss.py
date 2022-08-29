with torch.no_grad():
    online_q_values = self(next_states)
    max_online_actions = online_q_values.argmax(dim=1, keepdim=True)

    target_q_values = target_net(next_states)
    action_q_values = torch.gather(input=target_q_values, dim=1, index=max_online_actions)

    targets = rewards + gamma * (1 - dones) * action_q_values

q_values = self(states)
action_q_values = torch.gather(input=q_values, dim=1, index=actions)

loss = F.huber_loss(action_q_values, targets)
...