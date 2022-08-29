x = F.relu(self.fc1(x))

action_probability = F.softmax(self.actor(x), dim=-1)
state_value = self.critic(x)