self.fc1 = nn.Linear(in_features=input_size, out_features=128)
self.actor = nn.Linear(in_features=128, out_features=output_size)
self.critic = nn.Linear(in_features=128, out_features=1)