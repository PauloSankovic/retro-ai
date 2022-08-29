from networks import cnn, CnnStructure

cnn_layers = [
    CnnStructure(in_channels=4, out_channels=32, kernel_size=8, stride=4),
]
net = cnn(observation_space, cnn_layers, [512], 4)
print(net)

# Sequential(
#   (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
#   (1): Flatten(start_dim=1, end_dim=-1)
#   (2): Linear(in_features=12800, out_features=512, bias=True)
#   (3): ReLU()
#   (4): Linear(in_features=512, out_features=4, bias=True)
# )