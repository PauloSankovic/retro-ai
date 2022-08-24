import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=18, kernel_size=(3, 3), stride=(1, 1), padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=18 * 16 * 16, out_features=64),
    nn.Linear(in_features=64, out_features=10)
)
