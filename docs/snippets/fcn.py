import torch.nn as nn

model = nn.Sequential(
    nn.Linear(in_features=4, out_features=8, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=8, out_features=6, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=6, out_features=2, bias=True),
)
