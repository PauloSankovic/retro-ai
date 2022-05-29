import torch
import torch.nn as nn

from .structures import CnnStructure


def build_fc(dims: list[int]):
    layers = []
    for i in range(1, len(dims)):
        layers.append(nn.Linear(in_features=dims[i - 1], out_features=dims[i], bias=True))
        if i != len(dims) - 1:
            layers.append(nn.ReLU())
    return layers


def fc(fc_dims: list[int]) -> nn.Sequential:
    layers = build_fc(fc_dims)

    net = nn.Sequential(*layers)
    return net


def cnn(observation_space, cnn_structure: list[CnnStructure], fc_hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    cnn_layers = []
    for i, structure in enumerate(cnn_structure):
        conv = nn.Conv2d(
            in_channels=structure.in_channels,
            out_channels=structure.out_channels,
            kernel_size=structure.kernel_size,
            stride=structure.stride,
            padding=structure.padding,
            bias=True
        )

        cnn_layers.append(conv)
        if i != len(cnn_structure) - 1:
            cnn_layers.append(nn.ReLU())

    with torch.no_grad():
        seq = nn.Sequential(*cnn_layers, nn.Flatten())
        tensors = seq(torch.tensor(observation_space.sample()[None]).float()).shape[1]

    fc_layers = build_fc([tensors, *fc_hidden_dims, out_dim])

    net = nn.Sequential(*cnn_layers, nn.Flatten(), *fc_layers)
    return net
