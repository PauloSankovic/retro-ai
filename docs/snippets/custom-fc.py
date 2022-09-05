from networks import fc

net = fc([4, 128, 64, 2])
print(net)

# Sequential(
#   (0): Linear(in_features=4, out_features=128, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=128, out_features=64, bias=True)
#   (3): ReLU()
#   (4): Linear(in_features=64, out_features=2, bias=True)
# )