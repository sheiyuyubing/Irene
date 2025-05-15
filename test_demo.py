from net import *
import torch
net = PolicyNetwork()

input_data = torch.randn(100,15,19,19)

output_data = net(input_data)

print(output_data.shape)

print(output_data)
