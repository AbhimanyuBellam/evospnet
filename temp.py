import matplotlib.pyplot as plt
import numpy as np

import torch
from base_net import BasicNet


base = BasicNet

print(base())

# file_path = f"/home/iec/abhimanyu/etc/splitnet/results/basic_net_res/plots/temp.jpg"

# model_path = "results/basic_net_res/weights/sgd/basicnet/basicnet_0.pth"
# model_path = "results/basic_net_res/weights/tests/encode_decode1.pth"

# network = BasicNet()
# network.load_state_dict(torch.load(model_path))
# network = network.float()
# # print(network.layers[0].weight.data)

# layer0 = network.layers[0].weight.data.flatten()
# print(torch.min(layer0), torch.max(layer0))

# layer1 = network.layers[1].weight.data.flatten()
# print(torch.min(layer1), torch.max(layer1))

# layer0 = network.layers[0].bias.data.flatten()
# print(torch.min(layer0), torch.max(layer0))

# layer1 = network.layers[1].bias.data.flatten()
# print(torch.min(layer1), torch.max(layer1))

# x =[i+1 for i in range(10)]
# y1 = [i//2 for i in range(10)]
# y2 = [i*2 for i in range(10)]

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, 'b-')

# ax1.set_xlabel("Generations" )
# ax1.set_ylabel("Test Cross Entropy Loss", color='b')
# ax2.set_ylabel("Test Accuracy", color='g')
# plt.savefig(file_path)


# import matplotlib.pyplot as plt
# x = np.arange(0, 10, 0.1)
# y1 = 0.05 * x**2
# y2 = -1 *y1

# fig, ax1 = plt.subplots()

# ax2 = ax1.twinx()
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, 'b-')

# ax1.set_xlabel('X data')
# ax1.set_ylabel('Y1 data', color='g')
# ax2.set_ylabel('Y2 data', color='b')

# plt.savefig("uuh.jpg")
