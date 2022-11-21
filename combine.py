import copy
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from base_net import BasicNet


# class CombLevel1(nn.Module):
#     # to have = [120, 84]
#     def __init__(self, input_size=784, output_size=10, layers=[20, 8]):
#         super().__init__()
#         self.d1 = nn.Linear(input_size, layers[0])  # hidden Layer 1
#         self.d2 = nn.Linear(layers[0], layers[1])
#         self.d3 = nn.Linear(layers[1], output_size)  # output Layer

#         self.layers = [self.d1, self.d2, self.d3]

#     def forward(self, X):
#         # print(X.shape)
#         X = F.relu(self.layers[0](X))
#         # print(X.shape)
#         X = F.relu(self.layers[1](X))
#         # print(X.shape)
#         X = self.layers[2](X)
#         # print(X.shape)
#         return F.log_softmax(X, dim=1)

class CombLevel1(nn.Module):
    # to have = [120, 84]
    def __init__(self, input_size=784, output_size=10, layers=[10]):
        super().__init__()
        self.d1 = nn.Linear(input_size, layers[0])  # hidden Layer 1
        self.d2 = nn.Linear(layers[0], output_size)  # output Layer

        self.layers = [self.d1, self.d2]

    def forward(self, X):
        # print(X.shape)
        X = F.relu(self.layers[0](X))
        # print(X.shape)
        X = self.layers[1](X)
        # print(X.shape)
        return F.log_softmax(X, dim=1)


root_dir = "/home/iec/abhimanyu/etc/splitnet/results/basic_net_res"
model_names = os.listdir(f"{root_dir}/weights")
model_paths = []
for i in range(len(model_names)):
    model_paths.append(f"{root_dir}/weights/{model_names[i]}")

# print(model_paths)

model1 = BasicNet()
model1.load_state_dict(torch.load(
    "results/basic_net_res/weights/sgd/basicnet/basicnet_0.pth"))

model2 = BasicNet()
model2.load_state_dict(torch.load(
    "results/basic_net_res/weights/sgd/basicnet/basicnet_1.pth"))

model_comb1 = CombLevel1()

# TODO to combine in order of fitness


def combine(net1, net2, combined_model_base):
    combined_model_base = CombLevel1()
    with torch.no_grad():
        for layer_num in range(len(net1.layers)):
            net1_layer_weights = net1.layers[layer_num].weight
            net2_layer_weights = net2.layers[layer_num].weight

            net1_layer_bias = net1.layers[layer_num].bias
            net2_layer_bias = net2.layers[layer_num].bias

            axis = 1
            if layer_num == 0:
                axis = 0
            combined_weights_matrix = torch.cat(
                (net1_layer_weights, net2_layer_weights), axis)

            if layer_num == len(net1.layers)-1:
                #
                combined_bias = (net1_layer_bias + net2_layer_bias)/2
            else:
                combined_bias = torch.cat(
                    (net1_layer_bias, net2_layer_bias), 0)

            print(
                "Actual:", combined_model_base.layers[layer_num].weight.data.shape)
            print("to:", combined_weights_matrix.shape)

            print("Ac Bias:", combined_bias.shape)
            print("To bias:",
                  combined_model_base.layers[layer_num].bias.data.shape)
            combined_model_base.layers[layer_num].weight.data = combined_weights_matrix
            combined_model_base.layers[layer_num].bias.data = combined_bias

    return combined_model_base


print(model_comb1.layers[0].weight.data.shape)
changed_model = combine(model1, model2, model_comb1)
print("\n", changed_model.layers[0].weight.data.shape)

# print("DIFF:\n")
# print(changed_model.layers[0].weight.data -
#       model_comb1.layers[0].weight.data)

torch.save(changed_model.state_dict(),
           "results/basic_net_res/weights/sgd/combined_models/comb_1.pth")
