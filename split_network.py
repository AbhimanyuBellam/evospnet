import torch
import numpy as np
import torch
import torch.nn as nn
import sklearn
import numpy as np
import pandas as pd

import torch.nn.functional as F


class NetL4(nn.Module):
    def __init__(self, input_size=784, output_size=10, layers=[40]):
        super().__init__()
        self.d1 = nn.Linear(input_size, layers[0])
        self.d2 = nn.Linear(layers[0], output_size)
        self.layers = [self.d1, self.d2]

    def forward(self, X):
        X = F.relu(self.layers[0](X))
        X = self.layers[1](X)
        return F.log_softmax(X, dim=1)


class NetL3(nn.Module):
    def __init__(self, input_size=784, output_size=10, layers=[20]):
        super().__init__()
        self.d1 = nn.Linear(input_size, layers[0])
        self.d2 = nn.Linear(layers[0], output_size)
        self.layers = [self.d1, self.d2]

    def forward(self, X):
        X = F.relu(self.layers[0](X))
        X = self.layers[1](X)
        return F.log_softmax(X, dim=1)


class NetL2(nn.Module):
    def __init__(self, input_size=784, output_size=10, layers=[10]):
        super().__init__()
        self.d1 = nn.Linear(input_size, layers[0])
        self.d2 = nn.Linear(layers[0], output_size)
        self.layers = [self.d1, self.d2]

    def forward(self, X):
        X = F.relu(self.layers[0](X))
        X = self.layers[1](X)
        return F.log_softmax(X, dim=1)


class NetL1(nn.Module):
    def __init__(self, input_size=784, output_size=10, layers=[5]):
        super().__init__()
        self.d1 = nn.Linear(input_size, layers[0])
        self.d2 = nn.Linear(layers[0], output_size)
        self.layers = [self.d1, self.d2]

    def forward(self, X):
        X = F.relu(self.layers[0](X))
        X = self.layers[1](X)
        return F.log_softmax(X, dim=1)

# def split_network():


def combine_binary(net1, net2, combined_model_base=NetL2()):
    # combined_model_base = CombLevel1()
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


def multi_combine(networks, combined_model_base):

    with torch.no_grad():
        for layer_num in range(len(networks[0].layers)):
            out_net_layer_weights = None
            out_net_layer_bias = None
            for net_num, net in enumerate(networks):
                # if layer_num == 0:

                if net_num == 0:
                    out_net_layer_weights = net.layers[layer_num].weight
                    out_net_layer_bias = net.layers[layer_num].bias
                    continue

                curr_weights = net.layers[layer_num].weight
                curr_bias = net.layers[layer_num].bias

                axis = 1
                if layer_num == 0:
                    axis = 0

                # print("\nout_net_layer_weights", out_net_layer_weights)
                out_net_layer_weights = torch.cat(
                    (out_net_layer_weights, curr_weights), axis)

                if layer_num == len(net.layers)-1:
                    out_net_layer_bias = (out_net_layer_bias + curr_bias)/2
                else:
                    out_net_layer_bias = torch.cat(
                        (out_net_layer_bias, curr_bias), 0)

            combined_model_base.layers[layer_num].weight.data = out_net_layer_weights
            combined_model_base.layers[layer_num].bias.data = out_net_layer_bias

            # net1_layer_weights = net1.layers[layer_num].weight
            # net2_layer_weights = net2.layers[layer_num].weight

            # net1_layer_bias = net1.layers[layer_num].bias
            # net2_layer_bias = net2.layers[layer_num].bias

            # axis = 1
            # if layer_num == 0:
            #     axis = 0
            # combined_weights_matrix = torch.cat(
            #     (net1_layer_weights, net2_layer_weights), axis)

            # if layer_num == len(net1.layers)-1:
            #     #
            #     combined_bias = (net1_layer_bias + net2_layer_bias)/2
            # else:
            #     combined_bias = torch.cat(
            #         (net1_layer_bias, net2_layer_bias), 0)

            # print(
            #     "Actual:", combined_model_base.layers[layer_num].weight.data.shape)
            # print("to:", combined_weights_matrix.shape)

            # print("Ac Bias:", combined_bias.shape)
            # print("To bias:",
            #       combined_model_base.layers[layer_num].bias.data.shape)
            # combined_model_base.layers[layer_num].weight.data = combined_weights_matrix
            # combined_model_base.layers[layer_num].bias.data = combined_bias

    return combined_model_base


def combine_group_binary(networks, net_class):
    next_level_nets = []
    for i in range(len(networks)):
        if i % 2 == 0:
            print("combining:", i, i+1)
            next_level_nets.append(combine_binary(
                networks[i], networks[i+1], net_class()))
    return next_level_nets


def net_weights_avg(net):
    sum = 0
    for layer_num in range(len(net.layers)):
        net_layer_weights = net.layers[layer_num].weight
        sum += torch.sum(net_layer_weights)

    return sum


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net4 = NetL4()
    # basic_net.to(device)
    # print(net4)

    # type 1 - binary
    l1s = [NetL1() for i in range(8)]
    l2s = combine_group_binary(l1s, NetL2)
    l3s = combine_group_binary(l2s, NetL3)
    l4 = combine_group_binary(l3s, NetL4)[0]
    # print(l4s)

    print(net_weights_avg(l4))

    # type 2 - multi
    l4_m = multi_combine(l1s, NetL4())
    print(net_weights_avg(l4_m))
    # print(l4_m)
