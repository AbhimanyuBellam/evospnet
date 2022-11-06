import random
import copy
import time
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import hyperparams as hyperparams
from differential_evolution import DifferentialEvolution
from base_net import BasicNet
from dataset import train_loader, test_loader
from differential_evolution import Candidate


class NeuroEvolution:
    def __init__(self):
        self.population_size = hyperparams.pop_size
        self.num_split_parts = hyperparams.num_split_parts
        self.num_iters = hyperparams.num_iters
        self.neural_cost_func = nn.CrossEntropyLoss()

        self.temp_net = BasicNet()
        self.decode_weights_lengths = []
        self.decode_weights_shapes = []

        self.decode_bias_lengths = []
        self.decode_bias_shapes = []

        self.network_dimensionality = len(
            self.encode_network(self.temp_net, decode_setter=True))

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = hyperparams.save_dir

    def decode_candidate(self, candidate):
        print("Decoding")
        for i in range(len(self.decode_weights_lengths)):
            if i == 0:
                # weights
                layer_weights = candidate[0:self.decode_weights_lengths[i]]

            else:
                # weights
                layer_weights = candidate[self.decode_bias_lengths[i-1]                                          :self.decode_weights_lengths[i]]

            layer_weights.reshape(self.decode_weights_shapes[i])

            # bias
            layer_bias = candidate[self.decode_weights_lengths[i]                                   :self.decode_bias_lengths[i]]
            layer_bias.reshape(self.decode_bias_shapes[i])

            # replace weights of temp net with new weights to calc cost -
            # TODO make it access different basicnet for parallel access, map candidate to network
            temp_net = BasicNet().to(self.device)
            with torch.no_grad():
                for j in range(len(temp_net.layers)):
                    temp_net.layers[i].weight.data = layer_weights
                    temp_net.layers[i].bias.data = layer_bias
        return temp_net

    def cost_func(self, candidate):
        print("Finding cost")
        # decode candidate
        network = self.decode_candidate(candidate)

        # evaluate cost with all inputs
        net_out = network.forward()

        # train_loss=[]
        train_acc = 0
        loss = 0
        for b, (X_train, y_train) in enumerate(train_loader):
            b += 1
            # run model, get loss
            # flatten and evaluate
            y_pred = network.forward(X_train.view(hyperparams.batch_size, -1))
            loss += self.neural_cost_func(y_pred, y_train)
            # gen_train_loss.append(loss)

            predicted = torch.max(y_pred.data, 1)[1]

            # train_acc += (predicted == y_test).sum()
        # gen_train_loss_total = sum(gen_train_loss)

        # destroy network object
        del network
        return loss

    def encode_network(self, network, decode_setter=False):
        print("Encoding network")
        layers = network.layers
        flattened_array = []

        for layer in layers:
            # weights
            weights = layer.weight.flatten()

            weights = weights.cpu().detach().tolist()
            weights_len = len(weights)

            # bias
            bias = layer.bias.flatten()

            bias = bias.cpu().detach().tolist()
            bias_len = len(bias)

            if decode_setter:
                self.decode_weights_shapes.append(weights.shape)
                self.decode_weights_lengths.append(weights_len)
                self.decode_bias_shapes.append(bias.shape)
                self.decode_bias_lengths.append(bias_len)

            layer_params = weights+bias
            flattened_array += layer_params

        return np.array(flattened_array)

    def initialize_population(self):
        print("Initializing population")
        self.all_split_population = []
        for i in range(self.num_split_parts):
            split_part_population = []
            for j in range(self.population_size):
                each_net = BasicNet()
                # convert network to 1D array
                flattened_net = self.encode_network(each_net)
                candidate = Candidate(flattened_net)
                split_part_population.append(candidate)
            self.all_split_population.append(split_part_population)
        # print("All pop:", self.all_split_population)

    def plot_results(self, save_dir, total_scores, split_num):
        file_path = f"{save_dir}/plots/split_{split_num}"
        fig, ax = plt.subplots()
        ax.plot(total_scores, [i for i in range(
            len(total_scores))], color="red", marker="o", label="gen_best_loss")
        ax.set_xlabel("Generations", fontsize=10)
        ax.set_ylabel("Cross Entropy Loss", fontsize=10)
        plt.legend(loc="upper left")
        fig.savefig(file_path,
                    format='jpeg',
                    dpi=100,
                    bbox_inches='tight')

    def run_evolutions(self):
        self.initialize_population()
        # TODO to run in parallel
        for i in range(len(self.all_split_population)):
            population = self.all_split_population[i]
            # temp_nets = [BasicNet() for j in range(len(population))]
            evolution_algo = DifferentialEvolution(
                cost_func=self.cost_func, dimensionality=self.network_dimensionality)

            gen_sol, total_pop, gen_avg_array, total_scores = evolution_algo.evolve(population, len(
                population), self.num_iters[i])

            self.plot_results(self.save_dir, total_scores, i)


"""
part 1, part 2 are lists of models in sorted order of fitness values
"""


def combine_in_order_of_fitness(part1, part2):
    combined_part = None
    return combined_part


if __name__ == "__main__":
    neuroevolution = NeuroEvolution()
    neuroevolution.run_evolutions()