import random
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt

import hyperparams as hyperparams
# from differential_evolution import DifferentialEvolution
# from base_net import BasicNet
from dataset import train_loader, test_loader
# from differential_evolution import Candidate
# from initialization import initialization
from split_network import NetL1, NetL2, NetL3, NetL4
from split_network import combine_group_binary, combine_binary, multi_combine

import multiprocessing as mp
# torch.multiprocessing.set_start_method('spawn')


class MultiNeuroEvolution:
    def __init__(self, net_classes):
        self.population_size = hyperparams.pop_size
        self.num_split_parts = hyperparams.num_split_parts
        self.num_iters = hyperparams.num_iters
        self.neural_cost_func = nn.CrossEntropyLoss()

        self.net_classes = net_classes
        self.current_net_class = net_classes[0]
        # self.temp_net = BasicNet()
        self.decode_weights_lengths = []
        self.decode_weights_shapes = []

        self.decode_bias_lengths = []
        self.decode_bias_shapes = []

        self.network_dimensionality = len(
            self.encode_network(self.current_net_class(), decode_setter=True))

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = None

        # self.temp_nets = [self.current_net_class.to(self.device)
        #                   for i in range(self.population_size)]
        self.counter_cost_fun = 0

        # self.bounds = [hyperparams.bound for i in range(
        #     self.network_dimensionality)]

    def generate_bounds(self, network):
        # net = BasicNet()
        self.bounds = []
        layers = network.layers
        for layer in layers:
            # weights
            weights = layer.weight.flatten()
            weights_len = len(weights)
            for i in range(weights_len):
                self.bounds.append(hyperparams.bound)

            # bias
            bias = layer.bias.flatten()
            bias_len = len(bias)
            for i in range(bias_len):
                self.bounds.append(hyperparams.bias_bound)
        # print("bounds:", self.bounds)

    def decode_candidate(self, candidate, index):
        # print("Decoding")
        # print(self.decode_weights_lengths)
        # print(self.decode_weights_shapes)
        # print(self.decode_bias_lengths)
        # print(self.decode_bias_shapes)
        # print("_____")

        # temp_net = self.temp_nets[index]
        # print("cand shape:", candidate.shape)
        # temp_net = BasicNet()
        temp_net = self.current_net_class()

        with torch.no_grad():
            for i in range(len(self.decode_weights_lengths)):
                if i == 0:
                    # weights
                    layer_weights = candidate[0:self.decode_weights_lengths[i]]

                else:
                    # weights
                    layer_weights = candidate[self.decode_bias_lengths[i-1]                                              :self.decode_weights_lengths[i]]

                # print("curr:", layer_weights.shape)
                # print("TO:", self.decode_weights_shapes[i])
                layer_weights = layer_weights.reshape(
                    list(self.decode_weights_shapes[i]))
                # print(layer_weights.shape)

                # bias
                layer_bias = candidate[self.decode_weights_lengths[i]                                       :self.decode_bias_lengths[i]]
                layer_bias = layer_bias.reshape(
                    list(self.decode_bias_shapes[i]))

                # replace weights of temp net with new weights to calc cost -
                # TODO make it access different basicnet for parallel access, map candidate to network
                # temp_net = BasicNet().to(self.device)

                # for j in range(len(temp_net.layers)):
                temp_net.layers[i].weight.data = torch.from_numpy(
                    layer_weights).to(self.device)
                temp_net.layers[i].bias.data = torch.from_numpy(
                    layer_bias).to(self.device)

        return temp_net

    def cost_func(self, candidate):
        self.counter_cost_fun += 1
        temp_index = self.counter_cost_fun % self.population_size
        start_time = time.time()

        # print("Finding cost")
        # decode candidate
        network = self.decode_candidate(
            candidate, temp_index)

        # train_loss=[]
        train_acc = 0
        loss = 0
        # network.eval()
        with torch.no_grad():
            for b, (X_train, y_train) in enumerate(train_loader):
                # run model, get loss
                # flatten and evaluate
                # print("X_shape:", X_train.shape)
                input_ = X_train.view(
                    hyperparams.batch_size, -1).to(self.device)
                y_train = y_train.to(self.device)
                # print("inp shape:", input_.shape)
                y_pred = network.forward(input_)
                loss += self.neural_cost_func(y_pred.to(self.device),
                                              y_train.to(self.device)).data.item()
                # gen_train_loss.append(loss)

                predicted = torch.max(y_pred.data, 1)[1]

                train_acc += (predicted == y_train).sum()
        # print("Train acc:", (train_acc)/(b*hyperparams.batch_size))
        # gen_train_loss_total = sum(gen_train_loss)
        loss /= len(train_loader)

        # destroy network object
        del network
        # print("Time CF:", time.time()-start_time)
        print(loss)
        return loss

    def encode_network(self, network, decode_setter=False):
        # print("Encoding network")
        layers = network.layers
        flattened_array = []
        position_weights = 0
        position_bias = 0
        for layer in layers:
            # weights
            weights = layer.weight.flatten()

            # weights = weights.cpu().detach().tolist()
            weights_len = len(weights)
            position_weights += weights_len

            # bias
            bias = layer.bias.flatten()

            # bias = bias.cpu().detach().tolist()
            bias_len = len(bias)
            position_bias = position_weights+bias_len

            if decode_setter:
                self.decode_weights_shapes.append(layer.weight.shape)
                self.decode_weights_lengths.append(position_weights)
                self.decode_bias_shapes.append(layer.bias.shape)
                self.decode_bias_lengths.append(position_bias)

            position_weights += bias_len

            layer_params = weights.cpu().detach().tolist() + \
                bias.cpu().detach().tolist()
            flattened_array += layer_params
            # print("Flattened_ len:", len(flattened_array))
        return np.array(flattened_array, dtype=np.float32)

    def cost_func_temp(self, candidate):
        candidate = np.float32(candidate)
        self.counter_cost_fun += 1
        temp_index = self.counter_cost_fun % self.population_size
        start_time = time.time()

        network = self.decode_candidate(candidate, temp_index)

        train_acc = 0
        loss = 0
        network.eval()
        with torch.no_grad():
            for b, (X_train, y_train) in enumerate(train_loader):
                input_ = X_train.view(
                    hyperparams.batch_size, -1).to(self.device)
                y_train = y_train.to(self.device)
                y_pred = network.forward(input_)
                loss += self.neural_cost_func(y_pred.to(self.device),
                                              y_train.to(self.device)).data.item()

                predicted = torch.max(y_pred.data, 1)[1]

                train_acc += (predicted == y_train).sum()
        # print("\nTrain acc:", (train_acc)/(b*hyperparams.batch_size))
        # gen_train_loss_total = sum(gen_train_loss)
        loss /= len(train_loader)

        # destroy network object
        del network
        # print("Time CF:", time.time()-start_time)
        # print("Loss:", loss)
        return loss

    def cost_func_with_acc(self, candidate):
        candidate = np.float32(candidate)
        self.counter_cost_fun += 1
        temp_index = self.counter_cost_fun % self.population_size
        start_time = time.time()

        network = self.decode_candidate(
            candidate, temp_index)

        train_acc = 0
        loss = 0
        # network.eval()
        with torch.no_grad():
            for b, (X_train, y_train) in enumerate(train_loader):
                input_ = X_train.view(
                    hyperparams.batch_size, -1).to(self.device)
                y_train = y_train.to(self.device)
                # print("inp shape:", input_.shape)
                y_pred = network.forward(input_)
                loss += self.neural_cost_func(y_pred.to(self.device),
                                              y_train.to(self.device)).data.item()

                predicted = torch.max(y_pred.data, 1)[1]
                train_acc += (predicted == y_train).sum()

        train_acc = (train_acc)/(b*hyperparams.batch_size)
        # print("Train acc:", train_acc)
        # gen_train_loss_total = sum(gen_train_loss)
        loss /= len(train_loader)

        # destroy network object
        del network
        # print("Time CF:", time.time()-start_time)
        # print("Loss:", loss)
        return loss, train_acc

    def cost_func_test_set_with_acc(self, candidate):
        candidate = np.float32(candidate)
        self.counter_cost_fun += 1
        temp_index = self.counter_cost_fun % self.population_size
        start_time = time.time()
        network = self.decode_candidate(
            candidate, temp_index)

        test_acc = 0
        loss = 0
        # network.eval()
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                input_ = X_test.view(
                    hyperparams.batch_size, -1).to(self.device)
                y_test = y_test.to(self.device)
                # print("inp shape:", input_.shape)
                y_pred = network.forward(input_)
                loss += self.neural_cost_func(y_pred.to(self.device),
                                              y_test.to(self.device)).data.item()

                predicted = torch.max(y_pred.data, 1)[1]

                test_acc += (predicted == y_test).sum()
        test_acc = (test_acc)/(b*hyperparams.batch_size)
        # print("Test acc:", test_acc)
        # gen_train_loss_total = sum(gen_train_loss)
        loss /= len(test_loader)

        # destroy network object
        del network
        # print("Time CF:", time.time()-start_time)
        # print("Loss:", loss)
        return loss, test_acc

    def run_basic_DE(self, best_solution, exp_name):
        from scipy.optimize import differential_evolution

        self.generate_bounds(network=self.current_net_class())

        self.decode_weights_lengths = []
        self.decode_weights_shapes = []

        self.decode_bias_lengths = []
        self.decode_bias_shapes = []
        self.network_dimensionality = len(
            self.encode_network(self.current_net_class(), decode_setter=True))

        print("Initializing population")
        split_part_population = []

        for i in range(self.population_size):
            vector = np.random.uniform(-1, 1, self.network_dimensionality)
            # split_part_population.append(np.array(init_vectors[i]))
            split_part_population.append(vector)

        # print(split_part_population)
        split_part_population = np.array(split_part_population)
        print("Population shape:", split_part_population.shape)

        self.gen_vectors = []
        self.convergence_history = []

        def store_callback(result, convergence):
            self.gen_vectors.append(result)
            self.convergence_history.append(convergence)
            # print("Converagence:", convergence)
            # print(self.gen_vectors)

        # print(split_part_population)
        # for l in range(len(split_part_population)):
        #     print(split_part_population[i].shape)
        # print(self.current_net_class)
        # print("bounds:", len(self.bounds))
        # print("iters, cores = ", hyperparams.num_iters, hyperparams.split_cores)

        # pool = mp.Pool(hyperparams.split_cores)
        if best_solution is not None:
            print("With best")
            print("Best sol shape:", best_solution.shape)
            result = differential_evolution(self.cost_func_temp, self.bounds, callback=store_callback, polish=False, x0=best_solution,
                                            init=split_part_population, maxiter=hyperparams.num_iters, workers=hyperparams.split_cores, seed=1, disp=True, updating='deferred')
        else:
            print("No best")
            result = differential_evolution(self.cost_func_temp, self.bounds, callback=store_callback, polish=False,
                                            init=split_part_population, maxiter=hyperparams.num_iters, workers=hyperparams.split_cores, seed=1, disp=True, updating='deferred')

        print("RESULT:", result.x, result.fun)

        file_path = f"{self.save_dir}/weights/{exp_name}.pth"
        best_net = self.decode_candidate(
            result.x, index=0)
        torch.save(best_net.state_dict(), file_path)

        return result.x, best_net

        # evaluate cost for every gen best vector again for plots

    def run_multi_evolutions(self):
        lowest_level_count = hyperparams.start_level_count
        current_max_count = copy.deepcopy(lowest_level_count)
        best_comb_solutions = []
        gen_best_nets = []
        gen_best_solutions = []
        gen_best_combined_sols = None

        for i in range(len(self.net_classes)):

            self.save_dir = f"{hyperparams.save_dir}/L{i+1}"
            self.current_net_class = self.net_classes[i]

            if i != 0:
                gen_best_combined_nets = combine_group_binary(
                    gen_best_nets, net_class=self.current_net_class)
                gen_best_combined_sols = []

                for j in range(len(gen_best_combined_nets)):
                    encoded_vec = self.encode_network(
                        gen_best_combined_nets[j])
                    gen_best_combined_sols.append(encoded_vec)

            # print(self.current_net_class())
            # print("DECODES")
            # print(self.decode_weights_lengths)
            # print(self.decode_weights_shapes)

            # print(self.decode_bias_lengths)
            # print(self.decode_bias_shapes)
            # # print("________")
            print(f"\nRunning {current_max_count} parts")
            gen_best_nets = []
            gen_best_solutions = []
            for cur_count in range(current_max_count):
                print(f"\n\n ---   Net class: {i+1}, Part:{cur_count+1}")
                if i == 0:
                    best_solution = None
                else:
                    best_solution = gen_best_combined_sols[cur_count]

                exp_name = f"class_{i}_part_{cur_count}"

                gen_best_sol, gen_best_net = self.run_basic_DE(
                    best_solution, exp_name=exp_name)

                print("Generating results")
                self.generate_results(exp_name)
                self.generate_test_results(exp_name)

                gen_best_solutions.append(gen_best_sol)
                gen_best_nets.append(gen_best_net)

                print("\n Sleeping for 10 sec")
                time.sleep(10)

            current_max_count = current_max_count // 2

    # initialization from ensemble of SGD outputs
    def run_SDG_init_DE(self):
        from scipy.optimize import differential_evolution
        import os

        # self.generate_bounds()
        self.generate_bounds(network=self.current_net_class())
        root_dir = "results/basic_net_res/weights/sgd/basicnet"
        model_names = os.listdir(root_dir)
        model_paths = []
        for i in range(len(model_names)):
            model_paths.append(f"{root_dir}/{model_names[i]}")

        print("Initializing population")
        split_part_population = []

        # init_vectors = initialization(
        #     np.array(self.bounds), self.population_size, technique=2)

        best_cand_index = None
        temp_score = 99999999
        # initialize and get best cand
        for i in range(len(model_paths)):
            each_net = BasicNet()
            # convert network to 1D array
            each_net.load_state_dict(torch.load(model_paths[i]))
            flattened_net = self.encode_network(each_net)
            score = self.cost_func_temp(flattened_net)
            if score < temp_score:
                best_cand_index = i
                temp_score = score
                print("path:", model_paths[i])
                print(best_cand_index, temp_score)
            split_part_population.append(flattened_net)

        print("From SGD len:", len(split_part_population))
        # sys.exit()

        for i in range(self.population_size - len(model_paths)):
            vector = np.random.uniform(-1, 1, self.network_dimensionality)
            # split_part_population.append(np.array(init_vectors[i]))
            split_part_population.append(vector)
        print("From SGD + uniform len:", len(split_part_population))

        # testing
        # for i in range(len(split_part_population)):
        #     # encoded = self.encode_network(split_part_population[i])
        #     # decoded = self.decode_candidate(encoded, index=0)
        #     print("Train loss:", self.cost_func_temp(split_part_population[i]))
        # sys.exit()

        # print(split_part_population)
        split_part_population = np.array(split_part_population)
        print(split_part_population.shape)

        self.gen_vectors = []
        self.convergence_history = []

        def store_callback(result, convergence):
            self.gen_vectors.append(result)
            self.convergence_history.append(convergence)
            # print("Converagence:", convergence)
            # print(self.gen_vectors)

        result = differential_evolution(self.cost_func_temp, self.bounds, callback=store_callback, polish=False, x0=split_part_population[best_cand_index],
                                        init=split_part_population, maxiter=hyperparams.num_iters, workers=hyperparams.split_cores, disp=True)
        print("RESULT:", result.x, result.fun)

        file_path = f"{self.save_dir}/weights/split_from_SGD_ensemble.pth"
        best_net = self.decode_candidate(
            result.x, index=0, net_class=self.current_net_class)
        torch.save(best_net.state_dict(), file_path)

    def generate_results(self, exp_name):
        all_gen_train_loss = []
        all_gen_train_acc = []

        for cand_vec in self.gen_vectors:
            loss, acc = self.cost_func_with_acc(cand_vec)
            all_gen_train_loss.append(loss)
            all_gen_train_acc.append(acc.item())
        # print(" ACC:", all_gen_train_acc)
        # print("LOSS:", all_gen_train_loss)
        file_path = f"{self.save_dir}/plots/train_{exp_name}.jpg"
        x = [i+1 for i in range(len(all_gen_train_loss))]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(x, all_gen_train_loss, 'b-')
        ax2.plot(x, all_gen_train_acc, 'g-')

        ax1.set_xlabel("Generations")
        ax1.set_ylabel("Train Cross Entropy Loss", color='b')
        ax2.set_ylabel("Train Accuracy", color='g')
        fig.savefig(file_path)

        file_path2 = f"{self.save_dir}/plots/train_convergence_{exp_name}.jpg"
        fig2, ax3 = plt.subplots()
        ax3.plot([i+1 for i in range(len(self.convergence_history))],
                 self.convergence_history)
        ax3.set_xlabel("Generations")
        ax3.set_ylabel("Population Convergence")
        fig2.savefig(file_path2)

    def generate_test_results(self, exp_name):
        all_gen_test_loss = []
        all_gen_test_acc = []

        for cand_vec in self.gen_vectors:
            loss, acc = self.cost_func_test_set_with_acc(cand_vec)
            all_gen_test_loss.append(loss)
            all_gen_test_acc.append(acc.item())

        file_path = f"{self.save_dir}/plots/test_{exp_name}.jpg"
        x = [i+1 for i in range(len(all_gen_test_loss))]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(x, all_gen_test_loss, 'b-')
        ax2.plot(x, all_gen_test_acc, 'g-')

        ax1.set_xlabel("Generations")
        ax1.set_ylabel("Test Cross Entropy Loss", color='b')
        ax2.set_ylabel("Test Accuracy", color='g')
        # plt.savefig(file_path)
        fig.savefig(file_path)


"""
part 1, part 2 are lists of models in sorted order of fitness values
"""


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    neuroevolution = MultiNeuroEvolution(
        net_classes=[NetL1, NetL2, NetL3, NetL4])
    # neuroevolution.run_evolutions()
    start = time.time()
    # neuroevolution.run_basic_DE()
    # neuroevolution.run_SDG_init_DE()
    neuroevolution.run_multi_evolutions()
    print("Evolution time:", time.time()-start)
    # neuroevolution.generate_results()
    # neuroevolution.generate_test_results()
    print("Total time:", time.time()-start)
