
# Hyper parameters

# no of candidate solutions in the population
# If using DE initialised with SGD ensemble, ensure pop_size is equal to number of models available
pop_size = 120

# how many parts are made with the given input
num_split_parts = 1

# num of iterations for each split part, to have length of num_split_parts
num_iters = 150

# num cores to use for cost func evaluation
split_cores = 8

# number of groups to combine to form a higher level solution. Keep 2, if multiple, change function in code
num_groups = 2

# start off lowest level count, 8 fixed
start_level_count = 8

# no of iterations of the split DE process. For each split part, this value is same.
gen_split = 6700

# batch size for evaluation
batch_size = 100

# save directory
save_dir = "results"
# save_dir = "results/basic_net_res/"

# no of iterations for integral case
# this variable is for our assertion
total_gen_integral = 300

# To continue the generations from n+1th gen, the nth gen (candidate solutions npy file) should be present
# and to run the integral case alone after giving all the values in this file please use the following command in the cmd
# <mpirun -n 10 --mca btl ^openib python3 DE_v8_2.py> (where 10 is the no of cores) or refer readme file


# bounds for DE
# bound = [-5.9, 4.7]
bound = [-1.5, 1.5]
bias_bound = [-15, 15]

# mutation factor range and it can be between [0,2]
F = [0, 1]

# crossover probability and it can be between [0,1]
Cr = 0.8
