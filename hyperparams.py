
# Hyper parameters :
# no of candidate solutions in the population
pop_size = 4

# how many parts are made with the given input
num_split_parts = 4

# num of iterations for each split part, to have length of num_split_parts
num_iters = [10, 10, 10]

# ideally taking no of cores=split will decrease the time, or split_cores can be less than the split value
split_cores = num_split_parts

# no of iterations of the split DE process. For each split part, this value is same.
gen_split = 6700

# batch size for evaluation
batch_size = 1000

# save directory
save_dir = "results/basic_net_res"

# no of iterations for integral case
# this variable is for our assertion
total_gen_integral = 300

# To continue the generations from n+1th gen, the nth gen (candidate solutions npy file) should be present
# and to run the integral case alone after giving all the values in this file please use the following command in the cmd
# <mpirun -n 10 --mca btl ^openib python3 DE_v8_2.py> (where 10 is the no of cores) or refer readme file

# if n = 1 then the comnined_population.npy file should be present else pop_gen_"gen_inte_start+1".npy files should be present

# Staring iteration value for the integral case
gen_inte_start = 1

# Stopping iterations value for the integral case
gen_inte_stop = 300
# print(gen_inte_stop-gen_inte_start+1)

# ideally taking no of cores=10 will give maximum time utility
integral_cores = 10

# when gen_inte_start!=1, the path of the file should be given
# eg: "./Results/pop_gen_300.npy"
new_input_integral_fname = None
assert total_gen_integral == (gen_inte_stop-gen_inte_start+1)
"the two terms in the condiditon should be equal"


# Give the path of the file
input_file_name = "./input/case2_2k_proper.json"

# bounds for DE
bound = [-4, 4]

# mutation factor range and it can be between [0,2]
F = [-1, 1]

# crossover probability and it can be between [0,1]
Cr = 0.8

# For cost_function_v11
# Below two varibales are used to bring cf1 and cf2  to the same range
cf1_scale = 4
cf2_scale = 1

# These two variables are used to give weightages/importance for cf1 and cf2. Ideally 0.85 and 0.15 are used.
# cf1 and cf2 weightages must add upto 1
cf1_weightage = 0.85
cf2_weightage = 0.15
#..................#

# Only give the name of the file
output_json_file_name = "output_final.json"
analyser_file_name = "cf12_best_solution_final.xlsx"

# Please don't change min_time_org unless the code is understood
# it is used in calculating no of subslots for the agents. ( with the function <<get_nss_and_agent_costs>> )
min_time_org = 1

if __name__ == '__main__':
    # Bash script extracts split_cores and integral_cores from the print output
    print("{},{}".format(split_cores, integral_cores))
