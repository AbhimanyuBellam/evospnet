import random
import copy
import time
import numpy as np
import torch.nn as nn
import hyperparams as hyperparams


class Candidate(object):
    def __init__(self, vector):
        self.vector = vector
        self.cost = -1


class DifferentialEvolution:
    """
    From Zhang's paper
    """

    def __init__(self, cost_func, dimensionality):
        self.cost_func = cost_func
        self.dimensionality = dimensionality
        self.bounds = [hyperparams.bound for i in range(self.dimensionality)]
        self.recombination = hyperparams.Cr

    @staticmethod
    def sigmoid(x):
        return 1.0/(1+np.exp(-x))

    @staticmethod
    def ensure_bounds(vec, bounds, tech=1):
        vec_new = []
        # assigning fixed bounds
        if(tech == 0):
            # cycle through each variable in vector
            for i in range(len(vec)):
                # variable exceedes the minimum boundary
                if vec[i] < bounds[i][0]:
                    vec_new.append(bounds[i][0])
                    # print("min:",vec[i])
                # variable exceedes the maximum boundary
                if vec[i] > bounds[i][1]:
                    vec_new.append(bounds[i][1])
                    # print("max:",vec[i])
                # the variable is fine
                if bounds[i][0] <= vec[i] and vec[i] <= bounds[i][1]:
                    vec_new.append(vec[i])
        # assing numbers randomly with in bounds
        elif(tech == 1):
            # cycle through each variable in vector
            for i in range(len(vec)):
                num = random.random()
                # variable exceedes the minimum boundary
                if vec[i] < bounds[i][0]:
                    vec_new.append(num)
                # variable exceedes the maximum boundary
                if vec[i] > bounds[i][1]:
                    vec_new.append(num)
                # the variable is fine
                if bounds[i][0] <= vec[i] and vec[i] <= bounds[i][1]:
                    vec_new.append(vec[i])
        # assigning numbers with in bounds without adding more repetitions
        elif(tech == 2):
            #print("bounds range:",bounds[0][1]-bounds[0][0])
            test = [i for i in range(bounds[0][0], bounds[0][1]+1)]
            test = set(test)-set(vec)
            test = list(test)
            count = 0
            for i in range(len(vec)):
                if vec[i] < bounds[i][0]:
                    vec_new.append(test[count])
                    count = count+1
                if vec[i] > bounds[i][1]:
                    vec_new.append(test[count])
                    count = count+1
                if bounds[i][0] <= vec[i] and vec[i] <= bounds[i][1]:
                    vec_new.append(vec[i])
            # print("count:",count)
        return np.array(vec_new, dtype=np.float32)

    def local_search(self, population, iteration, maxiter):
        # 1 & 2
        #asc_pop=sorted(population,key=lambda cand:cost_func(decode_1(cand,rand_keys,agent_wid_rand_key_dict).tolist())[0],reverse=True)
        pop_size = len(population)
        pop_scores = [[C, C.cost] for C in population]
        pop_scores.sort(key=lambda x: x[1], reverse=True)
        asc_pop = [obj[0] for obj in pop_scores]

        # 3: crossover - TODO parallelize on GPU
        for i in range(pop_size-1):
            X_i, X_ip1 = asc_pop[i], asc_pop[i+1]
            r = random.random()
            X_i_new = X_i.vector + r * (X_ip1.vector - X_i.vector)

            # may need to modify
            c_new = self.cost_func(X_i_new)
            c_next = X_ip1.cost
            if c_new < c_next:
                X_i.vector = X_i_new
                X_i.cost = c_new
                asc_pop[i] = X_i

        # 4 (modify - clean)
        pop_scores = [[C, C.cost] for C in asc_pop]
        pop_scores.sort(key=lambda x: x[1], reverse=True)
        asc_pop = [obj[0] for obj in pop_scores]
        #asc_pop=sorted(asc_pop,key=lambda cand:cost_func(decode_1(cand,rand_keys,agent_wid_rand_key_dict).tolist())[0],reverse=True)
        gen_scores = []
        gen_peanlty = []

        # 5 Nonuniform mutation operation - TODO parallelize on GPU
        num_dim = len(self.bounds)
        cont_term = np.exp((-2*iteration)/(1.0*maxiter)) * \
            self.sigmoid((maxiter/2.0)-iteration)
        for i in range(pop_size):
            p_m_i = (pop_size-i+1)/(1.0*pop_size)
            X_i = asc_pop[i].vector
            X_i_new = np.zeros(X_i.shape, dtype=np.float32)
            for j in range(num_dim):
                b = random.randint(0, 1)
                r = random.random()
                if b == 0:
                    X_i_new[j] = X_i[j]+(self.bounds[j][1]-X_i[j])*r*cont_term
                else:
                    X_i_new[j] = X_i[j]+(X_i[j]-self.bounds[j][0])*r*cont_term
            # print ("X_i",X_i)
            c_new = self.cost_func(X_i_new)
            c_pres = asc_pop[i].cost
            if c_new < c_pres:
                # X_i=X_i_new
                gen_scores.append(c_new)
                new = Candidate(X_i_new)
                new.cost = c_new  # new.P=pen_new
                asc_pop[i] = new
                # asc_pop[i]=X_i_new
            else:
                gen_scores.append(c_pres)

        # wat to return
        return asc_pop, gen_scores

    # To generate mutation factor b/t the given limits
    @staticmethod
    def mutation_factor(min_=-1, max_=1):
        mutate = random.uniform(min_, max_)
        return mutate

    def evolve(self, population2, popsize, maxiter):
        # ---TO INITIALIZE A POPULATION from certain distribution(step #1) ----------------
        population = np.array(population2)

        # --- SOLVE --------------------------------------------+

        overgenerations_score = []
        gen_avg_list = []
        total_time = []
        num_dims = len(self.bounds)
        # F=mutate_f()
        cost_temp = 9999999999999999
        gen_sol = None
        # calculates cost for all the candidates in the pool
        for cand in population:
            cost_new = self.cost_func(cand.vector)
            cand.cost = cost_new
            #cand.P = pen_new
            if cost_new < cost_temp:
                cost_temp = cost_new
                gen_sol = cand

        print("best cost before DE:", cost_temp)
        #print("best sol before DE:",gen_sol)

        # cycle through each generation (step #2)
        # DE performed in every genertation
        for generation in range(1, maxiter+1):
            start = time.time()
            F = self.mutation_factor(hyperparams.F[0], hyperparams.F[1])
            # print("mutate:",F)
            print('GENERATION:', generation)
            # population_new=[]

            gen_scores = []  # score keeping

            # cycle through each individual in the population
            for j in range(0, popsize):
                if j % 3 == 0:
                    print("candidate:", j)

                candidates = [i for i in range(0, popsize)]
                candidates.remove(j)
                in_loop = 0

                # TODO to run on GPU
                while(True):
                    random_index = random.sample(candidates, 3)
                    # print("random_index:",random_index)
                    # print(population[random_index[0]])
                    x_1 = population[random_index[0]].vector
                    x_2 = population[random_index[1]].vector
                    x_3 = population[random_index[2]].vector
                    x_t = population[j].vector
                    x_b = gen_sol     # target individual

                    i_rand = [random.randint(0, num_dims-1)
                              for i in range(random.randint(1, 20))]
                    Cr = self.recombination
                    v_donor = copy.deepcopy(x_t)

                    # performing mutation operation as Zhang DE
                    # TODO mainly here
                    for i in i_rand:
                        if(random.random() < Cr):
                            v_donor[i] = x_3[i] + F*(x_1[i] - x_2[i])
                    break
                    #v_donor= mutation(x_t.tolist(),x_2.tolist(),x_3.tolist(),Cr,i_rand,F,cost_func)

                    # check,count_b=bounds_check(v_donor,self.bounds)
                    # if(check):
                    #    break
                    # in_loop=in_loop+1
                    # if(count_b or in_loop==100):
                    #    break
                if(True):  # (count_b or in_loop==100) and not check):
                    v_donor = self.ensure_bounds(v_donor, self.bounds)

                score_donor = self.cost_func(v_donor)
                # cost_func(decode_1(x_t,rand_keys,agent_wid_rand_key_dict).tolist())
                score_parent = population[j].cost
                if score_donor < score_parent:
                    population[j].vector = v_donor
                    population[j].cost = score_donor

                # --- RECOMBINATION (step #3.B) ----------------+
            # _________________________
            time_a = time.time()
            population, gen_scores = self.local_search(
                population, generation, maxiter)
            print("time:", time.time()-time_a)

            # --- SCORE KEEPING --------------------------------+

            # current generation avg. fitness
            gen_avg = sum(gen_scores) / popsize
            #print ("gen_scores", gen_scores)
            gen_best = min(gen_scores)
            # fitness of best individual
            best_sol_index = gen_scores.index(min(gen_scores))
            # solution of best individual
            gen_sol = population[best_sol_index].vector

            print('      > GENERATION AVERAGE:', gen_avg)
            print('      > GENERATION BEST:', gen_best)
            #print ('         > BEST SOLUTION:',gen_sol,'\n')
            #print ('      > BestSolution penality:',gen_pealties[best_sol_index])
            end = time.time()
            # total_time.append(end-start)
            print("time of gen:{} : {}".format(generation, end-start))
            #print("total time elapsed:", sum(total_time))
            overgenerations_score.append(gen_best)  # score keeping
            gen_avg_list.append(gen_avg)

            # loop breaking condidion
            if (gen_best < 0.01):
                break

        total_pop_scores = []
        for i in range(popsize):
            t_pop = population[i].vector
            total_pop_scores.append([t_pop, population[i].cost])

        total_pop_scores.sort(key=lambda x: x[1])
        total_pop = [obj[0] for obj in total_pop_scores]

        total_scores = np.array(overgenerations_score)
        gen_avg_array = np.array(gen_avg_list)

        # np.save('Results/totalscore',total_scores)
        #np.save('Results/gen_avg', gen_avg_array)

        return gen_sol, total_pop, gen_avg_array, total_scores
