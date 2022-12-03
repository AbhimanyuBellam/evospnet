
import numpy as np


def initialization(bounds, popsize, technique=1):
    '''
    Parameters:
    bounds:np_array([[d1min,d1max],[d2min,d2max]...])
    popsize:population size
    technique: 1 - np.rand.unif,2-halton sequence,3-hammersley sequence
    '''
    count = popsize
    dim = len(bounds)
    # uniform random generator
    # bounds[:,0] -> all mins and viseversa
    if (technique == 1):
        pop_float = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(count, dim))
        population = pop_float.astype(int)

    elif (technique == 4):
        pop_float = (np.zeros((count, dim))).astype(int)
        if (dim % 2 == 0):
            dim_temp = int(dim/2)
        else:
            dim_temp = int(dim/2)+1

        pop_float_temp = np.random.uniform(
            bounds[1][0], bounds[1][1], size=(count, dim_temp))
        # print("evens:",pop_float_temp)
        for i in range(popsize):
            for j in range(dim):

                if j % 2 == 0:
                    pop_float[i][j] = pop_float_temp[i][int(j/2)]
                else:
                    pop_float[i][j] = bounds[0][1]-pop_float[i][j-1]
        population = pop_float.astype(int)
        # print(population)

    else:
        import chaospy.distributions.sampler.sequences as cdss
        # Halton seqence
        if (technique == 2):
            points_norm = cdss.halton.create_halton_samples(
                order=count, dim=dim)
            intervals = bounds[:, 1]-bounds[:, 0]
            pop_float = (points_norm.T)*(intervals) + bounds[:, 0]
            population = pop_float.astype(int)
        # hammersley sequence
        # technique==3
        else:
            points_norm = cdss.hammersley.create_hammersley_samples(
                order=count, dim=dim)
            intervals = bounds[:, 1]-bounds[:, 0]
            pop_float = (points_norm.T)*(intervals) + bounds[:, 0]
            population = pop_float
    return population


if __name__ == '__main__':
    #bounds = np.array([[1,60*79] for i in range(500)])
    bounds = np.array([[-1, 1] for i in range(10)])
    a = initialization(bounds, 10, technique=3)
    print("total pop:", a)
    # np.save("popu_hamme",a)
