from CNF_Creator import *
import numpy as np
# import time
import random
# import math
# Generates a random assignment


def generate_assignment(n=50):
    assignment = []
    for i in range(n):
        num = random.random()
        assignment.append(True if num >= 0.5 else False)
    return assignment


# Generates the population for GA
def generate_population(m=20, n=50):
    population = []
    for i in range(m):
        population.append(generate_assignment(n))
    return population


def fitness_mod(population, sentence):
    sat_percentage = []
    unsat_count = []
    for assignment in population:
        tot = len(sentence)
        sat = 0
        unsat_list = [0]*len(assignment)
        for clause in sentence:
            satisfy = False
            for var in clause:
                if(var > 0 and assignment[abs(var) - 1]) or (var < 0 and not assignment[abs(var) - 1]):
                    satisfy = True
                    break
            if(satisfy):
                sat = sat+1
            else:
                for var in clause:
                    unsat_list[abs(var)-1] = unsat_list[abs(var)-1]+1
        # print(len(unsat_list))
        sat_percentage.append(sat/tot)
        unsat_count.append(unsat_list)
    return sat_percentage, np.mean(unsat_count, axis=0)


def reproduce(x, y):
    # randint from 1 because we want some part of x in child
    c = random.randint(1, len(x)-1)
    ret = []
    for i in x[:c]:
        ret.append(i)
    for i in y[c:]:
        ret.append(i)
    return ret


def mod_mutate(x, unsat_counts):
    pos = random.choices(list(np.arange(0, len(x))),
                         k=1, weights=unsat_counts**2)
    ret = []
    for gg in x[:pos[0]]:
        ret.append(gg)
    ret.append(not x[pos[0]])
    for gg in x[pos[0]+1:]:
        ret.append(gg)
    return ret


def main():
    cnfC = CNF_Creator(n=50)
    sentence = cnfC.ReadCNFfromCSVfile()
    population = generate_population(m=10)
    fitness_array, unsat_arr = fitness_mod(population, sentence)
    print("Shape of Sentence: ", np.array(sentence).shape)
    # print(fitness_array)
    # print(unsat_arr)
    # gg = mod_mutate(population[0], unsat_arr)
    # gg = reproduce([0, 0, 0, 0], [1, 1, 1, 1])
    # Actual Genetic Algorithm start
    new_sample = []
    n = len(population)
    for i in range(2*len(population)):
        parent1, parent2 = random.choices(
            population, k=2, weights=(np.array(fitness_array))**2)
        # print(len(parents))
        child = reproduce(np.array(parent1), np.array(parent2))
        child_mod = mod_mutate(child, unsat_arr)
        new_sample.append(child)
        new_sample.append(child_mod)
    fitness_array_temp, unsat_counts_temp = fitness_mod(
        new_sample, sentence)
    fitness_array_temp_np = np.array(fitness_array_temp)
    print(fitness_array_temp_np)
    inds = np.array(fitness_array_temp_np.argsort())
    print(inds)
    inds = inds[::-1]
    print(inds)
    population = np.array(new_sample)[inds[:n]]
    fitness_array = fitness_array_temp_np[inds[:n]]
    unsat_counts = unsat_counts_temp[inds[:n]]
    # print(gg)
    # print(population[0])

    # print(np.mean(fitness_array[:][1]))
    # sat = fitness_array[:][0]
    # print(sat)


if __name__ == "__main__":
    main()
