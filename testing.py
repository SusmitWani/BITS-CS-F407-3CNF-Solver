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


def flip_bit(x, pos):
    x[pos] = not x[pos]
    return x


def mutate(x, delta=0.5):
    if(random.random() >= delta):
        return x
    pos = random.randint(0, len(x)-1)
    return flip_bit(x, pos)


def mod_mutate(x, unsat_counts, delta=0.5):
    if(random.random() >= delta):
        return x
    pos = random.choices(list(np.arange(0, len(x))),
                         k=1, weights=(unsat_counts)**2)
    return flip_bit(x, pos[0])


def main():
    cnfC = CNF_Creator(n=50)
    sentence = cnfC.ReadCNFfromCSVfile()
    population = generate_population(m=10)
    fitness_array, unsat_arr = fitness_mod(population, sentence)
    print("Shape of Sentence: ", np.array(sentence).shape)
    # print(fitness_array)
    # print(unsat_arr)
    print(population[0])
    gg = mod_mutate(population[0], unsat_arr, 1)
    # gg = reproduce([0, 0, 0, 0], [1, 1, 1, 1])
    # gg = flip_bit([True, True, True, False], 3)
    print(gg)
    # print(gg)
    # print(population[0])

    # print(np.mean(fitness_array[:][1]))
    # sat = fitness_array[:][0]
    # print(sat)


if __name__ == "__main__":
    main()
