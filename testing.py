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


def genetic_algo(population, fitness_array, sentence, delta=0.5):
    print("Started GA for population size", len(population),
          "and sentence length", len(sentence))
    total_time = 0
    pass_number = 0
    start_time = time.time()
    while(True):
        if pass_number % 1000 == 0:
            print("Fitness value of the best model for generation",
                  pass_number, "is", max(fitness_array))
        end_time = time.time()
        if(end_time - start_time > 45 or max(fitness_array) == 1):
            max_fitness = max(fitness_array)
            idx = fitness_array.index(max_fitness)
            total_time = end_time - start_time
            print('Best model : ', population[idx])
            print('Fitness value of best model : ', (max_fitness))
            print('Time taken : ', total_time, ' seconds')
            print("Generation Number: ", pass_number)
            print('\n\n')
            break
        new_population = []
        for i in range(len(population)):
            parents = random.choices(
                population, k=2, weights=np.array(fitness_array)**2)
            # print(len(parents))
            child = reproduce(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)
        population = new_population
        fitness_array = []
        for assignment in population:
            fitness = calculate_fitness(assignment, sentence)
            fitness_array.append(fitness)
        pass_number = pass_number + 1
    return total_time, max(fitness_array)


def genetic_algo_with_rejecc(population, fitness_array, sentence, delta=0.5):
    print("Started modified GA with rejection for population size",
          len(population), "and sentence length", len(sentence))
    total_time = 0
    pass_number = 0
    start_time = time.time()
    best_assignment = []
    best_fitness = 0
    while(True):
        # print("Best fitness: ", best_fitness)
        if pass_number % 100 == 0:
            print("Fitness value of the best model for generation",
                  pass_number, "is", max(fitness_array))
        end_time = time.time()
        if(end_time - start_time > 45 or max(fitness_array) == 1):
            total_time = end_time - start_time
            print('Best model : ', best_assignment)
            print('Fitness value of best model : ', (best_fitness))
            print('Time taken : ', total_time, ' seconds')
            print("Generation Number: ", pass_number)
            print('\n\n')
            break
        new_population = []
        while(len(new_population) != len(population)):
            parent1, parent2 = random.choices(
                population, k=2, weights=((np.array(fitness_array))**2))
            # print(len(parents))
            child = reproduce(parent1, parent2)
            child_fitness = calculate_fitness(child, sentence)
            p1_fitness = fitness_array[population.index(parent1)]
            p2_fitness = fitness_array[population.index(parent2)]
            if(child_fitness < p1_fitness or child_fitness < p2_fitness):
                continue
                # reject the weak child. Power same as parents is acceptable
            child = mutate(child, delta=delta)
            new_population.append(child)
        population = new_population
        fitness_array = [calculate_fitness(
            assignment, sentence) for assignment in population]
        best_fitness = max(max(fitness_array), best_fitness)
        if best_fitness == max(fitness_array):
            best_assignment = population[fitness_array.index(best_fitness)]
        pass_number = pass_number + 1
    return total_time, best_fitness, best_assignment


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
