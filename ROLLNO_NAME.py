from CNF_Creator import *
import numpy as np
import time
import random
# import math
# random.seed(0)


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


# Claculates fitness of a particular assignment
def calculate_fitness(assignment, sentence):
    tot = np.array(sentence).shape[0]
    sat = 0
    for clause in sentence:
        satisfy = False
        for var in clause:
            if (var > 0 and assignment[abs(var) - 1]) or (var < 0 and not assignment[abs(var) - 1]):
                satisfy = True
                break
        if satisfy:
            sat = sat + 1
    return (sat/tot)


def fitness_with_nonsat(assignment, sentence):
    num_vars = len(assignment)
    keys = np.arange(0, num_vars)
    vals = [0]*num_vars
    nonsat_vars_times = dict(zip(keys, val))
    tot = np.array(sentence).shape[0]
    sat = 0
    for clause in sentence:
        satisfy = False
        for var in clause:
            if (var > 0 and assignment[abs(var) - 1]) or (var < 0 and not assignment[abs(var) - 1]):
                satisfy = True
                break
        if satisfy:
            sat = sat + 1


def reproduce(x, y):
    # randint from 1 because we want some part of x in child
    c = random.randint(1, len(x)-1)
    child = x[:c] + y[c:]
    return child


def mutate(x):
    pos = random.randint(0, len(x)-1)
    x[pos] = not x[pos]
    return x


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
            if(random.random() < delta):
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
    while(True):
        if pass_number % 100 == 0:
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
        while(len(new_population) != len(population)):
            parent1, parent2 = random.choices(
                population, k=2, weights=np.array(fitness_array)**2)
            # print(len(parents))
            child = reproduce(parent1, parent2)
            child_fitness = calculate_fitness(child, sentence)
            p1_fitness = fitness_array[population.index(parent1)]
            p2_fitness = fitness_array[population.index(parent2)]
            if(child_fitness < p1_fitness or child_fitness < p2_fitness):
                continue
                # reject the weak child. Power same as parents is acceptable
            if(random.random() < delta):
                child = mutate(child)
            new_population.append(child)
        population = new_population
        fitness_array = []
        for assignment in population:
            fitness = calculate_fitness(assignment, sentence)
            fitness_array.append(fitness)
        pass_number = pass_number + 1
    return total_time, max(fitness_array)


def GArejecc_with_select_mut(population, fitness_array, sentence, delta=0.5):
    print("Started modified GA with selective mutation and rejection for population size", len(
        population), "and sentence length", len(sentence))
    total_time = 0
    pass_number = 0
    start_time = time.time()
    while(True):
        if pass_number % 100 == 0:
            print("Fitness value of the best model for generation",
                  pass_number, "is", max(fitness_array))
        end_time = time.time()
        # end loop code
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

        while(len(new_population) != len(population)):
            parent1, parent2 = random.choices(
                population, k=2, weights=np.array(fitness_array)**2)
            # print(len(parents))
            child = reproduce(parent1, parent2)
            child_fitness = calculate_fitness(child, sentence)
            p1_fitness = fitness_array[population.index(parent1)]
            p2_fitness = fitness_array[population.index(parent2)]
            if(child_fitness < p1_fitness or child_fitness < p2_fitness):
                continue
                # reject the weak child. Power same as parents is acceptable
            if(random.random() < delta):
                child = mutate(child)
            new_population.append(child)
        population = new_population
        fitness_array = []
        for assignment in population:
            fitness = calculate_fitness(assignment, sentence)
            fitness_array.append(fitness)
        pass_number = pass_number + 1
    return total_time, max(fitness_array)


def main():
    start_time = time.time()

    # -------------------------------START CODE HERE---------------------------
    # n is number of symbols in the 3-CNF sentence
    # m is number of clauses in the 3-CNF sentence
    cnfC = CNF_Creator(n=50)
    # sentence = cnfC.CreateRandomSentence(m=100)
    # print('Random sentence : ', sentence)

    sentence = cnfC.ReadCNFfromCSVfile()
    # print('\nSentence from CSV file : ', sentence)

    population = generate_population(m=50)
    # print(list(population))
    # print(np.array(population).shape)

    fitness_array = []
    for assignment in population:
        fitness = calculate_fitness(assignment, sentence)
        fitness_array.append(fitness)

    # print(fitness_array)
    # print(len(fitness_array))
    times = []
    sat_percentage = []
    for i in range(10):
        # t, f = genetic_algo(population, fitness_array, sentence, 0.4)
        t, f = genetic_algo_with_rejecc(
            population, fitness_array, sentence, 0.75)
        times.append(t)
        sat_percentage.append(f)
    print(list(times))
    print(np.mean(times))

    print(list(sat_percentage))
    print("Average fitness value of best GA model: ", np.mean(sat_percentage))
    # print(reproduce([1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2]))
    # -------------------------------END CODE HERE-----------------------------

#    print('\n\n')
#    print('Roll No : 2020H1030999G')
    # print('Number of clauses in sentense : ', len(sentence))
#    print('Best model : ',[1, -2, 3, -4, -5, -6, 7, 8, 9, 10, 11, 12, -13, -14, -15, -16, -17, 18, 19, -20, 21, -22, 23, -24, 25, 26, -27, -28, 29, -30, -31, 32, 33, 34, -35, 36, -37, 38, -39, -40, 41, 42, 43, -44, -45, -46, -47, -48, -49, -50])
#    print('Fitness value of best model : 99%')
#    print('Time taken : 5.23 seconds')
#    print('\n\n')

    end_time = time.time()
    print("Average time needed: ", (end_time - start_time)/10)


if __name__ == '__main__':
    main()
