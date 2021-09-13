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
    # tot = np.array(sentence).shape[0]
    tot = len(sentence)
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


# Get fitness and a map of unsat clauses and their counts
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
    # print(sat_percentage)
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


def get_best_neb(x, sentence):
    neighbours = []
    for i in range(len(x)):
        neighbours.append(flip_bit(x, i))
    fitness_arr = [calculate_fitness(assignment, sentence)
                   for assignment in neighbours]
    return neighbours[fitness_arr.index(max(fitness_arr))]


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


def GArejecc_with_select_mut(population, fitness_array, unsat_counts, sentence):
    print("Started modified GA with selective mutation and rejection for population size", len(
        population), "and sentence length", len(sentence))
    total_time = 0
    pass_number = 0
    start_time = time.time()
    n = len(population)
    best_assignment = []
    best_fitness = 0
    while(True):
        if pass_number % 100 == 0:
            print("Fitness value of the best model for generation",
                  pass_number, "is", max(fitness_array))
            # print(unsat_counts)
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
        # Actual Genetic Algorithm start
        new_sample = []
        while len(new_sample) != 2*n:
            parent1, parent2 = random.choices(
                population, k=2, weights=(np.array(fitness_array))**2)
            # print(len(parents))
            # p1_fitness = fitness_array[population.index(parent1)]
            # p2_fitness = fitness_array[population.index(parent2)]
            p1_fitness = calculate_fitness(parent1, sentence)
            p2_fitness = calculate_fitness(parent2, sentence)
            child = reproduce(np.array(parent1), np.array(parent2))
            child_fitness = calculate_fitness(child, sentence)
            if(child_fitness < p1_fitness or child_fitness < p2_fitness):
                continue
            child_mod1 = mod_mutate(child, (unsat_counts+0.1), delta=0.8)
            child_mod2 = mutate(child, delta=0.8)
            # new_sample.append(child)
            new_sample.append(child_mod1)
            new_sample.append(child_mod2)
        fitness_array_temp, unsat_counts_temp = fitness_mod(
            new_sample, sentence)
        fitness_array_temp_np = np.array(fitness_array_temp)
        inds = np.array(fitness_array_temp_np.argsort())
        inds = inds[::-1]
        # print(inds)
        population = list(np.array(new_sample)[inds[:n]])
        fitness_array = list(fitness_array_temp_np[inds[:n]])
        _, unsat_counts = fitness_mod(
            population, sentence)
        unsat_counts = unsat_counts_temp
        best_fitness = max(max(fitness_array), best_fitness)
        if best_fitness == max(fitness_array):
            best_assignment = population[fitness_array.index(best_fitness)]
        pass_number = pass_number + 1
    return total_time, best_fitness, best_assignment


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

    # fitness_array = [calculate_fitness(
    #     assignment, sentence) for assignment in population]
    fitness_array, unsat_counts = fitness_mod(
        population, sentence)

    # print(fitness_array)
    # print(len(fitness_array))
    times = []
    sat_percentage = []
    best_assignments = []
    runs = 5
    for i in range(runs):
        # t, f = genetic_algo(population, fitness_array, sentence, 0.4)
        # t, f, a = genetic_algo_with_rejecc(
        #     population, fitness_array, sentence)
        t, f, a = GArejecc_with_select_mut(
            population, fitness_array, unsat_counts, sentence)
        times.append(t)
        sat_percentage.append(f)
        best_assignments.append(a)

    print(list(times))
    print(np.mean(times))

    print(list(sat_percentage))
    print("Success rate of GA:", sat_percentage.count(1)*runs)
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
