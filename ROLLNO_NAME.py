from CNF_Creator import *
import numpy as np
import time
import random


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


def print_assignment(x):
    assignment = []
    start = 1
    for item in x:
        if item:
            assignment.append(start)
        else:
            assignment.append(-1*start)
        start = start+1
    return list(assignment)


def GArejecc_with_select_mut(population, fitness_array, unsat_counts, sentence, delta=0.8):
    # print("Started modified GA with selective mutation and rejection for population size", len(
    #     population), "and sentence length", len(sentence))
    total_time = 0
    pass_number = 0
    n = len(population)
    best_fitness = max(fitness_array)
    best_assignment = population[fitness_array.index(best_fitness)]
    start_time = time.time()
    while(True):
        # if pass_number % 100 == 0:
        #     print("Fitness value of the best model for generation",
        #           pass_number, "is", max(fitness_array))
        end_time = time.time()
        if(end_time - start_time > 45 or max(fitness_array) == 1):
            total_time = end_time - start_time
            # print('Best model : ', best_assignment)
            # print('Fitness value of best model : ', best_fitness)
            # print('Time taken : ', total_time, ' seconds')
            # print("Generation Number: ", pass_number)
            # print('\n\n')
            break
        # Actual Genetic Algorithm start
        new_sample = []
        while len(new_sample) != 2*n:
            parent1, parent2 = random.choices(
                population, k=2, weights=(np.array(fitness_array))**2)
            p1_fitness = calculate_fitness(parent1, sentence)
            p2_fitness = calculate_fitness(parent2, sentence)
            child = list(reproduce(np.array(parent1), np.array(parent2)))
            child_fitness = calculate_fitness(child, sentence)
            if(child_fitness < p1_fitness or child_fitness < p2_fitness):
                continue
            child_mod1 = mod_mutate(child, (unsat_counts+0.1), delta=delta)
            child_mod2 = mutate(child, delta=delta)
            new_sample.append(child_mod1)
            new_sample.append(child_mod2)
        fitness_array_temp, unsat_counts_temp = fitness_mod(
            new_sample, sentence)
        fitness_array_temp_np = np.array(fitness_array_temp)
        inds = np.array(fitness_array_temp_np.argsort())
        inds = inds[::-1]
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
    cnfC = CNF_Creator(n=50)
    sentence = cnfC.ReadCNFfromCSVfile()
    population = generate_population(m=50)
    fitness_array, unsat_counts = fitness_mod(population, sentence)
    t, f, a = GArejecc_with_select_mut(
            population, fitness_array, unsat_counts, sentence)
    model = print_assignment(a)
    print('\n\n')
    print('Roll No : 2018A7PS0116G')
    print('Number of clauses in CSV file : ', len(sentence))
    print('Best model : ', model)
    print('Fitness value of best model : {}%'.format(100*f))
    print('Time taken :', t, 'seconds')
    print('\n\n')

    #Entire section is commented. Uncomment to run the code to generate values to plot graphs.
    # -------------------------------START CODE HERE---------------------------
    # # n is number of symbols in the 3-CNF sentence
    # # m is number of clauses in the 3-CNF sentence
    # time_per_run = []
    # average_fitness = []
    # success_rate = []
    # for sentence_len in range(20, 301, 20):
    #     times = []
    #     sat_percentage = []
    #     best_assignments = []
    #     runs = 20
    #     start_time = time.time()
    #     for i in range(runs):
    #         cnfC = CNF_Creator(n=50)
    #         # sentence = cnfC.ReadCNFfromCSVfile()
    #         sentence = cnfC.CreateRandomSentence(m=sentence_len)
    #         population = generate_population(m=50)
    #         fitness_array, unsat_counts = fitness_mod(population, sentence)
    #         # t, f = genetic_algo(population, fitness_array, sentence, 0.4)
    #         # t, f, a = genetic_algo_with_rejecc(
    #         #     population, fitness_array, sentence, delta=0.75)
    #         t, f, a = GArejecc_with_select_mut(
    #             population, fitness_array, unsat_counts, sentence)
    #         times.append(t)
    #         sat_percentage.append(f)
    #         best_assignments.append(a)
    #
    #     end_time = time.time()
    #     time_per_run.append((end_time - start_time)/runs)
    #     average_fitness.append(sum(sat_percentage)/len(sat_percentage))
    #     success_rate.append(sat_percentage.count(1)/runs)
    #     print("Sentence len completed:", sentence_len)
    #     # print("Success rate of GA:", sat_percentage.count(1)/runs)
    #     # print("Average fitness value:", sum(sat_percentage)/len(sat_percentage))
    #     # print("Average time needed for each run: ", (end_time - start_time)/runs)
    #
    # print(time_per_run)
    # print(average_fitness)
    # print(success_rate)
    # -------------------------------END CODE HERE-------------------------------


if __name__ == '__main__':
    main()
