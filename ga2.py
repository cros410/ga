from pyeasyga.pyeasyga import GeneticAlgorithm
import random

data = [('pear', 50), ('apple', 35), ('banana', 40)]
ga = GeneticAlgorithm(data, 20, 50, 0.8, 0.2, True, True)

def create_individual(data):
    print("***CREATE INDIVIDUAL***")
    ind = [random.randint(0, 1) for _ in xrange(len(data))]
    print(ind)
    return ind

ga.create_individual = create_individual


def crossover(parent_1, parent_2):
    crossover_index = random.randrange(1, len(parent_1))
    child_1 = parent_1[:crossover_index ] + parent_2[crossover_index :]
    child_2 = parent_2[:crossover_index ] + parent_1[crossover_index :]
    print("***HERENCIA***")
    print(child_1)
    print(child_2)
    return child_1, child_2

ga.crossover_function = crossover


def mutate(individual):
    mutate_index = random.randrange(len(individual))
    if individual[mutate_index] == 0:
        individual[mutate_index] == 1
    else:
        individual[mutate_index] == 0
    print("***MUTAR***")
    print(individual)

ga.mutate_function = mutate


def selection(population):
    r = random.choice(population)
    print("***SELECTION***")
    print(r)
    return r

ga.selection_function = selection

def fitness (individual, data):
    fitness = 0
    if individual.count(1) == 2:
        for (selected, (fruit, profit)) in zip(individual, data):
            if selected:
                fitness += profit
    print("***FITNESS***")
    print(fitness)
    return fitness

ga.fitness_function = fitness
ga.run()
print ga.best_individual()

for individual in ga.last_generation():
    print individual