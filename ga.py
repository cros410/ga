from pyeasyga.pyeasyga import GeneticAlgorithm
import random
import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

custom_data_home = 'D:\Christian-Data\Proyectos\Python\data'
mnist  = datasets.fetch_mldata('MNIST original', data_home=custom_data_home)
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

data = [('uno', 1)]
ga = GeneticAlgorithm(data,
                        population_size=3,
                        generations=3,
                        crossover_probability=0.8,
                        mutation_probability=0.02,
                        elitism=True,
                        maximise_fitness=True)

#CREAR UN ELEMENTO PARA LA EVALUACION
def create_individual(data):
    r_capas = random.randint(1, 5)
    r_nodos = random.randint(1, 50)
    f = []
    for i in range(r_capas):
        f.append(r_nodos)
    print("***CREAR INDIVIDUO***")
    print(f)
    return f
ga.create_individual = create_individual


def crossover(p1, p2):
    
    l1 = [len(p1), len(p2)]
    l2 = [p1[0] , p2[0]]
    capas = reduce(lambda x, y: x + y, l1) / len(l1) 
    nodos = reduce(lambda x, y: x + y, l2) / len(l2)
    hijo = []
    for i in range(capas):
        hijo.append(nodos)
    print("***HERENCIA***")
    print(hijo)
    return hijo,hijo
ga.crossover_function = crossover

def mutate(individual):
    print("***MUTACION***")
    print(individual)
ga.mutate_function = mutate

def selection(population):
    v = random.choice(population)
    print("***SELECTION***")
    print(v)
    return v
ga.selection_function = selection

def fitness (individual, data):
    mlp = MLPClassifier(hidden_layer_sizes=(tuple(individual)), max_iter=5, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
    print("***FITNESS***")
    mlp.fit(X_train, y_train)
    print(mlp.score(X_train, y_train))
    return (1-mlp.score(X_train, y_train))
ga.fitness_function = fitness

ga.run()

print ga.best_individual()

for individual in ga.last_generation():
    print individual

