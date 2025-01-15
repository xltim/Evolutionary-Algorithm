import numpy as np
import random
import matplotlib.pyplot as plt

# Parameter
POPULATION_SIZE = 100
GENOME_LENGTH = 50
MUTATION_RATE = 0.1
GENERATIONS = 200
NUM_ISLANDS = 3
MIGRATION_INTERVAL = 10
MIGRATION_RATE = 5

def initialize_population():
    return [[random.randint(0, 1) for _ in range(GENOME_LENGTH)] for _ in range(POPULATION_SIZE)]

def fitness(individual):
    return sum(individual)

def select_parents(population, fitness_scores):
    parents = []
    for _ in range(2):
        parents.append(random.choice(population))
    return parents

def crossover(parent1, parent2):
    point = random.randint(1, GENOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual

def migrate(populations):
    for i in range(len(populations)):
        source = populations[i]
        target = populations[(i + 1) % len(populations)]
        migrants = random.sample(source, MIGRATION_RATE)
        target.extend(migrants)
        populations[i] = [ind for ind in source if ind not in migrants]
    return populations

def evolutionary_algorithm_island():
    populations = [initialize_population() for _ in range(NUM_ISLANDS)]
    best_fitness_per_generation = [[] for _ in range(NUM_ISLANDS)]

    for generation in range(GENERATIONS):
        for i in range(NUM_ISLANDS):
            population = populations[i]
            fitness_scores = {tuple(ind): fitness(ind) for ind in population}

            best_individual = max(population, key=lambda ind: fitness_scores[tuple(ind)])
            best_fitness = fitness_scores[tuple(best_individual)]
            best_fitness_per_generation[i].append(best_fitness)

            new_population = []
            while len(new_population) < POPULATION_SIZE:
                parents = select_parents(population, fitness_scores)
                child1, child2 = crossover(parents[0], parents[1])
                new_population.append(mutate(child1))
                new_population.append(mutate(child2))

            populations[i] = new_population[:POPULATION_SIZE]

        if (generation + 1) % MIGRATION_INTERVAL == 0:
            populations = migrate(populations)

    # Durchschnittliche Fitness aller Inseln für jede Generation
    average_fitness = [np.mean([best_fitness_per_generation[i][g] for i in range(NUM_ISLANDS)]) for g in range(GENERATIONS)]
    return average_fitness

def run_multiple_times(algorithm, runs=100):
    results = [algorithm() for _ in range(runs)]
    max_generations = max(len(run) for run in results)
    padded_results = [run + [run[-1]] * (max_generations - len(run)) for run in results]
    return np.array(padded_results)

# Sammeln der Daten für den Inselalgorithmus
all_runs = run_multiple_times(evolutionary_algorithm_island, runs=100)
average_fitness = np.mean(all_runs, axis=0)
std_dev_fitness = np.std(all_runs, axis=0)

# Plot mit Mittelwert und Standardabweichung
generations = range(average_fitness.shape[0])
plt.plot(generations, average_fitness, label="Durchschnittliche Fitness")
plt.fill_between(generations, average_fitness - std_dev_fitness, average_fitness + std_dev_fitness, alpha=0.2, label="Standardabweichung")
plt.title("Konvergenzrate und Lösungsgüte (Inselalgorithmus, 100 Runs)")
plt.xlabel("Generationen")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.show()
