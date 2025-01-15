import random
from multiprocessing import Pool, get_context
import matplotlib.pyplot as plt
import numpy as np

POPULATION_SIZE = 100
GENOME_LENGTH = 50
MUTATION_RATE = 0.1
GENERATIONS = 200

def initialize_population():
    return [[random.randint(0, 1) for _ in range(GENOME_LENGTH)] for _ in range(POPULATION_SIZE)]

def fitness(individual):
    return sum(individual)

def evaluate_fitness_parallel(population):
    with get_context("spawn").Pool() as pool:
        fitness_scores = pool.map(fitness, population, chunksize=10)
    return fitness_scores

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

def evolutionary_algorithm():
    population = initialize_population()
    best_fitness_per_generation = []

    for generation in range(GENERATIONS):
        raw_fitness_scores = evaluate_fitness_parallel(population)
        fitness_scores = {tuple(population[i]): raw_fitness_scores[i] for i in range(len(population))}

        best_individual = max(population, key=lambda ind: fitness_scores[tuple(ind)])
        best_fitness = fitness_scores[tuple(best_individual)]
        best_fitness_per_generation.append(best_fitness)

        if best_fitness == GENOME_LENGTH:
            print(f"Optimal solution found in generation {generation}!")
            break

        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parents = select_parents(population, fitness_scores)
            child1, child2 = crossover(parents[0], parents[1])
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population[:POPULATION_SIZE]

    return best_fitness_per_generation

if __name__ == "__main__":
    try:
        all_runs = [evolutionary_algorithm() for _ in range(2)]
        max_generations = max(len(run) for run in all_runs)
        padded_results = [run + [run[-1]] * (max_generations - len(run)) for run in all_runs]

        average_fitness = np.mean(padded_results, axis=0)
        std_dev_fitness = np.std(padded_results, axis=0)

        generations = range(len(average_fitness))
        plt.plot(generations, average_fitness, label="Durchschnittliche Fitness")
        plt.fill_between(generations, average_fitness - std_dev_fitness, average_fitness + std_dev_fitness, alpha=0.2, label="Standardabweichung")
        plt.title("Konvergenzrate und Lösungsgüte (datenparallel, 2 Runs)")
        plt.xlabel("Generationen")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.show()

    except KeyboardInterrupt:
        print("Process interrupted manually.")
