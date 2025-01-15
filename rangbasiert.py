import random
import matplotlib.pyplot as plt
import numpy as np

POPULATION_SIZE = 10
GENOME_LENGTH = 8
MUTATION_RATE = 0.1
GENERATIONS = 50

def initialize_population():
    return [[random.randint(0, 1) for _ in range(GENOME_LENGTH)] for _ in range(POPULATION_SIZE)]

def fitness(individual):
    return sum(individual)

def assign_ranks(population, fitness_scores):
    sorted_population = sorted(population, key=lambda ind: fitness_scores[tuple(ind)])
    ranks = {tuple(ind): rank + 1 for rank, ind in enumerate(sorted_population)}
    return ranks

def select_parents(population, fitness_scores):
    ranks = assign_ranks(population, fitness_scores)
    total_rank = sum(ranks.values())
    parents = []
    for _ in range(2):
        pick = random.uniform(0, total_rank)
        current = 0
        for ind in population:
            current += ranks[tuple(ind)]
            if current >= pick:
                parents.append(ind)
                break
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
        fitness_scores = {tuple(ind): fitness(ind) for ind in population}

        best_individual = max(population, key=lambda ind: fitness_scores[tuple(ind)])
        best_fitness = fitness_scores[tuple(best_individual)]
        best_fitness_per_generation.append(best_fitness)

        if best_fitness == GENOME_LENGTH:
            break

        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parents = select_parents(population, fitness_scores)
            child1, child2 = crossover(parents[0], parents[1])
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population[:POPULATION_SIZE]

    return best_fitness_per_generation

results = [evolutionary_algorithm() for _ in range(100)]
max_generations = max(len(run) for run in results)
padded_results = [run + [run[-1]] * (max_generations - len(run)) for run in results]

data = np.array(padded_results)
average_fitness = np.mean(data, axis=0)
std_dev_fitness = np.std(data, axis=0)

plt.plot(range(len(average_fitness)), average_fitness, label="Durchschnittliche Fitness")
plt.fill_between(range(len(average_fitness)), average_fitness - std_dev_fitness, average_fitness + std_dev_fitness, alpha=0.2, label="Standardabweichung")
plt.title("Konvergenz der Fitness (Rangbasierte Selektion, 100 Runs)")
plt.xlabel("Generationen")
plt.ylabel("Beste Fitness")
plt.legend()
plt.grid(True)
plt.show()
