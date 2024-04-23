import random
from Drones.droneV1 import AiDrone
import numpy as np


# def crossover(genome1: dict, genome2: dict) -> dict:
#     child_genome = {'weight': {}, 'bias': {}}
#     for chromosome in ['weight', 'bias']:
#         for genes in ['fc1', 'fc2', 'fc3']:
#             gene1 = genome1[chromosome][genes]
#             gene2 = genome2[chromosome][genes]
#
#             # random crossover splitting
#             ratio = np.random.rand()
#
#             # weighted sum
#             child_gene = gene1*ratio + gene2*(1-ratio)
#             child_genome[chromosome][genes] = child_gene
#
#     return child_genome


def crossover(genome1: dict, genome2:dict) -> dict:
    child_genome = {'weights': {}, 'bias': {}}
    for section in ['weights', 'bias']:
        for layer in range(len(genome1[section])):
            gene1: np.ndarray = genome1[section][layer]
            gene2: np.ndarray = genome2[section][layer]
            child_gene: np.ndarray = np.zeros_like(gene1)

            # random crossover splitting point
            splitpoint = int(np.random.rand()*gene1.shape[0])

            # split on 0th dimention of matrix
            if gene1.ndim > 1:
                child_gene[:splitpoint, :] = gene1[:splitpoint, :]  # upto splitpoint
                child_gene[splitpoint:, :] = gene2[splitpoint:, :]  # after splitpoint
            else:
                child_gene[:splitpoint] = gene1[:splitpoint]  # upto splitpoint
                child_gene[splitpoint:] = gene2[splitpoint:]  # after splitpoint

            # assign
            child_genome[section][layer] = child_gene

    return child_genome


# def crossover(genome1: dict, genome2:dict) -> dict:
#     child_genome = {'weight': {}, 'bias': {}}
#     for chromosome in ['weight', 'bias']:
#         for genes in ['fc1', 'fc2', 'fc3']:
#             gene1: np.ndarray = genome1[chromosome][genes]
#             gene2: np.ndarray = genome2[chromosome][genes]
#             child_gene: np.ndarray = np.zeros_like(gene1)
#
#             # random crossover split value
#             splitval = np.random.rand()
#             mask = np.random.rand(*child_gene.shape) < splitval
#
#             # if value < split point use gene 1, else gene two
#             child_gene[mask] = gene1[mask]
#             child_gene[~mask] = gene2[~mask]
#
#             # assign
#             child_genome[chromosome][genes] = child_gene
#
#     return child_genome


def mutate(genome: dict) -> dict:
    mutation_chance = 0.5

    for section in ['weights', 'bias']:
        for layer in range(len(genome[section])):
            gene: np.ndarray = genome[section][layer]
            # mutation_values = np.random.uniform(0.5, 1.5, size=gene.shape)
            # mutation_values = np.random.uniform(-0.1, 0.1, size=gene.shape)
            mutation_values = np.random.normal(scale=1, size=gene.shape)
            mask = np.random.rand(*gene.shape) < mutation_chance

            gene[mask] += mutation_values[mask]
            # gene[mask] += gene[mask] * mutation_values[mask]
            genome[section][layer] = gene

    return genome


def next_generation(drones: list[AiDrone], gen_size):
    drones: list[AiDrone] = sorted(drones, key=lambda x: x.score, reverse=True)  # sort from high to low score

    elite: int = len(drones) // 10
    next_gen: list[AiDrone] = drones[:elite]  # take 10% of the best prev gen

    # Scale Score by similarity factor to encourage generation diversity
    weights = np.array([drone.score/drone.similarity for drone in drones], dtype=np.float32)
    # weights = np.array([drone.score for drone in drones], dtype=np.float32)
    # Normalize weights
    weights -= weights.min()

    # create new ones for the next gen
    for i in range(gen_size-elite):
        # Select parents from random weighted choice
        try:
            parent1, parent2 = random.choices(drones, weights=weights, k=2)
        except ValueError as e:
            weights += 1
            parent1, parent2 = random.choices(drones, weights=weights, k=2)

        # crete new genome by crossing over
        crossed: dict = crossover(parent1.genome, parent2.genome)

        # mutate the genome
        child_genome: dict = mutate(crossed)

        # create and append the child
        child = AiDrone([20, 15], genome=child_genome)
        next_gen.append(child)

    for drone in next_gen:
        drone.reset()

    # error if gensize is not met
    assert len(next_gen) == gen_size
    return next_gen


if __name__ == '__main__':
    arr1 = np.array([[1, 2],
                     [3, 4]])
    print(arr1.ndim)
