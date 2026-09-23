import os
import sys
import copy
import random
import time
import numpy as np
import json
import itertools
from tqdm import tqdm

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)

__all__ = ["EvolutionFinder"]

class EvolutionFinder:
    """
    config_space is a flattened dict of the config space, with each key being a config key and each value being a list of possible values
    example: {"r": [0, 1, 2], "wid_0": [0, 1, 2], "wid_1": [0, 1, 2], "d_0": [0, 1, 2], "d_1": [0, 1, 2]}

    profile is a function that takes a config dict and returns a tuple of (accuracy, latency)
    example: config = {"r": 0, "wid_0": 0, "wid_1": 0, "d_0": 0, "d_1": 0}
    accuracy, latency = profile(config)

    latency_constraint is the maximum latency allowed

    optimize returns the best config, accuracy and latency found within the latency constraint
    """
    def __init__(self, config_space, profile, latency_constraint, mutate_prob=0.1, population_size=10, max_time_budget=120, parent_ratio=0.8, mutation_ratio=0.5):
        self.config_space = config_space
        self.profile = profile
        self.latency_constraint = latency_constraint
        self.mutate_prob = mutate_prob
        self.population_size = population_size
        self.max_time_budget = max_time_budget
        self.parent_ratio = parent_ratio
        self.mutation_ratio = mutation_ratio

        # self.config_space["r"] = [0]
        # self.config_space["wid_0"] = [2]
        # self.config_space["wid_1"] = [0]
        # self.config_space["wid_2"] = [0]
        # self.config_space["wid_3"] = [0]
        # self.config_space["wid_4"] = [0]

    def random_sample(self):
        while True:
            sample = {}
            for key, value in self.config_space.items():
                if value:
                    sample[key] = random.choice(value)
                else:
                    sample[key] = value
            accuracy, latency = self.profile(sample)
            if latency < self.latency_constraint:
                print("Random sample:", accuracy, latency)
                return sample, accuracy, latency
            
    def mutate(self, sample):
        while True:
            new_sample = copy.deepcopy(sample)
            for key, value in new_sample.items():
                if random.random() < self.mutate_prob:
                    if self.config_space[key]:
                        new_sample[key] = random.choice(self.config_space[key])
                    else:
                        new_sample[key] = self.config_space[key]
            accuracy, latency = self.profile(new_sample)
            if latency < self.latency_constraint:
                return new_sample, accuracy, latency
            
    def crossover(self, sample1, sample2):
        while True:
            new_sample = {}
            for key, value in sample1.items():
                new_sample[key] = value if random.random() < 0.5 else sample2[key]
            accuracy, latency = self.profile(new_sample)
            if latency < self.latency_constraint:
                return new_sample, accuracy, latency

    def optimize(self):
        parent_size = int(round(self.population_size * self.parent_ratio))
        mutation_size = int(round(self.population_size * self.mutation_ratio))

        # Initialization
        population = [self.random_sample() for _ in range(self.population_size)]
        # max by accuracy and min by latency
        best_sample = max(population, key=lambda x: (x[1], -x[2]))
        best_accuracy = best_sample[1]
        best_latency = best_sample[2]

        start_time = time.time()
        acc_list = []
        with tqdm(total=self.max_time_budget) as pbar:
            while (time.time() - start_time) < self.max_time_budget:
                pbar.update(time.time() - start_time - pbar.n)

                # Selection: Sort by accuracy and select the top to be parents
                parents = sorted(population, key=lambda x: (x[1], -x[2]), reverse=True)[:parent_size]

                # Reproduction: Crossover and mutation
                children = []

                # Mutation
                for _ in range(mutation_size):
                    children.append(self.mutate(random.choice(parents)[0]))

                # Crossover
                for _ in range(self.population_size - mutation_size):
                    parent1, parent2 = random.choice(parents)[0], random.choice(parents)[0]
                    children.append(self.crossover(parent1, parent2))

                # Replacement: Create new_population
                population = parents[:parent_size] + children
                population = sorted(population, key=lambda x: (x[1], -x[2]), reverse=True)

                # Update the best sample found so far
                current_best = population[0]
                if current_best[1] > best_accuracy and current_best[2] <= self.latency_constraint:
                    best_sample, best_accuracy, best_latency = current_best
                acc_list.append(best_accuracy)

        print("Best sample:", best_sample)
        print("Best accuracy:", best_accuracy)
        print("Best latency:", best_latency)
        return best_sample, best_accuracy, best_latency, acc_list
