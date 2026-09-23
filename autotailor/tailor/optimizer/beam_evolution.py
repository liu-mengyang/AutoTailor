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

__all__ = ["BeamEvolution"]

class BeamEvolution:
    """
    config_space is a flattened dict of the config space, with each key being a config key and each value being a list of possible values
    example: {"r": [0, 1, 2], "wid_0": [0, 1, 2], "wid_1": [0, 1, 2], "d_0": [0, 1, 2], "d_1": [0, 1, 2]}

    profile is a function that takes a config dict and returns a tuple of (accuracy, latency)
    example: config = {"r": 0, "wid_0": 0, "wid_1": 0, "d_0": 0, "d_1": 0}
    accuracy, latency = profile(config)

    latency_constraint is the maximum latency allowed

    optimize returns the best accuracy and latency found within the latency constraint
    """
    def __init__(self, config_space, profile, latency_constraint, n_samples=10, top_k=3, max_depth=4, mutate_prob=0.1, population_size=10, max_time_budget=30, parent_ratio=0.8, mutation_ratio=0.5):
        
        self.config_space = config_space
        # modify for motivation
        # for k, v in self.config_space.items():
        #     if "reduce_depth" in k and len(self.config_space[k]) > 0:
        #         self.config_space[k] = [0]
        #     if "resolution" in k:
        #         self.config_space[k] = [128, 172, 216, 256]
        #     if "kernel_size" in k:
        #         self.config_space[k] = [5]
        #     if "TransformerBlock-v_scale" in k:
        #         self.config_space[k] = [4]
        #     if "TransformerBlock-expand_ratio" in k:
        #         self.config_space[k] = [4,5]
        #     if "BottleneckResidualBlock-expand_ratio" in k:
        #         self.config_space[k] = [6]
        # print(self.config_space)
        self.profile = profile
        self.latency_constraint = latency_constraint
        self.n_samples = n_samples
        self.top_k = top_k
        self.max_depth = max_depth
        self.mutate_prob = mutate_prob
        self.population_size = population_size
        self.max_time_budget = max_time_budget
        self.parent_ratio = parent_ratio
        self.mutation_ratio = mutation_ratio

    """
    get_partial_config_space returns the partial config space at a certain depth
    if the partial config space has only one item, it will continue to the next depth
    """
    def get_partial_config_space(self, depth=0):
        keys = list(self.config_space.keys())
        partial_config_space = {
            keys[depth]: self.config_space[keys[depth]] if len(self.config_space[keys[depth]]) > 0 else [None]
        }
        partial_config_space_items = len(partial_config_space[keys[depth]])
        
        while depth < len(keys) - 1 and partial_config_space_items <= 1:
            depth += 1
            partial_config_space[keys[depth]] = self.config_space[keys[depth]] if len(self.config_space[keys[depth]]) > 0 else [None]
            partial_config_space_items = len(partial_config_space[keys[depth]])
        return depth + 1, partial_config_space
    
    """
    iter_partial_config_space returns the cartesian product of the partial config space
    """
    def iter_partial_config_space(self, partial_config_space):
        combos = list(itertools.product(*(partial_config_space[key] for key in partial_config_space)))
        res = [{key: value for key, value in zip(partial_config_space, combo)} for combo in combos]
        return res

    """
    random_sample_from_partial_config returns a random sample from the partial config space
    """
    def random_sample_from_partial_config(self, partial_config):
        sample = {}
        for key, value in self.config_space.items():
            if key in partial_config:
                sample[key] = partial_config[key]
            else:
                if value:
                    sample[key] = random.choice(value)
                else:
                    sample[key] = value
        return sample
    
    def random_sample_within_latency(self, partial_config):
        while True:
            sample = self.random_sample_from_partial_config(partial_config)
            accuracy, latency = self.profile(sample)
            if latency is not None and latency < self.latency_constraint:
                return sample, accuracy, latency
    
    """
    concat_partial_config_with_scores returns a list of partial configs with scores
    for each partial config, it will iterate through the partial config space and calculate the score
    the score is the maximum accuracy found within the latency constraint, sampled n_samples times
    """
    def concat_partial_config_with_scores(self, partial_config_with_scores, partial_config_space, n_samples=50):
        flag = False
        new_partial_config_with_scores = []
        for partial_config, _ in partial_config_with_scores:
            for new_partial_config in self.iter_partial_config_space(partial_config_space):
                score = 0
                for _ in range(n_samples):
                    sample = self.random_sample_from_partial_config({**partial_config, **new_partial_config})
                    accuracy, latency = self.profile(sample)
                    if latency is not None and latency < self.latency_constraint:
                        score += 1
                        flag = True
                score /= n_samples
                print(f"[sampler] partial_config: {partial_config}, new_partial_config: {new_partial_config}, score: {score}")
                new_partial_config_with_scores.append(({**partial_config, **new_partial_config}, score))
        # return new_partial_config_with_scores
        # if the maximum score is still 0, then increase n_samples to 2 * n_samples
        if not flag:
            print(f"[sampler] Increasing n_samples to {2 * n_samples}...")
            return self.concat_partial_config_with_scores(partial_config_with_scores, partial_config_space, 2 * n_samples)
        
        return new_partial_config_with_scores

    
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
            if latency is not None and latency < self.latency_constraint:
                return new_sample, accuracy, latency
            
    def crossover(self, sample1, sample2):
        while True:
            new_sample = {}
            for key, value in sample1.items():
                new_sample[key] = value if random.random() < 0.5 else sample2[key]
            accuracy, latency = self.profile(new_sample)
            if latency is not None and latency < self.latency_constraint:
                return new_sample, accuracy, latency

    def optimize(self):
        # 1. Beam search, find the partial config space with maximum probability
        partial_config_with_scores = [({}, 0)] # (partial_config, score)
        depth = 0
        while depth < min(self.max_depth, len(self.config_space)) and partial_config_with_scores[0][1] < 0.8:
            depth, partial_config_space = self.get_partial_config_space(depth)
            print(f"[optimizer] depth: {depth}, partial_config_space: {partial_config_space}")
            partial_config_with_scores = self.concat_partial_config_with_scores(partial_config_with_scores, partial_config_space, self.n_samples)

            partial_config_with_scores = sorted(partial_config_with_scores, key=lambda x: x[1], reverse=True)[:self.top_k]
            partial_config_with_scores = [x for x in partial_config_with_scores if x[1] > 0]
            print(f"[optimizer] partial_config_with_scores: {partial_config_with_scores}")
        partial_config_space = partial_config_with_scores[0][0]

        # 2. Evolutionary search, find the best config within the partial config space
        parent_size = int(round(self.population_size * self.parent_ratio))
        mutation_size = int(round(self.population_size * self.mutation_ratio))

        # Initialization
        population = [self.random_sample_within_latency(partial_config_space) for _ in range(self.population_size)]
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
                # print("[evolution] parents:", [(x[0]["r"], x[1], x[2]) for x in parents])

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
                if current_best[1] >= best_accuracy and current_best[2] <= self.latency_constraint:
                    best_sample, best_accuracy, best_latency = current_best
                acc_list.append(best_accuracy)

        print("Best sample:", best_sample)
        print("Best accuracy:", best_accuracy)
        print("Best latency:", best_latency)
        return best_sample, best_accuracy, best_latency, acc_list

        