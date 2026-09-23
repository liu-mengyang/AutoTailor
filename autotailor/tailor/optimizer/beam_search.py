import os
import sys
import copy
import random
import numpy as np
import json
import itertools
import GPyOpt

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)

__all__ = ["BeamSearch"]

class BeamSearch:
    """
    config_space is a flattened dict of the config space, with each key being a config key and each value being a list of possible values
    example: {"r": [0, 1, 2], "wid_0": [0, 1, 2], "wid_1": [0, 1, 2], "d_0": [0, 1, 2], "d_1": [0, 1, 2]}

    profile is a function that takes a config dict and returns a tuple of (accuracy, latency)
    example: config = {"r": 0, "wid_0": 0, "wid_1": 0, "d_0": 0, "d_1": 0}
    accuracy, latency = profile(config)

    latency_constraint is the maximum latency allowed

    optimize returns the best accuracy and latency found within the latency constraint
    """
    def __init__(self, config_space, profile, latency_constraint, n_samples=50, top_k=5, max_depth=7, max_iter=100):
        self.config_space = config_space
        self.profile = profile
        self.latency_constraint = latency_constraint
        self.n_samples = n_samples
        self.top_k = top_k
        self.max_depth = max_depth
        # self.max_iter = max_iter

        # Debug only, for better speed
        # self.config_space['r'] = [0]
        # self.config_space['wid_0'] = [2]
        # self.config_space['wid_1'] = [0]
        # self.config_space['wid_2'] = [0]
        # self.config_space['wid_3'] = [0]

    """
    get_partial_config_space returns the partial config space at a certain depth
    if the partial config space has only one item, it will continue to the next depth
    """
    def get_partial_config_space(self, depth=0):
        keys = list(self.config_space.keys())
        partial_config_space = {
            keys[depth]: self.config_space[keys[depth]]
        }
        partial_config_space_items = len(partial_config_space[keys[depth]])
        while depth < len(keys) - 1 and partial_config_space_items == 1:
            depth += 1
            partial_config_space[keys[depth]] = self.config_space[keys[depth]]
            partial_config_space_items = len(partial_config_space[keys[depth]])
        return depth + 1, partial_config_space
    
    """
    iter_partial_config_space returns the cartesian product of the partial config space
    """
    def iter_partial_config_space(self, partial_config_space):
        combos = list(itertools.product(*(partial_config_space[key] for key in partial_config_space)))
        return [{key: value for key, value in zip(partial_config_space, combo)} for combo in combos]

    """
    random_sample_from_partial_config returns a random sample from the partial config space
    """
    def random_sample_from_partial_config(self, partial_config):
        sample = {}
        for key, value in self.config_space.items():
            if key in partial_config:
                sample[key] = partial_config[key]
            else:
                sample[key] = random.choice(value)
        return sample
    
    """
    concat_partial_config_with_scores returns a list of partial configs with scores
    for each partial config, it will iterate through the partial config space and calculate the score
    the score is the maximum accuracy found within the latency constraint, sampled n_samples times
    """
    def concat_partial_config_with_scores(self, partial_config_with_scores, partial_config_space, n_samples=50):
        new_partial_config_with_scores = []
        for partial_config, _ in partial_config_with_scores:
            for new_partial_config in self.iter_partial_config_space(partial_config_space):
                score = 0
                for _ in range(n_samples):
                    sample = self.random_sample_from_partial_config({**partial_config, **new_partial_config})
                    accuracy, latency = self.profile(sample)
                    if latency < self.latency_constraint:
                        score = max(score, accuracy)
                print(f"[sampler] partial_config: {partial_config}, new_partial_config: {new_partial_config}, score: {score}")
                new_partial_config_with_scores.append(({**partial_config, **new_partial_config}, score))
        return new_partial_config_with_scores

    def optimize(self):
        partial_config_with_scores = [({}, 0)] # (partial_config, score)
        depth = 0
        while depth < min(self.max_depth, len(self.config_space)):
            depth, partial_config_space = self.get_partial_config_space(depth)
            print(f"[optimizer] depth: {depth}, partial_config_space: {partial_config_space}")
            partial_config_with_scores = self.concat_partial_config_with_scores(partial_config_with_scores, partial_config_space, self.n_samples)

            partial_config_with_scores = sorted(partial_config_with_scores, key=lambda x: x[1], reverse=True)[:self.top_k]
            partial_config_with_scores = [x for x in partial_config_with_scores if x[1] > 0]
            print(f"[optimizer] partial_config_with_scores: {partial_config_with_scores}")

        best_config, best_accuracy, best_latency = {}, 0, 0
        rest_config_space = {key: value for key, value in self.config_space.items() if key not in partial_config_with_scores[0][0]}
        print(f"[optimizer] rest_config_space: {rest_config_space}")
        
        for partial_config, _ in partial_config_with_scores:
            print(f"[optimizer] partial_config: {partial_config}")
            for _ in range(self.n_samples * 10):
                sample = self.random_sample_from_partial_config(partial_config)
                accuracy, latency = self.profile(sample)
                if latency < self.latency_constraint and accuracy > best_accuracy:
                    best_config, best_accuracy, best_latency = sample, accuracy, latency
            print(f"[optimizer] best_config: {best_config}, best_accuracy: {best_accuracy}, best_latency: {best_latency}")

        return best_config, best_accuracy, best_latency
        