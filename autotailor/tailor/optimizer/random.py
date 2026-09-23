import os
import sys
import copy
import random
import numpy as np
import json
import time
from tqdm import tqdm

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)

__all__ = ["Random"]

class Random:
    """
    config_space is a flattened dict of the config space, with each key being a config key and each value being a list of possible values

    profile is a function that takes a config dict and returns a tuple of (accuracy, latency)

    latency_constraint is the maximum latency allowed
    """
    def __init__(self, config_space, profile, latency_constraint, population_size=100, max_time_budget=120):
        self.config_space = config_space
        self.profile = profile
        self.latency_constraint = latency_constraint
        self.population_size = population_size
        self.max_time_budget = max_time_budget

    def random_sample(self):
        sample = {}
        for k, v in self.config_space.items():
            if v:
                sample[k] = random.choice(v)
            else:
                sample[k] = []
        return sample

    """
    optimize returns the best accuracy and latency found within the latency constraint
    """
    def optimize(self):
        sample = self.random_sample()
        
        best_accuracy = None
        while True:
            accuracy, latency = self.profile(sample)
            if latency <= self.latency_constraint:
                best_accuracy = accuracy
                break
        
        start_time = time.time()
        acc_list = []
        with tqdm(total=self.max_time_budget) as pbar:
            while (time.time() - start_time) < self.max_time_budget:
                pbar.update(time.time() - start_time - pbar.n)

                for i in range(self.population_size):
                    while True:
                        sample = self.random_sample()
                        accuracy, latency = self.profile(sample)
                        if latency <= self.latency_constraint:
                            break
                    if accuracy > best_accuracy:
                        best_sample = sample
                        best_latency = latency
                        best_accuracy = best_accuracy
                acc_list.append(best_accuracy)
                
        print("Best sample:", best_sample)
        print("Best accuracy:", best_accuracy)
        print("Best latency:", best_latency)
        return best_sample, best_accuracy, best_latency, acc_list