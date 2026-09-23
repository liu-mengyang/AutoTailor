import numpy as np
import random
import os
import copy
import math
import json
from sklearn.metrics import mean_squared_error


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def data_validation(data, cdata):
    ''' convert sampled data to valid configuration, e.g.,: kernel size = 1, 3, 5, 7

    @params:
    data: the origin data value.
    cdata: valid configuration value.
    '''
    newlist = []
    for da in cdata:
        value = [abs(da - x) for x in data]
        newlist.append(value)

    newlist = list(np.asarray(newlist).T)    
    cda = [list(d).index(min(d)) for d in newlist]
    redata = [cdata[x] for x in cda]
    return redata


def sample_based_on_distribution(data, sample_num, sample_bins = 40):
    ''' calculate inversed cdf, for sampling by possibility
    '''
    import scipy.interpolate as interpolate
    hist, bin_edges = np.histogram(data, bins=sample_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(sample_num)
    data = inv_cdf(r)
    new_data = [int(x) for x in data]
    return new_data


def sample_in_range(mind, maxd, sample_num):
    '''sample #sample_num data from a range [mind, maxd)
    '''
    # if the sample_num is bigger than sample population, we only keep the number of population to avoid repetition
    if maxd - mind <= sample_num:
        data = list(range(mind, maxd))
        random.shuffle(data)
        return data
    else:
        return random.sample(range(mind, maxd), sample_num)

def dump_profiled_results(results, detail = False, metrics = ["latency"]):
    ''' convert Latency instance to string and return profiled results

    @params

    detail: if False, only metrics result will be dumped to the profiled results. Otherwise models information
        will be dumpled, too.
    '''
    dumped_results = {}
    for module_key, module in results.items():
        dumped_results[module_key] = {}
        for model_key, model in module.items():
            dumped_results[module_key][model_key] = {}
            if detail:
                for info_key, info in model.items():
                    if info_key == 'latency':
                        dumped_results[module_key][model_key]['latency'] = str(model['latency'])
                    else:
                        dumped_results[module_key][model_key][info_key] = info
            else:
                for info_key, info in model.items():
                    if info_key in metrics:
                        if info_key == 'latency':
                            dumped_results[module_key][model_key]['latency'] = str(model['latency'])
                        else:
                            dumped_results[module_key][model_key][info_key] = model[info_key]
    return dumped_results

def read_profiled_results(results):
    results_copy = copy.deepcopy(results)
    for item in results_copy.values():
        if isinstance(item, dict):
            for model in item.values():
                if 'latency' in model:
                    model['latency'] = Latency(model['latency'])
    return results_copy

def save_sample_data(samples_data, save_path):
    new_samples_data = merge_info(new_info=samples_data, info_save_path=save_path)
    with open(save_path, 'w') as fp:
        json.dump(new_samples_data, fp, indent=4)

def save_profiled_results(models, save_path, detail, metrics = ["latency"]):
    new_models = merge_info(new_info=models, info_save_path=save_path)
    with open(save_path, 'w') as fp:
        json.dump(dump_profiled_results(new_models, detail=detail, metrics=metrics), fp, indent=4)


def get_accuracy(y_pred, y_true, threshold = 0.01):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) <= threshold)
    return len(b[0]) / len(y_true)


def latency_metrics(y_pred, y_true):
    """
    evaluation metrics for prediction performance
    """
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    acc5 = get_accuracy(y_pred, y_true, threshold=0.05)
    acc10 = get_accuracy(y_pred, y_true, threshold=0.10)
    acc15 = get_accuracy(y_pred, y_true, threshold=0.15)
    return rmse, rmspe, rmse / np.mean(y_true), acc5, acc10, acc15

def merge_info(new_info, info_save_path = None, prev_info = None):
    ''' merge `new_info` with previous info and return the updated info. This method is used in two cases: 

    1. before save `new_info` to `info_save_path`, we need to check if the `info_save_path` is an existing file. If `info_save_path`
    exists, this method will help merge the previous info saved in `info_save_path` for a incrementally storage and avoid information
    loss. In this case, params `new_info` and `info_save_path` are needed.

    2. extend the dictionary of `prev_info` with `new_info`. In this case, params `new_info` and `prev_info` are needed.
    
    @params
    
    new_info (dict): new information
    
    info_save_path (str): the path to save the new info. We need to check if the path is empty and mantain the previous info
        in `info_save_path`.
    
    prev_info (dict): the previous information
    '''
    if (info_save_path == None and prev_info == None) or (info_save_path != None and prev_info != None):
        raise ValueError("One and only one params of `info_save_path` and `prev_info` is needed.")

    if info_save_path != None and os.path.isfile(info_save_path):
        with open(info_save_path, 'r') as fp:
            prev_info = json.load(fp)

    if prev_info == None:
        return new_info

    if isinstance(prev_info, str):
        with open(prev_info, 'r') as fp:
            prev_info = json.load(fp)
    if isinstance(new_info, str):
        with open(new_info, 'r') as fp:
            new_info = json.load(fp)

    for module_key in new_info.keys():
        if module_key in prev_info:
            prev_info[module_key].update(new_info[module_key])
        else:
            prev_info[module_key] = new_info[module_key]
    return prev_info


class Latency:
    def __init__(self, avg=0, std=0):
        if isinstance(avg, str):
            avg, std = avg.split('+-')
            self.avg = float(avg)
            self.std = float(std)
        elif isinstance(avg, Latency):
            self.avg, self.std = avg.avg, avg.std
        else:
            self.avg = avg
            self.std = std

    def __str__(self):
        return f'{self.avg} +- {self.std}'

    def __add__(self, rhs):
        if isinstance(rhs, Latency):
            return Latency(self.avg + rhs.avg, math.sqrt(self.std ** 2 + rhs.std ** 2))
        else:
            return Latency(self.avg + rhs, self.std)
    
    def __radd__(self, lhs):
        return self.__add__(lhs)

    def __mul__(self, rhs):
        return Latency(self.avg * rhs, self.std * rhs)

    def __rmul__(self, lhs):
        return self.__mul__(lhs)

    def __le__(self, rhs):
        return self.avg < rhs.avg
    
    def __gt__(self, rhs):
        return self.avg > rhs.avg

    def __neg__(self):
        return Latency(-self.avg, -self.std)

    def __sub__(self, rhs):
        return self + rhs.__neg__()