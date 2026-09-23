import time
import json
import os

from loguru import logger


def timestamp(name, stage):
    logger.info('TIMESTAMP, %s, %s, %f' % (name, stage, time.time()))
    
    
def save_profiled_results(models, save_path, detail, metrics = ["latency"]):
    new_models = merge_info(new_info=models, info_save_path=save_path)
    with open(save_path, 'w') as fp:
        json.dump(dump_profiled_results(new_models, detail=detail, metrics=metrics), fp, indent=4)


def dump_profiled_results(results, detail = False, metrics = ["latency"]):
    ''' convert Latency instance to string and return profiled results

    @params

    detail: if False, only metrics result will be dumped to the profiled results. Otherwise models information
        will be dumpled, too.
    '''
    dumped_results = {}
    for module_key, module in results.items():
        dumped_results[module_key] = {}
        if detail:
            for info_key, info in module.items():
                if info_key == 'latency':
                    dumped_results[module_key]['latency'] = str(module['latency'])
                else:
                    dumped_results[module_key][info_key] = info
        else:
            for info_key, info in module.items():
                if info_key in metrics:
                    if info_key == 'latency':
                        dumped_results[module_key]['latency'] = str(module['latency'])
                    else:
                        dumped_results[module_key][info_key] = module[info_key]
    return dumped_results


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