import math
import json
import random
from itertools import product

import numpy as np
import pandas as pd

def load_design_space(space_file_path):
    space_dict = json.load(open(space_file_path))
    for kernel_type, cfg_dict in space_dict.items():
        if kernel_type == 'conv':
            hws = cfg_dict['HW']
            kernel_sizes = cfg_dict['KERNEL_SIZE']
            strides = cfg_dict['STRIDES']
            cins = cfg_dict['CINS']
            couts = cfg_dict['COUTS']
    
    first_step_features = [list(hws), list(kernel_sizes), list(strides)]
    second_step_features = [cins, couts]
    return first_step_features, second_step_features

def load_distribution_dataset(kernel_type, dataset_file_path):
    df = pd.read_csv(dataset_file_path)
    
    if kernel_type == 'conv':
        hws = df["input_h"]
        cins = df["cin"]
        couts = df["cout"]
        kernel_sizes = df["ks"]
        strides = df["stride"]

    first_step_features = [list(hws), list(kernel_sizes), list(strides)]
    unique_first_step_features = [list(pd.unique(hws)), list(pd.unique(kernel_sizes)), list(pd.unique(strides))]
    second_step_features = [cins, couts]
    
    return first_step_features, unique_first_step_features, second_step_features

def sample_in_range(mind, maxd, sample_num):
    if maxd - mind <= sample_num:
        data = list(range(mind, maxd+1))
        random.shuffle(data)
        return data
    else:
        return random.sample(range(mind, maxd), sample_num)

def get_flop_params(kernel_type, hw, cin, cout, kernel_size, stride):
    if kernel_type == 'conv':
        params = cout * (kernel_size * kernel_size * cin + 1)
        flops = 2 * hw / stride * hw / stride * params
    else:
        raise NotImplementedError
    return flops, params

def encoding(kernel_type, sampled_features):
    ncfgs = []
    nparams = [] # calculate the number of parameters for configs sort
    if kernel_type == 'conv':
        for hw, kernel_size, stride, cin, cout in sampled_features:
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
            nparams.append(get_flop_params(kernel_type, hw, cin, cout, kernel_size, stride))
    else:
        raise NotImplementedError
    # sort all sampling configs by number of parameters, from the smallest to the largest
    # the procedure is for better profiling
    ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
    
    kernels = {}
    for i, cfg in enumerate(ncfgs):
        random_id = f'{i:06d}'
        kernels[random_id] = {}
        kernels[random_id]['config'] = cfg
    return kernels

def random_sampling(kernel_type, sample_num, save_path, first_step_features, second_step_features):
    """Sampling by normal distribution

    Args:
        kernel_type (str): the type of kernel
        sample_num (int): the number of samples
        save_path (str): the path of saved samples
        first_step_features (list): dataset for fixed features
        second_step_features (list): dataset for random features
    """
    first_step_feature_combines = list(product(*first_step_features))
    per_feature_number = math.ceil(sample_num / len(first_step_feature_combines))
    print(f"#First step features: {len(first_step_feature_combines)}")
    print(f"#Samples of each combination: {per_feature_number}")
    features = []
    
    random.shuffle(first_step_feature_combines)
    for fs_feature in first_step_feature_combines:
        sampled_second_step_features = []
        for second_feature in second_step_features:
            sampled_features = sample_in_range(min(second_feature), max(second_feature), per_feature_number)
            sampled_second_step_features.append(sampled_features)
        
        for i in range(per_feature_number):
            feature = list(fs_feature)
            for second_feature in sampled_second_step_features:
                feature.append(second_feature[i])
            features.append(tuple(feature))
    sampled_cfgs = encoding(kernel_type, features)
    save_cfgs(sampled_cfgs, save_path)

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

def distribution_based_sampling(kernel_type, sample_num, save_path, first_step_features, second_step_features):
    """Sampling by prior distribution

    Args:
        kernel_type (str): the type of kernel
        sample_num (int): the number of samples
        save_path (str): the path of saved samples
        first_step_features (list): dataset for fixed features
        second_step_features (list): dataset for random features
    """
    features = []
    hws, kernel_sizes, strides = first_step_features
    
    sampled_second_features = []
    for feature in second_step_features:
        sampled_features = sample_based_on_distribution(feature, sample_num)
        sampled_second_features.append(sampled_features)
    
    clampped_sampled_hws = []
    clampped_sampled_kernel_sizes = []
    clampped_sampled_strides = []

    sampled_hws = sample_based_on_distribution(hws, sample_num)
    closest_index = [np.argmin((np.array(hws)-hw)**2) for hw in sampled_hws]
    for i,hw in enumerate(sampled_hws):
        clampped_sampled_hws.append(hws[closest_index[i]])

    sampled_kernel_sizes = sample_based_on_distribution(kernel_sizes, sample_num)
    closest_index = [np.argmin((np.array(kernel_sizes)-ks)**2) for ks in sampled_kernel_sizes]
    for i,ks in enumerate(sampled_kernel_sizes):
        clampped_sampled_kernel_sizes.append(kernel_sizes[closest_index[i]])

    sampled_strides = sample_based_on_distribution(strides, sample_num)
    closest_index = [np.argmin((np.array(strides)-stride)**2) for stride in sampled_strides]
    for i,stride in enumerate(sampled_strides):
        clampped_sampled_strides.append(strides[closest_index[i]])
    
    for i in range(sample_num):
        feature = []
        feature.append(int(clampped_sampled_hws[i]))
        feature.append(int(clampped_sampled_kernel_sizes[i]))
        feature.append(int(clampped_sampled_strides[i]))
        for second_feature in sampled_second_features:
            feature.append(second_feature[i])
        features.append(tuple(feature))
    
    sampled_cfgs = encoding(kernel_type, features)
    save_cfgs(sampled_cfgs, save_path)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_cfgs(cfgs, save_path):
    with open(save_path, 'w') as fp:
        json.dump(cfgs, fp, indent=4, cls=NpEncoder)

