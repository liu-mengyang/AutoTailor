import os

import pandas as pd

class BaseSampler:
    def __init__(self):
        self.kernels = {}
        self.save_dir = os.getenv("WORKSPACE")

    def read_zoo(filename):
        pass

    def prior_sampling(self, sample_num):
        pass

    def finegrained_sampling(self, configs, sample_num):
        pass


class GEMMSampler(BaseSampler):
    def __init__(self):
        self.kernel_type = "gemm"
        self.kernels = {}
        self.save_dir = os.getenv("RESULTS")
    
    def read_zoo(self, filename = "gemm.csv"):
        conv_df = pd.read_csv(filename)
        inp_shape = conv_df['INSHAPE']
        hws = int(inp_shape[0,-1].split(',')[1])
        cins = conv_df["CIN"]
        couts = conv_df["COUT"]
        return hws, cins, couts
    
    def get_flop_params(self, config_dict):
        inp_shape = conv_df['INSHAPE']
        hw = int(inp_shape[0,-1].split(',')[1])
        cin = config_dict['CIN']
        cout = config_dict['COUT']
        params = cin * cout
        flops = 2 * hw * params
        return flops, params


class ConvSampler(BaseSampler):
    def __init__(self):
        self.kernel_type = "conv-bn-relu"
        self.kernels = {}
        self.save_dir = os.getenv("RESULTS")
    
    def read_zoo(self, filename = "conv.csv"):
        conv_df = pd.read_csv(filename)
        if 'HW' in conv_df:
            hws = conv_df['HW']
        else:
            # old csv file from nn-meter
            hws = conv_df["input_h"]
        cins = conv_df["CIN"]
        couts = conv_df["COUT"]
        ks = conv_df["KERNEL_SIZE"]
        strides = conv_df["STRIDE"]
        return hws, cins, couts, ks, strides
    
    def get_flop_params(self, config_dict):
        hw = config_dict['HW']
        kernel_size = config_dict['KERNEL_SIZE']
        stride = config_dict['STRIDES']
        cin = config_dict['CIN']
        cout = config_dict['COUT']
        params = cout * (kernel_size * kernel_size * cin + 1)
        flops = 2 * hw / stride * hw / stride * params
        return flops, params


class DwConvSampler(BaseSampler):

    def __init__(self):
        self.kernel_type = "dwconv-bn-relu"
        self.kernels = {}
        self.save_dir = os.getenv("RESULTS")

    def read_zoo(self, filename = "dwconv.csv"):
        dwconv_df = pd.read_csv(filename)
        if 'HW' in dwconv_df:
            hws = dwconv_df['HW']
        else:
            # old csv file from nn-meter
            hws = dwconv_df["input_h"]
        if 'CHANNEL_SIZE' in dwconv_df:
            cs = dwconv_df['CHANNEL_SIZE']
        else:
            # old csv file from nn-meter
            cs = dwconv_df["cin"]
        ks = dwconv_df["KERNEL_SIZE"]
        strides = dwconv_df["STRIDE"]
        return hws, cs, ks, strides

    def get_flop_params(self, config_dict):
        hw = config_dict['HW']
        kernel_size = config_dict['KERNEL_SIZE']
        stride = config_dict['STRIDES']
        cs = config_dict['CHANNEL_SIZE']
        params = cs * (kernel_size * kernel_size + 1)
        flops = 2 * hw / stride * hw / stride * params
        return flops, params