import os
import random
import json
import math
import sys
from itertools import product
import string

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import data_validation, sample_based_on_distribution, sample_in_range, NpEncoder
from autotailor.tailor.eff_predictors.litepred.litepred_trainer import VariationalAutoencoder
from autotailor.tailor.eff_predictors.litepred.sampler.sampling_tree import *

__BUILTIN_KERNELS__ = {
    # builtin name: [kernel class name, kernel sampler class name]
    "conv": ["ConvBlock", "ConvSampler"],
    "conv-bn-relu": ["ConvBnRelu", "ConvSampler"],
    "conv-bn-relu6": ["ConvBnRelu6", "ConvSampler"],
    "conv-bn": ["ConvBn", "ConvSampler"],
    "conv-relu": ["ConvRelu", "ConvSampler"],
    "conv-relu6": ["ConvRelu6", "ConvSampler"],
    "conv-hswish": ["ConvHswish", "ConvSampler"],
    "conv-bn-hswish": ["ConvBnHswish", "ConvSampler"],
    "gemm": ["GEMMBlock", "GEMMSampler"],
    "matmul": ["MatMulBlock", "MatMulSampler"],
    "linearmatmul": ["LinearMatMulBlock", "LinearMatMulSampler"],
    # dwconv
    "dwconv": ["DwConvBlock", "DwConvSampler"],
    "dwconv-bn": ["DwConvBn", "DwConvSampler"],
    "dwconv-relu": ["DwConvRelu", "DwConvSampler"],
    "dwconv-relu6": ["DwConvRelu6", "DwConvSampler"],
    "dwconv-bn-relu": ["DwConvBnRelu", "DwConvSampler"],
    "dwconv-bn-relu6": ["DwConvBnRelu6", "DwConvSampler"],
    "dwconv-bn-hswish": ["DwConvBnHswish", "DwConvSampler"],

    # others
    # "maxpool": ["MaxPoolBlock", "PoolingSampler"],
    # "avgpool": ["AvgPoolBlock", "PoolingSampler"],
    # "fc": ["FCBlock", "FCSampler"],
    # "concat": ["ConcatBlock", "ConcatSampler"],
    # "split": ["SplitBlock", "CinEvenSampler"],
    # "channelshuffle": ["ChannelShuffle", "CinEvenSampler"],
    # "se": ["SEBlock", "CinEvenSampler"],
    # "global-avgpool": ["GlobalAvgPoolBlock", "GlobalAvgPoolSampler"],
    # "bnrelu": ["BnRelu", "HwCinSampler"],
    # "bn": ["BnBlock", "HwCinSampler"],
    # "hswish": ["HswishBlock", "HwCinSampler"],
    # "swish": ["SwishBlock", "HwCinSampler"],
    # "relu": ["ReluBlock", "HwCinSampler"],
    # "addrelu": ["AddRelu", "HwCinSampler"],
    # "add": ["AddBlock", "HwCinSampler"],
}

__MODULE__ = sys.modules[__name__]
class SamplerManager:
    def __init__(self):
        super().__init__()
        self.samplers = []
        
    def init_sampler(self,
                    kernel_type = 'conv-bn-relu', 
                    ):
        if kernel_type not in __BUILTIN_KERNELS__:
            raise ValueError(f"Unsupported kernel type: {kernel_type}. Please register the kernel first.")

        sample_name = __BUILTIN_KERNELS__[kernel_type][1]
        
        self.samplers.append(sample_name)
        
        return getattr(__MODULE__,sample_name)()


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

    def encoding(self, ncfgs):
        kernels = {}
        for i, cfg in enumerate(ncfgs):
            random_id = f'{i:06d}'
            kernels[random_id] = {}
            kernels[random_id]['config'] = cfg
        return kernels

    def save_cfgs(self, cfgs, save_path):
        with open(save_path, 'w') as fp:
            json.dump(cfgs, fp, indent=4, cls=NpEncoder)


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
        strides = conv_df["STRIDES"]
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
    
    def random_sampling(self, sample_num, prior_file_path):
        hws, cins, couts, kernel_sizes, strides = self.read_zoo(prior_file_path)
        
        sampling_ids = random.sample(range(len(hws)), k=sample_num)
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for id in sampling_ids:
            c = {
                'HW': hws[id],
                'CIN': cins[id],
                'COUT': couts[id],
                'KERNEL_SIZE': kernel_sizes[id],
                'STRIDES': strides[id],
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)
    
    def uniform_sampling(self, sample_num, prior_file_path):
        hws, cins, couts, kernel_sizes, strides = self.read_zoo(prior_file_path)
        hier_cfg_dict = {}
        # ks-stride-hw-cin-cout

        for i in range(len(hws)):
            ks = kernel_sizes[i]
            stride = strides[i]
            hw = hws[i]
            cin = cins[i]
            cout = couts[i]

            if ks not in hier_cfg_dict:
                hier_cfg_dict[ks] = {}
            if stride not in hier_cfg_dict[ks]:
                hier_cfg_dict[ks][stride] = {}
            if hw not in hier_cfg_dict[ks][stride]:
                hier_cfg_dict[ks][stride][hw] = {}
            if cin not in hier_cfg_dict[ks][stride][hw]:
                hier_cfg_dict[ks][stride][hw][cin] = []
            if cout not in hier_cfg_dict[ks][stride][hw][cin]:
                hier_cfg_dict[ks][stride][hw][cin].append(cout)
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        count = 0
        for ks in hier_cfg_dict:
            for stride in hier_cfg_dict[ks]:
                for hw in hier_cfg_dict[ks][stride]:
                    for cin in hier_cfg_dict[ks][stride][hw]:
                        for cout in hier_cfg_dict[ks][stride][hw][cin]:
                            count += 1
                            c = {
                                    'HW': hw,
                                    'CIN': cin,
                                    'COUT': cout,
                                    'KERNEL_SIZE': ks,
                                    'STRIDES': stride,
                                }
                            ncfgs.append(c)
        
        if count > sample_num:
            ncfgs = random.choices(ncfgs, k=sample_num)
            print('The number of design space is larger than sampling number.')
        
        for cfg in ncfgs:
            nparams.append(self.get_flop_params(cfg))
        
        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)
        
    def prior_sampling(self, sample_num, prior_file_path = "conv.csv"):
        ''' 
        Sampling configs for conv kernels based on conv_zoo, which contains configuration values from existing model zoo for conv kernel. 
        The values are stored in prior_config_lib/conv.csv.
        Returned params include: (hw, cin, cout, kernel_size, strides)
        '''
        hws, cins, couts, kernel_sizes, strides = self.read_zoo(prior_file_path)
        new_cins = sample_based_on_distribution(cins, sample_num)
        new_couts = sample_based_on_distribution(couts, sample_num)

        new_hws = sample_based_on_distribution(hws, sample_num)
        new_kernel_sizes = sample_based_on_distribution(kernel_sizes, sample_num)
        new_strides = sample_based_on_distribution(strides, sample_num)

        new_kernel_sizes = data_validation(new_kernel_sizes, list(set(kernel_sizes)))
        new_strides = data_validation(new_strides, list(set(strides)))
        new_hws = data_validation(new_hws, list(set(hws)))
    
        random.shuffle(new_hws)
        random.shuffle(new_strides)
        random.shuffle(new_kernel_sizes)

        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts, new_kernel_sizes, new_strides):
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)
    
    def vae_sampling(self, sample_num, prior_file_path, vae_path):
        latent_dims = 128
        encoder_hidden_dims = 256
        decoder_hidden_dims = 256
        source_dims = 5
        encoder_layers = 5
        decoder_layers = 5
        
        vae_model = VariationalAutoencoder(
            latent_dims=latent_dims,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            source_dims=source_dims,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )
        
        vae_model.load_state_dict(torch.load(vae_path, map_location='cpu'))
        vae_model = vae_model.cpu()
        decoder = vae_model.decoder
        
        def Gconv(model,z,scale,min_val):
            input_z = torch.from_numpy(z).to(torch.float32)
            output_x = model.forward(input_z).detach().numpy()
            output_x[output_x > 1] = 1
            output_x[output_x < 0] = 0
            return output_x * scale + min_val

        def Rconv(x):
            x0 = list(x.flatten())
            x0 = [np.exp2(x0[0]), np.exp2(x0[1]), np.exp2(x0[2]), x0[3], x0[4]]
            x0 = [round(val) for val in x0]
            return x0

        def reconstraint_convdata(x: pd.DataFrame):
            x.loc[x['KERNEL_SIZE'] == 1, 'KERNEL_SIZE'] = 1
            x.loc[x['KERNEL_SIZE'] == 2, 'KERNEL_SIZE'] = 1
            x.loc[x['KERNEL_SIZE'] == 3, 'KERNEL_SIZE'] = 3
            x.loc[x['KERNEL_SIZE'] == 4, 'KERNEL_SIZE'] = 3
            x.loc[x['KERNEL_SIZE'] == 6, 'KERNEL_SIZE'] = 5
            x.loc[x['KERNEL_SIZE'] > 7, 'KERNEL_SIZE'] = 7
            x.loc[x['CIN'] > 3, 'CIN'] = \
                np.ceil(x[x['CIN'] > 3]['CIN'] / 8).astype(np.int32) * 8
            x.loc[x['COUT'] > 3, 'COUT'] = \
                np.ceil(x[x['COUT'] > 3]['COUT'] / 8).astype(np.int32) * 8
            return x

        conv_features = ['HW', 'CIN', 'COUT', 'KERNEL_SIZE', 'STRIDES']
        conv_lut_configs = pd.read_csv(prior_file_path)
        
        conv_df = conv_lut_configs[conv_features].copy()
        # normalize
        conv_df['HW'] = np.log2(conv_df['HW'])
        conv_df['CIN'] = np.log2(conv_df['CIN'])
        conv_df['COUT'] = np.log2(conv_df['COUT'])
        conv_data = conv_df.to_numpy().astype(np.float32)
        conv_min_val = conv_data.min(axis=0)
        conv_max_val = conv_data.max(axis=0)
        # normalize
        conv_data = (conv_data - conv_min_val) / (conv_max_val - conv_min_val)
        conv_scale = conv_max_val - conv_min_val
        
        z = np.random.normal(size=(sample_num, latent_dims))
        x_hat_0 = [Rconv(Gconv(decoder, row,conv_scale,conv_min_val)) for row in z]
        df = pd.DataFrame(data=np.array(x_hat_0), columns=conv_features)
        df['SCALED_FREQ'] = 1 / sample_num
        
        df = reconstraint_convdata(df)
        df = df[conv_features]
        
        hws = df["HW"]
        cins = df["CIN"]
        couts = df["COUT"]
        kernel_sizes = df["KERNEL_SIZE"]
        strides = df["STRIDES"]
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cin, cout, kernel_size, stride in zip(hws, cins, couts, kernel_sizes, strides):
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)
    
    def hier_sampling(self, sample_num, space_path, enable_hw=False, enable_tiling=False):
        conv_model = ConvSamplingModel(space_path)
        conv_model.construct_tree()
        
        cases = []
        if not enable_hw:
            for ks, first_space_dict in conv_model.hier_cfg_tree.items():
                for stride, second_space_dict in first_space_dict.items():
                    num_samples = 0
                    for hw, last_space_dict in second_space_dict.items():
                        num_samples += last_space_dict['#POINT']
                    cases.append({'ks': ks, 'stride': stride, 'num_samples': num_samples})

            
        else:
            for ks, first_space_dict in conv_model.hier_cfg_tree.items():
                for stride, second_space_dict in first_space_dict.items():
                    for hw, last_space_dict in second_space_dict.items():
                        num_samples = last_space_dict['#POINT']
                        cases.append({'ks': ks, 'stride': stride, 'hw': hw, 'num_samples': num_samples})
        
        ratios = []
        for d in cases:
            ratio = d['num_samples'] / conv_model.num_samples
            ratios.append(ratio)
    
        # calculate sampling number for each case
        total_target_num_samples = 0
        for i, ratio in enumerate(ratios):
            cases[i]['target_num'] = math.floor(ratio * sample_num)
            total_target_num_samples += cases[i]['target_num']
        # pad samples
        rest_num = sample_num - total_target_num_samples
        rest_samples_id = random.choices(range(len(cases)), k=rest_num)
        for id in rest_samples_id:
            sampled_flag = False
            cur_id = id
            while(not sampled_flag):
                if cases[cur_id]['target_num'] < cases[cur_id]['num_samples']:
                    cases[cur_id]['target_num'] += 1
                    sampled_flag = True
                else:
                    if cur_id == len(cases)-1:
                        cur_id = 0
                    else:
                        cur_id += 1
        
        # sampling
        kss = []
        strides = []
        hws = []
        cins = []
        couts = []
        
        if not enable_hw:
            for case in cases:
                ks = case['ks']
                stride = case['stride']
                
                combs = []
                for hw, hw_dict in conv_model.hier_cfg_tree[ks][stride].items():
                    for cin, cout_lst in hw_dict['CINS'].items():
                        for cout in cout_lst:
                            combs.append((hw, cin, cout))
                samples = random.sample(combs, k=case['target_num'])
                for sample in samples:
                    hw = sample[0]
                    cin = sample[1]
                    cout = sample[2]
                    kss.append(ks)
                    strides.append(stride)
                    hws.append(hw)
                    cins.append(cin)
                    couts.append(cout)
        else:
            for case in cases:
                ks = case['ks']
                stride = case['stride']
                hw = case['hw']
                
                combs = []
                for cin, temp_couts in conv_model.hier_cfg_tree[ks][stride][hw]['CINS'].items():
                    for cout in temp_couts:
                        combs.append((cin, cout))
                samples = random.sample(combs, k=case['target_num'])
                for sample in samples:
                    cin = sample[0]
                    cout = sample[1]
                    kss.append(ks)
                    strides.append(stride)
                    hws.append(hw)
                    cins.append(cin)
                    couts.append(cout)
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cin, cout, kernel_size, stride in zip(hws, cins, couts, kss, strides):
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)
    
    def from_csv(self, csv_path):
        hws, cins, couts, kernel_sizes, strides = self.read_zoo(csv_path)
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cin, cout, kernel_size, stride in zip(hws, cins, couts, kernel_sizes, strides):
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)

    def nnmeter_sampling(self, sample_num, prior_file_path = "conv.csv"):
        '''
        Sampling configs for conv kernels based on conv_zoo, which contains configuration values from existing model zoo for conv kernel. 
        Besides, extending some configs.
        The values are stored in prior_config_lib/conv.csv.
        Returned params include: (hw, cin, cout, kernel_size, strides)
        '''
        hws, cins, couts, kernel_sizes, strides = self.read_zoo(prior_file_path)
        new_cins = sample_based_on_distribution(cins, sample_num)
        new_couts = sample_based_on_distribution(couts, sample_num)

        # 70% of sampled data are from prior distribution
        count1 = int(sample_num * 0.7)
        new_hws = sample_based_on_distribution(hws, count1)
        new_kernel_sizes = sample_based_on_distribution(kernel_sizes, count1)
        new_strides = sample_based_on_distribution(strides, count1)

        new_kernel_sizes = data_validation(new_kernel_sizes, list(set(kernel_sizes)))
        new_strides = data_validation(new_strides, list(set(strides)))
        new_hws = data_validation(new_hws, list(set(hws)))
    
        # since conv is the largest and most-challenging kernel, we add some frequently used configuration values
        new_hws.extend([112] * int((sample_num - count1) * 0.2) + [56] * int((sample_num - count1) * 0.4) + [28] * int((sample_num - count1) * 0.4)) # frequent settings
        new_kernel_sizes.extend([5] * int((sample_num - count1) * 0.4) + [7] * int((sample_num - count1) * 0.6)) # frequent settings
        new_strides.extend([2] * int((sample_num - count1) * 0.4) + [1] * int((sample_num - count1) * 0.6)) # frequent settings
        random.shuffle(new_hws)
        random.shuffle(new_strides)
        random.shuffle(new_kernel_sizes)

        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts, new_kernel_sizes, new_strides):
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)

    def finegrained_sampling(self, configs, sample_num):
        ''' 
        Sampling configs for conv kernels
        Returned params include: (hw, cin, cout, kernel_size, strides)
        '''
        ncfgs = []
        for cfg in configs:
            cin = cfg['CIN']
            cout = cfg['COUT']
            cins = sample_in_range(int(cin * 0.5), int(cin * 1.2), sample_num)
            couts = sample_in_range(int(cout * 0.5), int(cout * 1.2), sample_num)
            for cin, cout in zip(cins, couts):
                c = {
                    'HW': cfg['HW'],
                    'CIN': cin,
                    'COUT': cout,
                    'KERNEL_SIZE': cfg['KERNEL_SIZE'],
                    'STRIDES': cfg['STRIDES'],
                }
                ncfgs.append(c)
        return self.encoding(ncfgs)


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

    def random_sampling(self, sample_num, prior_file_path):
        hws, channel_sizes, kernel_sizes, strides = self.read_zoo(prior_file_path)
        
        sampling_ids = random.sample(range(len(hws)), k=sample_num)
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for id in sampling_ids:
            c = {
                'HW': hws[id],
                'CHANNEL_SIZE': channel_sizes[id],
                'KERNEL_SIZE': kernel_sizes[id],
                'STRIDES': strides[id],
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)

    def uniform_sampling(self, sample_num, prior_file_path):
        hws, css, kernel_sizes, strides = self.read_zoo(prior_file_path)
        hier_cfg_dict = {}
        # ks-stride-hw-cin-cout

        for i in range(len(hws)):
            ks = kernel_sizes[i]
            stride = strides[i]
            hw = hws[i]
            cs = css[i]

            if ks not in hier_cfg_dict:
                hier_cfg_dict[ks] = {}
            if stride not in hier_cfg_dict[ks]:
                hier_cfg_dict[ks][stride] = {}
            if hw not in hier_cfg_dict[ks][stride]:
                hier_cfg_dict[ks][stride][hw] = []
            if cs not in hier_cfg_dict[ks][stride][hw]:
                hier_cfg_dict[ks][stride][hw].append(cs)

        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        count = 0
        for ks in hier_cfg_dict:
            for stride in hier_cfg_dict[ks]:
                for hw in hier_cfg_dict[ks][stride]:
                    for cs in hier_cfg_dict[ks][stride][hw]:
                            count += 1
                            c = {
                                    'HW': hw,
                                    'CHANNEL_SIZE': cs,
                                    'KERNEL_SIZE': ks,
                                    'STRIDES': stride,
                                }
                            ncfgs.append(c)
                            
        if count > sample_num:
            ncfgs = random.choices(ncfgs, k=sample_num)
        
        for cfg in ncfgs:
            nparams.append(self.get_flop_params(cfg))
        
        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)

    def prior_sampling(self, sample_num, prior_file_path = "dwconv.csv"):
        ''' 
        Sampling configs for dwconv kernels based on dwconv zoo, which contains configuration values from existing model zoo for dwconv kernel. 
        The values are stored in prior_config_lib/dwconv.csv.
        Returned params include: (hw, cin, kernel_size, strides)
        '''
        hws, channel_sizes, kernel_sizes, strides = self.read_zoo(prior_file_path)
        new_channel_sizes = sample_based_on_distribution(channel_sizes, sample_num)
   
        new_hws = sample_based_on_distribution(hws,sample_num)
        new_kernel_sizes = sample_based_on_distribution(kernel_sizes, sample_num)
        new_strides = sample_based_on_distribution(strides, sample_num)
    
        new_kernel_sizes = data_validation(new_kernel_sizes, list(set(kernel_sizes)))
        new_strides = data_validation(new_strides, list(set(strides)))
        new_hws = data_validation(new_hws, list(set(hws)))
        
        random.shuffle(new_hws)
        random.shuffle(new_kernel_sizes)
        random.shuffle(new_strides)

        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cs, kernel_size, stride in zip(new_hws, new_channel_sizes, new_kernel_sizes, new_strides):
            c = {
                'HW': hw,
                'CHANNEL_SIZE': cs,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]

        return self.encoding(ncfgs)
    
    def vae_sampling(self, sample_num, prior_file_path, vae_path):
        latent_dims = 128
        encoder_hidden_dims = 256
        decoder_hidden_dims = 256
        source_dims = 4
        encoder_layers = 5
        decoder_layers = 5
        
        dwconv_vae_model = VariationalAutoencoder(
            latent_dims=latent_dims,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            source_dims=source_dims,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )
        
        dwconv_vae_model.load_state_dict(torch.load(vae_path, map_location='cpu'))
        
        dwconv_vae_model = dwconv_vae_model.cpu()
        dwconv_vae_model.eval()
        decoder = dwconv_vae_model.decoder
        
        def Gdwconv(model,z,scale,min_val):
            input_z = torch.from_numpy(z).to(torch.float32)
            output_x = model.forward(input_z).detach().numpy()
            output_x[output_x > 1] = 1
            output_x[output_x < 0] = 0
            return output_x * scale + min_val

        def Rdwconv(x):
            x0 = list(x.flatten())
            x0 = [np.exp2(x0[0]), np.exp2(x0[1]), x0[2], x0[3]]
            x0 = [round(val) for val in x0]
            return x0
        
        def reconstraint_dwconvdata(x: pd.DataFrame):
            x.loc[x['KERNEL_SIZE'] <= 3, 'KERNEL_SIZE'] = 3
            x.loc[x['KERNEL_SIZE'] == 4, 'KERNEL_SIZE'] = 3
            x.loc[x['KERNEL_SIZE'] == 6, 'KERNEL_SIZE'] = 5
            x.loc[x['KERNEL_SIZE'] > 7, 'KERNEL_SIZE'] = 7
            x.loc[x['CHANNELS'] > 3, 'CHANNELS'] = \
                np.ceil(x[x['CHANNELS'] > 3]['CHANNELS'] / 8).astype(np.int32) * 8
            return x

        dwconv_features = ['HW', 'CHANNELS', 'KERNEL_SIZE', 'STRIDE']
        
        dwconv_lut_configs = pd.read_csv(prior_file_path)
        
        dwconv_df = dwconv_lut_configs[dwconv_features].copy()
        # normalize
        dwconv_df['HW'] = np.log2(dwconv_df['HW'])
        dwconv_df['CHANNELS'] = np.log2(dwconv_df['CHANNELS'])
        dwconv_data = dwconv_df.to_numpy().astype(np.float32)
        dwconv_min_val = dwconv_data.min(axis=0)
        dwconv_max_val = dwconv_data.max(axis=0)
        # normalize
        dwconv_data = (dwconv_data - dwconv_min_val) / (dwconv_max_val - dwconv_min_val)
        dwconv_scale = dwconv_max_val - dwconv_min_val
        
        z = np.random.normal(size=(sample_num, latent_dims))
        x_hat_0 = [Rdwconv(Gdwconv(decoder, row,dwconv_scale,dwconv_min_val)) for row in z]
        df = pd.DataFrame(data=np.array(x_hat_0), columns=dwconv_features)
        df['SCALED_FREQ'] = 1 / sample_num

        df = reconstraint_dwconvdata(df)
        df = df[dwconv_features]
        
        hws = df["HW"]
        cs = df["CHANNELS"]
        kernel_sizes = df["KERNEL_SIZE"]
        strides = df["STRIDE"]
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, c, kernel_size, stride in zip(hws, cs, kernel_sizes, strides):
            c = {
                'HW': hw,
                'CHANNEL_SIZE': c,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)
    
    def hier_sampling(self, sample_num, space_path, enable_hw=False, enable_tiling=False):
        dwconv_model = DepthConvSamplingModel(space_path)
        dwconv_model.construct_tree()
        
        cases = []
        if not enable_hw:
            for ks, first_space_dict in dwconv_model.hier_cfg_tree.items():
                for stride, second_space_dict in first_space_dict.items():
                    num_samples = 0
                    for hw, last_space_dict in second_space_dict.items():
                        num_samples += last_space_dict['#POINT']
                    cases.append({'ks': ks, 'stride': stride, 'num_samples': num_samples})

            
        else:
            for ks, first_space_dict in dwconv_model.hier_cfg_tree.items():
                for stride, second_space_dict in first_space_dict.items():
                    for hw, last_space_dict in second_space_dict.items():
                        num_samples = last_space_dict['#POINT']
                        cases.append({'ks': ks, 'stride': stride, 'hw': hw, 'num_samples': num_samples})
        
        ratios = []
        for d in cases:
            ratio = d['num_samples'] / dwconv_model.num_samples
            ratios.append(ratio)
    
        # calculate sampling number for each case
        total_target_num_samples = 0
        for i, ratio in enumerate(ratios):
            cases[i]['target_num'] = math.floor(ratio * sample_num)
            total_target_num_samples += cases[i]['target_num']
        # pad samples
        rest_num = sample_num - total_target_num_samples
        rest_samples_id = random.choices(range(len(cases)), k=rest_num)
        for id in rest_samples_id:
            sampled_flag = False
            cur_id = id
            while(not sampled_flag):
                if cases[cur_id]['target_num'] < cases[cur_id]['num_samples']:
                    cases[cur_id]['target_num'] += 1
                    sampled_flag = True
                else:
                    if cur_id == len(cases)-1:
                        cur_id = 0
                    else:
                        cur_id += 1
        
        # sampling
        kss = []
        strides = []
        hws = []
        css = []
        if not enable_hw:
            for case in cases:
                ks = case['ks']
                stride = case['stride']
                
                combs = []
                for hw, hw_dict in dwconv_model.hier_cfg_tree[ks][stride].items():
                    for cs in hw_dict['CHANNEL_SIZES']:
                        combs.append((hw, cs))
                samples = random.sample(combs, k=case['target_num'])
                for sample in samples:
                    hw = sample[0]
                    cs = sample[1]
                    kss.append(ks)
                    strides.append(stride)
                    hws.append(hw)
                    css.append(cs)
        else:
            for case in cases:
                ks = case['ks']
                stride = case['stride']
                hw = case['hw']
                
                combs = []
                for cs in dwconv_model.hier_cfg_tree[ks][stride][hw]['CHANNEL_SIZES']:
                    combs.append(cs)
                samples = random.sample(combs, k=case['target_num'])
                for cs in samples:
                    kss.append(ks)
                    strides.append(stride)
                    hws.append(hw)
                    css.append(cs)
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cs, kernel_size, stride in zip(hws, css, kss, strides):
            c = {
                'HW': hw,
                'CHANNEL_SIZE': cs,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)
    
    def nnmeter_sampling(self, sample_num, prior_file_path = "dwconv.csv"):
        ''' 
        Sampling configs for dwconv kernels based on dwconv zoo, which contains configuration values from existing model zoo for dwconv kernel. 
        Besides, extending some configs.
        The values are stored in prior_config_lib/dwconv.csv.
        Returned params include: (hw, cin, kernel_size, strides)
        '''
        hws, channel_sizes, kernel_sizes, strides = self.read_zoo(prior_file_path)
        new_channel_sizes = sample_based_on_distribution(channel_sizes, sample_num)
   
        count1 = int(sample_num * 0.8)
        new_hws = sample_based_on_distribution(hws,count1)
        new_kernel_sizes = sample_based_on_distribution(kernel_sizes, count1)
        new_strides = sample_based_on_distribution(strides, count1)
    
        new_kernel_sizes = data_validation(new_kernel_sizes, list(set(kernel_sizes)))
        new_strides = data_validation(new_strides, list(set(strides)))
        new_hws = data_validation(new_hws, list(set(hws)))
    
        new_hws.extend([112] * int((sample_num - count1) * 0.4) + [56] * int((sample_num - count1) * 0.4) + [28] * int((sample_num - count1) * 0.2))  
        new_kernel_sizes.extend([5] * int((sample_num - count1) * 0.4) + [7] * int((sample_num - count1) * 0.6))
        new_strides.extend([2] * int((sample_num - count1) * 0.5) + [1] * int((sample_num - count1) * 0.5))
        random.shuffle(new_hws)
        random.shuffle(new_kernel_sizes)
        random.shuffle(new_strides)

        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cs, kernel_size, stride in zip(new_hws, new_channel_sizes, new_kernel_sizes, new_strides):
            c = {
                'HW': hw,
                'CHANNEL_SIZE': cs,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]

        return self.encoding(ncfgs)
    

    def finegrained_sampling(self, configs, sample_num):
        ncfgs = []
        for cfg in configs:
            channel_sizes = sample_in_range(int(cfg['CHANNEL_SIZE'] * 0.5), int(cfg['CHANNEL_SIZE'] * 1.2), sample_num)
            for cs in channel_sizes:
                c = {
                    'HW': cfg['HW'],
                    'CHANNEL_SIZE': cs,
                    'KERNEL_SIZE': cfg['KERNEL_SIZE'],
                    'STRIDES': cfg['STRIDES'],
                }
                ncfgs.append(c)
        return self.encoding(ncfgs)

    def from_csv(self, csv_path):
        hws, css, kernel_sizes, strides = self.read_zoo(csv_path)
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cs, kernel_size, stride in zip(hws, css, kernel_sizes, strides):
            c = {
                'HW': hw,
                'CHANNEL_SIZE': cs,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)


class MatMulSampler(BaseSampler):
    def __init__(self):
        self.kernel_type = "matmul"
        self.kernels = {}
        self.save_dir = os.getenv("RESULTS")

    def read_zoo(self, filename = "matmul.csv"):
        matmul_df = pd.read_csv(filename)
        cs = matmul_df['C']
        ms = matmul_df['M']
        ns = dwconv_df["N"]
        ks = dwconv_df["K"]
        return cs, ms, ns, ks
    
    def vae_sampling(self, sample_num, prior_file_path, vae_path):
        latent_dims = 128
        encoder_hidden_dims = 256
        decoder_hidden_dims = 256
        source_dims = 4
        encoder_layers = 5
        decoder_layers = 5
        
        matmul_vae_model = VariationalAutoencoder(
            latent_dims=latent_dims,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            source_dims=source_dims,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )
        
        matmul_vae_model.load_state_dict(torch.load(vae_path, map_location='cpu'))
        
        matmul_vae_model = matmul_vae_model.cpu()
        matmul_vae_model.eval()
        decoder = matmul_vae_model.decoder
        
        def Gmatmul(model,z,scale,min_val):
            input_z = torch.from_numpy(z).to(torch.float32)
            output_x = model.forward(input_z).detach().numpy()
            output_x[output_x > 1] = 1
            output_x[output_x < 0] = 0
            return output_x * scale + min_val

        def Rmatmul(x):
            x0 = list(x.flatten())
            x0 = [np.exp2(x0[0]), np.exp2(x0[1]), np.exp2(x0[2]), np.exp2(x0[3])]
            x0 = [round(val) for val in x0]
            return x0
        
        # def reconstraint_dwconvdata(x: pd.DataFrame):
        #     x.loc[x['KERNEL_SIZE'] < 3, 'KERNEL_SIZE'] = 3
        #     x.loc[x['KERNEL_SIZE'] == 4, 'KERNEL_SIZE'] = 3
        #     x.loc[x['KERNEL_SIZE'] == 6, 'KERNEL_SIZE'] = 5
        #     x.loc[x['KERNEL_SIZE'] > 7, 'KERNEL_SIZE'] = 7
        #     x.loc[x['CHANNELS'] > 3, 'CHANNELS'] = \
        #         np.ceil(x[x['CHANNELS'] > 3]['CHANNELS'] / 8).astype(np.int32) * 8
        #     return x

        matmul_features = ['C', 'M', 'N', 'K']
        
        matmul_lut_configs = pd.read_csv(prior_file_path)
        
        matmul_df = matmul_lut_configs[matmul_features].copy()
        # normalize
        matmul_df['C'] = np.log2(matmul_df['C'])
        matmul_df['M'] = np.log2(matmul_df['M'])
        matmul_df['N'] = np.log2(matmul_df['N'])
        matmul_df['K'] = np.log2(matmul_df['K'])
        matmul_data = matmul_df.to_numpy().astype(np.float32)
        matmul_min_val = matmul_data.min(axis=0)
        matmul_max_val = matmul_data.max(axis=0)
        # normalize
        matmul_data = (matmul_data - matmul_min_val) / (matmul_max_val - matmul_min_val)
        matmul_scale = matmul_max_val - matmul_min_val
        
        z = np.random.normal(size=(sample_num, latent_dims))
        x_hat_0 = [Rmatmul(Gmatmul(decoder, row,matmul_scale,matmul_min_val)) for row in z]
        df = pd.DataFrame(data=np.array(x_hat_0), columns=matmul_features)
        df['SCALED_FREQ'] = 1 / sample_num

        df = df[matmul_features]
        
        cs = df["C"]
        ms = df["M"]
        ns = df["N"]
        ks = df["K"]
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for c, m, n, k in zip(cs, ms, ns, ks):
            conf = {
                'C': c,
                'M': m,
                'N': n,
                'K': k,
            }
            ncfgs.append(conf)
            nparams.append(self.get_flop_params(conf))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)

    def get_flop_params(self, config_dict):
        # inp_shape = config_dict['INSHAPE']
        c = config_dict['C']
        m = config_dict['M']
        n = config_dict['N']
        k = config_dict['K']
        flops = 2 * c * m * n * k
        return flops


class LinearMatMulSampler(BaseSampler):
    def __init__(self):
        self.kernel_type = "linearmatmul"
        self.kernels = {}
        self.save_dir = os.getenv("RESULTS")

    def read_zoo(self, filename = "linearmatmul.csv"):
        matmul_df = pd.read_csv(filename)
        ms = matmul_df['M']
        ns = dwconv_df["N"]
        ks = dwconv_df["K"]
        return cs, ms, ns, ks
    
    def vae_sampling(self, sample_num, prior_file_path, vae_path):
        latent_dims = 128
        encoder_hidden_dims = 256
        decoder_hidden_dims = 256
        source_dims = 3
        encoder_layers = 5
        decoder_layers = 5
        
        matmul_vae_model = VariationalAutoencoder(
            latent_dims=latent_dims,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            source_dims=source_dims,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )
        
        matmul_vae_model.load_state_dict(torch.load(vae_path, map_location='cpu'))
        
        matmul_vae_model = matmul_vae_model.cpu()
        matmul_vae_model.eval()
        decoder = matmul_vae_model.decoder
        
        def Gmatmul(model,z,scale,min_val):
            input_z = torch.from_numpy(z).to(torch.float32)
            output_x = model.forward(input_z).detach().numpy()
            output_x[output_x > 1] = 1
            output_x[output_x < 0] = 0
            return output_x * scale + min_val

        def Rmatmul(x):
            x0 = list(x.flatten())
            x0 = [np.exp2(x0[0]), np.exp2(x0[1]), np.exp2(x0[2])]
            x0 = [round(val) for val in x0]
            return x0
        
        # def reconstraint_dwconvdata(x: pd.DataFrame):
        #     x.loc[x['KERNEL_SIZE'] < 3, 'KERNEL_SIZE'] = 3
        #     x.loc[x['KERNEL_SIZE'] == 4, 'KERNEL_SIZE'] = 3
        #     x.loc[x['KERNEL_SIZE'] == 6, 'KERNEL_SIZE'] = 5
        #     x.loc[x['KERNEL_SIZE'] > 7, 'KERNEL_SIZE'] = 7
        #     x.loc[x['CHANNELS'] > 3, 'CHANNELS'] = \
        #         np.ceil(x[x['CHANNELS'] > 3]['CHANNELS'] / 8).astype(np.int32) * 8
        #     return x

        matmul_features = ['M', 'N', 'K']
        
        matmul_lut_configs = pd.read_csv(prior_file_path)
        
        matmul_df = matmul_lut_configs[matmul_features].copy()
        # normalize
        matmul_df['M'] = np.log2(matmul_df['M'])
        matmul_df['N'] = np.log2(matmul_df['N'])
        matmul_df['K'] = np.log2(matmul_df['K'])
        matmul_data = matmul_df.to_numpy().astype(np.float32)
        matmul_min_val = matmul_data.min(axis=0)
        matmul_max_val = matmul_data.max(axis=0)
        # normalize
        matmul_data = (matmul_data - matmul_min_val) / (matmul_max_val - matmul_min_val)
        matmul_scale = matmul_max_val - matmul_min_val
        
        z = np.random.normal(size=(sample_num, latent_dims))
        x_hat_0 = [Rmatmul(Gmatmul(decoder, row,matmul_scale,matmul_min_val)) for row in z]
        df = pd.DataFrame(data=np.array(x_hat_0), columns=matmul_features)
        df['SCALED_FREQ'] = 1 / sample_num

        df = df[matmul_features]
        
        ms = df["M"]
        ns = df["N"]
        ks = df["K"]
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for m, n, k in zip(ms, ns, ks):
            conf = {
                'M': m,
                'N': n,
                'K': k,
            }
            ncfgs.append(conf)
            nparams.append(self.get_flop_params(conf))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)

    def get_flop_params(self, config_dict):
        # inp_shape = config_dict['INSHAPE']
        m = config_dict['M']
        n = config_dict['N']
        k = config_dict['K']
        params = n * k
        flops = 2 * m * n * k
        return flops, params


class GEMMSampler(BaseSampler):
    def __init__(self):
        self.kernel_type = "gemm"
        self.kernels = {}
        self.save_dir = os.getenv("RESULTS")
    
    def read_zoo(self, filename = "gemm.csv"):
        conv_df = pd.read_csv(filename)
        
        hws = conv_df["HW"]
        cins = conv_df["CIN"]
        couts = conv_df["COUT"]
        self.conv_df = conv_df
        return hws, cins, couts
    
    def get_flop_params(self, config_dict):
        # inp_shape = config_dict['INSHAPE']
        hw = config_dict['HW']
        cin = config_dict['CIN']
        cout = config_dict['COUT']
        params = cin * cout
        flops = 2 * hw * params
        return flops, params

    def from_csv(self, csv_path):
        hws, cins, couts = self.read_zoo(csv_path)
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cin, cout in zip(hws, cins, couts):
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)
    
    def hier_sampling(self, sample_num, space_path, enable_hw=False, enable_tiling=False):
        gemm_model = GEMMSamplingModel(space_path)
        gemm_model.construct_tree()
        
        cases = []
        

        for hw, last_space_dict in gemm_model.hier_cfg_tree.items():
            num_samples = last_space_dict['#POINT']
            cases.append({'hw': hw, 'num_samples': num_samples})
        
        ratios = []
        for d in cases:
            ratio = d['num_samples'] / gemm_model.num_samples
            ratios.append(ratio)
    
        # calculate sampling number for each case
        total_target_num_samples = 0
        for i, ratio in enumerate(ratios):
            cases[i]['target_num'] = math.floor(ratio * sample_num)
            total_target_num_samples += cases[i]['target_num']
        # pad samples
        rest_num = sample_num - total_target_num_samples
        rest_samples_id = random.choices(range(len(cases)), k=rest_num)
        for id in rest_samples_id:
            sampled_flag = False
            cur_id = id
            while(not sampled_flag):
                if cases[cur_id]['target_num'] < cases[cur_id]['num_samples']:
                    cases[cur_id]['target_num'] += 1
                    sampled_flag = True
                else:
                    if cur_id == len(cases)-1:
                        cur_id = 0
                    else:
                        cur_id += 1
        
        # sampling
        cins = []
        couts = []
        hws = []
        
        for case in cases:
            hw = case['hw']
                
            combs = []
            for cin, temp_couts in gemm_model.hier_cfg_tree[hw]['CINS'].items():
                for cout in temp_couts:
                    combs.append((cin, cout))
            samples = random.sample(combs, k=case['target_num'])
            for sample in samples:
                cin = sample[0]
                cout = sample[1]
                hws.append(hw)
                cins.append(cin)
                couts.append(cout)
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cin, cout in zip(hws, cins, couts):
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)
    
    def prior_sampling(self, sample_num, prior_file_path = "conv.csv"):
        ''' 
        Sampling configs for conv kernels based on conv_zoo, which contains configuration values from existing model zoo for conv kernel. 
        The values are stored in prior_config_lib/conv.csv.
        Returned params include: (hw, cin, cout, kernel_size, strides)
        '''
        hws, cins, couts = self.read_zoo(prior_file_path)
        new_cins = sample_based_on_distribution(cins, sample_num)
        new_couts = sample_based_on_distribution(couts, sample_num)

        new_hws = sample_based_on_distribution(hws, sample_num)
       
        new_hws = data_validation(new_hws, list(set(hws)))
    
        random.shuffle(new_hws)

        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cin, cout in zip(new_hws, new_cins, new_couts):
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)
    
    def vae_sampling(self, sample_num, prior_file_path, vae_path):
        latent_dims = 128
        encoder_hidden_dims = 256
        decoder_hidden_dims = 256
        source_dims = 3
        encoder_layers = 5
        decoder_layers = 5
        
        gemm_vae_model = VariationalAutoencoder(
            latent_dims=latent_dims,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            source_dims=source_dims,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )
        
        gemm_vae_model.load_state_dict(torch.load(vae_path, map_location='cpu'))
        
        gemm_vae_model = gemm_vae_model.cpu()
        gemm_vae_model.eval()
        decoder = gemm_vae_model.decoder
        
        def Gdwconv(model,z,scale,min_val):
            input_z = torch.from_numpy(z).to(torch.float32)
            output_x = model.forward(input_z).detach().numpy()
            output_x[output_x > 1] = 1
            output_x[output_x < 0] = 0
            return output_x * scale + min_val

        def Rdwconv(x):
            x0 = list(x.flatten())
            x0 = [np.exp2(x0[0]), np.exp2(x0[1]), np.exp2(x0[2])]
            x0 = [round(val) for val in x0]
            return x0
        
        # def reconstraint_dwconvdata(x: pd.DataFrame):
        #     x.loc[x['KERNEL_SIZE'] < 3, 'KERNEL_SIZE'] = 3
        #     x.loc[x['KERNEL_SIZE'] == 4, 'KERNEL_SIZE'] = 3
        #     x.loc[x['KERNEL_SIZE'] == 6, 'KERNEL_SIZE'] = 5
        #     x.loc[x['KERNEL_SIZE'] > 7, 'KERNEL_SIZE'] = 7
        #     x.loc[x['CHANNELS'] > 3, 'CHANNELS'] = \
        #         np.ceil(x[x['CHANNELS'] > 3]['CHANNELS'] / 8).astype(np.int32) * 8
        #     return x

        gemm_features = ['HW', 'CIN', 'COUT']
        
        gemm_lut_configs = pd.read_csv(prior_file_path)
        
        gemm_df = gemm_lut_configs[gemm_features].copy()
        # normalize
        gemm_df['HW'] = np.log2(gemm_df['HW'])
        gemm_df['CIN'] = np.log2(gemm_df['CIN'])
        gemm_df['COUT'] = np.log2(gemm_df['COUT'])
        gemm_data = gemm_df.to_numpy().astype(np.float32)
        gemm_min_val = gemm_data.min(axis=0)
        gemm_max_val = gemm_data.max(axis=0)
        # normalize
        gemm_data = (gemm_data - gemm_min_val) / (gemm_max_val - gemm_min_val)
        gemm_scale = gemm_max_val - gemm_min_val
        
        z = np.random.normal(size=(sample_num, latent_dims))
        x_hat_0 = [Rdwconv(Gdwconv(decoder, row,gemm_scale,gemm_min_val)) for row in z]
        df = pd.DataFrame(data=np.array(x_hat_0), columns=gemm_features)
        df['SCALED_FREQ'] = 1 / sample_num

        # df = reconstraint_dwconvdata(df)
        df = df[gemm_features]
        
        hws = df["HW"]
        couts = df["CIN"]
        cins = df["COUT"]
        
        
        ncfgs = []
        nparams = [] # calculate the number of parameters for configs sort
        for hw, cin, cout in zip(hws, cins, couts):
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout
            }
            ncfgs.append(c)
            nparams.append(self.get_flop_params(c))

        # sort all sampling configs by number of parameters, from the smallest to the largest
        # the procedure is for better profiling
        ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]
        return self.encoding(ncfgs)
