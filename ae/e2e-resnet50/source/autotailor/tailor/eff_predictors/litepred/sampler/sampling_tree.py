import pandas as pd


class SamplingTree:
    def __init__(self,
                 design_space_path):
        self.design_space_path = design_space_path
        
        self.hier_cfg_tree = {}
        
    def construct_tree(self):
        pass
   

class ConvSamplingModel(SamplingTree):
    cost_models = {}
    tiling_strides = [8, 4, 2, 1]
    sampling_strides = [32, 16, 8, 4, 2]
    num_samples = 0
    
    def read_zoo(self, filename):
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
    
    def construct_tree(self):
        hws, cins, couts, kernel_sizes, strides = self.read_zoo(self.design_space_path)
        
        for i in range(len(hws)):
            ks = kernel_sizes[i]
            stride = strides[i]
            hw = hws[i]
            cin = cins[i]
            cout = couts[i]

            if ks not in self.hier_cfg_tree:
                self.hier_cfg_tree[ks] = {}
            if stride not in self.hier_cfg_tree[ks]:
                self.hier_cfg_tree[ks][stride] = {}
            if hw not in self.hier_cfg_tree[ks][stride]:
                self.hier_cfg_tree[ks][stride][hw] = {'CINS': {}, 'COUTS': {}}
            if cin not in self.hier_cfg_tree[ks][stride][hw]['CINS']:
                self.hier_cfg_tree[ks][stride][hw]['CINS'][cin] = []
            if cout not in self.hier_cfg_tree[ks][stride][hw]['COUTS']:
                self.hier_cfg_tree[ks][stride][hw]['COUTS'][cout] = []

            self.hier_cfg_tree[ks][stride][hw]['COUTS'][cout].append(cin)
            self.hier_cfg_tree[ks][stride][hw]['CINS'][cin].append(cout)
        
        for ks, first_dict in self.hier_cfg_tree.items():
            for stride, second_dict in first_dict.items():
                for hw, final_dict in second_dict.items():
                    self.hier_cfg_tree[ks][stride][hw]['MIN_CIN'] = min(list(final_dict['CINS'].keys()))
                    self.hier_cfg_tree[ks][stride][hw]['MAX_CIN'] = max(list(final_dict['CINS'].keys()))
                    self.hier_cfg_tree[ks][stride][hw]['MIN_COUT'] = min(list(final_dict['COUTS'].keys()))
                    self.hier_cfg_tree[ks][stride][hw]['MAX_COUT'] = max(list(final_dict['COUTS'].keys()))
                    num_points = 0
                    for cin, couts in final_dict['CINS'].items():
                        for cout in couts:
                            num_points += 1
                    self.hier_cfg_tree[ks][stride][hw]['#POINT'] = num_points
                    self.num_samples += num_points
        
    def tiling(self):
        for ks, first_dict in self.hier_cost_model_tree.items():
            for stride, second_dict in first_dict.items():
                for hw, final_dict in second_dict.items():
                    # skip profiling type
                    if final_dict['BUILDING_TYPE'] == 'profiling':
                        continue
                    
                    cin_range = final_dict['CIN_RANGE']
                    cout_range = final_dict['COUT_RANGE']
        
                    cins = list(range(cin_range[0], cin_range[1]+1))
                    couts = list(range(cout_range[0], cout_range[1]+1))

                    cins_tilings = [[], [], [], []]
                    couts_tilings = [[], [], [], []]

                    for cin in cins:
                        if cin % 8 == 0:
                            cins_tilings[0].append(cin)
                        elif cin % 4 == 0:
                            cins_tilings[1].append(cin)
                        elif cin % 2 == 0:
                            cins_tilings[2].append(cin)
                        else:
                            cins_tilings[3].append(cin)

                    for cout in couts:
                        if cout % 8 == 0:
                            couts_tilings[0].append(cout)
                        elif cout % 4 == 0:
                            couts_tilings[1].append(cout)
                        elif cout% 2 == 0:
                            couts_tilings[2].append(cout)
                        else:
                            couts_tilings[3].append(cout)
                    
                    self.hier_cost_model_tree[ks][stride][hw]['CINS_TILINGS'] = cins_tilings
                    self.hier_cost_model_tree[ks][stride][hw]['COUTS_TILINGS'] = couts_tilings
        
    def sampling(self):
        features = []
        test_features = []
        for tag, cost_model in self.cost_models.items():
            print(cost_model)
            ks = cost_model.ks
            stride = cost_model.stride
            hw = cost_model.hw
            
            
            for cin in cost_model.sampled_cins:
                for cout in cost_model.sampled_couts:
                    feature = []
                    feature.append(hw)
                    feature.append(ks)
                    feature.append(stride)
                    feature.append(cin)
                    feature.append(cout)
                    features.append(feature)
            
            num_sampled_points = len(cost_model.sampled_cins) * len(cost_model.sampled_couts)
            num_test_points = int(num_sampled_points * 0.5)
            
            out_of_sampled_cins = [cin for cin in cost_model.cins if cin not in cost_model.sampled_cins]
            out_of_sampled_couts = [cout for cout in cost_model.couts if cout not in cost_model.sampled_couts]
            
            if len(out_of_sampled_cins) != 0:
                test_cins = random.choices(out_of_sampled_cins, k=num_test_points)
            else:
                test_cins = random.choices(cost_model.sampled_cins, k=num_test_points)
            if len(out_of_sampled_couts) != 0:
                test_couts = random.choices(out_of_sampled_couts, k=num_test_points)
            else:
                test_couts = random.choices(cost_model.sampled_couts, k=num_test_points)
            
            for i in range(len(test_cins)):
                test_feature = []
                test_feature.append(hw)
                test_feature.append(ks)
                test_feature.append(stride)
                test_feature.append(test_cins[i])
                test_feature.append(test_couts[i])
                test_features.append(test_feature)
        
        return features, test_features
    
    def load_profiled_results(self, profiled_path):
        profiled_results = json.load(open(profiled_path))
        
        for tag, cost_model in self.cost_models.items():
            cost_model.load_profiled_dict(profiled_results)
    
    def build_cost_models(self):
        for tag, cost_model in self.cost_models.items():
            cost_model.build()
    
    def validate_cost_models(self, profiled_path):
        profiled_results = json.load(open(profiled_path))
        
        for tag, cost_model in self.cost_models.items():
            cost_model.validate(profiled_results)
    
    def validate_tree(self, profiled_path):
        profiled_results = json.load(open(profiled_path))
        predict_latencies = []
        validate_latencies = []
        for profile_tag, profiled_dict in profiled_results.items():
            for id, config_dict in profiled_dict.items():
                config = config_dict['config']
                latency = float(config_dict['latency'].split(' +- ')[0])
                
                ks = config['KERNEL_SIZE']
                stride = config['STRIDES']
                hw = config['HW']
                
                cin = config['CIN']
                cout = config['COUT']
                
                # check cin tiling
                for stride in self.tiling_strides:
                    if cin % stride == 0:
                        cin_tiling = stride
                        break
                # check cout tiling
                for stride in self.tiling_strides:
                    if cout % stride == 0:
                        cout_tiling = stride
                        break
                
                tag = f'k{ks}s{stride}hw{hw}_{cin_tiling}x{cout_tiling}'
                if tag in self.cost_models:
                    cost_model = self.cost_models[tag]
                    predict_latency = cost_model.predict(config)
                    predict_latencies.append(predict_latency)
                    validate_latencies.append(latency)
                else:
                    print(f'Cost model {tag} has not built')
        
        rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(predict_latencies, validate_latencies)
        print(f'The result of validating cost model tree')
        print(f'rmse: {rmse}')
        print(f'acc5: {acc5}')
        print(f'acc10: {acc10}')
        print(f'acc15: {acc15}')
        
        return rmse, rmspe, error, acc5, acc10, acc15


class DepthConvSamplingModel(SamplingTree):
    cost_models = {}
    tiling_strides = [8, 4, 2, 1]
    sampling_strides = [32, 16, 8, 4, 2]
    num_samples = 0
    
    def read_zoo(self, filename):
        conv_df = pd.read_csv(filename)
        if 'HW' in conv_df:
            hws = conv_df['HW']
        else:
            # old csv file from nn-meter
            hws = conv_df["input_h"]
        css = conv_df["CHANNEL_SIZE"]
        ks = conv_df["KERNEL_SIZE"]
        strides = conv_df["STRIDE"]
        return hws, css, ks, strides
    
    def construct_tree(self):
        hws, css, kernel_sizes, strides = self.read_zoo(self.design_space_path)
        
        for i in range(len(hws)):
            ks = kernel_sizes[i]
            stride = strides[i]
            hw = hws[i]
            cs = css[i]

            if ks not in self.hier_cfg_tree:
                self.hier_cfg_tree[ks] = {}
            if stride not in self.hier_cfg_tree[ks]:
                self.hier_cfg_tree[ks][stride] = {}
            if hw not in self.hier_cfg_tree[ks][stride]:
                self.hier_cfg_tree[ks][stride][hw] = {'CHANNEL_SIZES':[]}
            if cs not in self.hier_cfg_tree[ks][stride][hw]['CHANNEL_SIZES']:
                self.hier_cfg_tree[ks][stride][hw]['CHANNEL_SIZES'].append(cs)
            
        
        for ks, first_dict in self.hier_cfg_tree.items():
            for stride, second_dict in first_dict.items():
                for hw, last_dict in second_dict.items():
                    cs_lst = last_dict['CHANNEL_SIZES']
                    self.hier_cfg_tree[ks][stride][hw]['MIN_CHANNEL_SIZE'] = min(cs_lst)
                    self.hier_cfg_tree[ks][stride][hw]['MAX_CHANNEL_SIZE'] = max(cs_lst)
                    num_points = 0
                    for cs in cs_lst:
                        num_points += 1
                    self.hier_cfg_tree[ks][stride][hw]['#POINT'] = num_points
                    self.num_samples += num_points
        
    def tiling(self):
        raise NotImplementedError
    

class GEMMSamplingModel(SamplingTree):
    cost_models = {}
    sampling_strides = [32, 16, 8, 4, 2]
    num_samples = 0
    
    def read_zoo(self, filename):
        conv_df = pd.read_csv(filename)
        if 'HW' in conv_df:
            hws = conv_df['HW']
        else:
            # old csv file from nn-meter
            hws = conv_df["input_h"]
        cins = conv_df["CIN"]
        couts = conv_df["COUT"]
        return hws, cins, couts
    
    def construct_tree(self):
        hws, cins, couts = self.read_zoo(self.design_space_path)
        
        for i in range(len(hws)):
            cout = couts[i]
            hw = hws[i]
            cin = cins[i]


            if hw not in self.hier_cfg_tree:
                self.hier_cfg_tree[hw] = {'CINS': {}, 'COUTS': {}}
            if cin not in self.hier_cfg_tree[hw]['CINS']:
                self.hier_cfg_tree[hw]['CINS'][cin] = []
            if cout not in self.hier_cfg_tree[hw]['COUTS']:
                self.hier_cfg_tree[hw]['COUTS'][cout] = []

            self.hier_cfg_tree[hw]['COUTS'][cout].append(cin)
            self.hier_cfg_tree[hw]['CINS'][cin].append(cout)
            
        
        for hw, final_dict in self.hier_cfg_tree.items():
            self.hier_cfg_tree[hw]['MIN_CIN'] = min(list(final_dict['CINS'].keys()))
            self.hier_cfg_tree[hw]['MAX_CIN'] = max(list(final_dict['CINS'].keys()))
            self.hier_cfg_tree[hw]['MIN_COUT'] = min(list(final_dict['COUTS'].keys()))
            self.hier_cfg_tree[hw]['MAX_COUT'] = max(list(final_dict['COUTS'].keys()))
            num_points = 0
            for cin, couts in final_dict['CINS'].items():
                for cout in couts:
                    num_points += 1
            self.hier_cfg_tree[hw]['#POINT'] = num_points
            self.num_samples += num_points
        
    def tiling(self):
        raise NotImplementedError


class MatMulSamplingModel(SamplingTree):
    cost_models = {}
    sampling_strides = [32, 16, 8, 4, 2]
    num_samples = 0
    
    def read_zoo(self, filename):
        conv_df = pd.read_csv(filename)
        hws = conv_df['HW']
        ms = conv_df["M"]
        ks = conv_df["K"]
        ns = conv_df["N"]
        return hws, ms, ks, ns
    
    def construct_tree(self):
        hws, ms, ks, ns = self.read_zoo(self.design_space_path)
        
        for i in range(len(hws)):
            hw = hws[i]
            m = ms[i]
            k = ks[i]
            n = ns[i]

            if hw not in self.hier_cfg_tree:
                self.hier_cfg_tree[hw] = {'M':{}, 'K':{}, 'N':{}}
            if m not in self.hier_cfg_tree[hw]:
                self.hier_cfg_tree[hw][m] = {}
                self.hier_cfg_tree[hw]['M'][m] = []
            if k not in self.hier_cfg_tree[hw][m]:
                self.hier_cfg_tree[hw][m][k] = []
                self.hier_cfg_tree[hw]['K'][k] = []
            if n not in self.hier_cfg_tree[hw][m][k]:
                self.hier_cfg_tree[hw][m][k].append(n)
                self.hier_cfg_tree[hw]['N'][n] = []
            

            self.hier_cfg_tree[hw]['M'][m].append({'K': k, 'N': n})
            self.hier_cfg_tree[hw]['K'][k].append({'M': m, 'N': n})
            self.hier_cfg_tree[hw]['N'][n].append({'M': m, 'K': k})
            
        
        for hw, final_dict in self.hier_cfg_tree.items():
            self.hier_cfg_tree[hw]['MIN_M'] = min(list(final_dict['M'].keys()))
            self.hier_cfg_tree[hw]['MAX_M'] = max(list(final_dict['M'].keys()))
            self.hier_cfg_tree[hw]['MIN_K'] = min(list(final_dict['K'].keys()))
            self.hier_cfg_tree[hw]['MAX_K'] = max(list(final_dict['K'].keys()))
            self.hier_cfg_tree[hw]['MIN_N'] = min(list(final_dict['N'].keys()))
            self.hier_cfg_tree[hw]['MAX_N'] = max(list(final_dict['N'].keys()))
            num_points = 0
            for m, kns in final_dict['M'].items():
                for kn in kns:
                    num_points += 1
            self.hier_cfg_tree[hw]['#POINT'] = num_points
            self.num_samples += num_points
        
    def tiling(self):
        raise NotImplementedError

