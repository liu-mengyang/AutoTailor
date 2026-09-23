feature_for_kernel = {
    # conv
    "conv":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-bn-relu":         ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-bn-relu6":        ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-bn":              ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-relu":            ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-relu6":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-hswish":          ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-bn-hswish":       ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv-swish":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    # dwconv
    "dwconv":         ["HW", "CHANNEL_SIZE", "KERNEL_SIZE", "STRIDES"],
    "dwconv-bn":            ["HW", "CHANNEL_SIZE", "KERNEL_SIZE", "STRIDES"],
    "dwconv-relu":          ["HW", "CHANNEL_SIZE", "KERNEL_SIZE", "STRIDES"],
    "dwconv-relu6":         ["HW", "CHANNEL_SIZE", "KERNEL_SIZE", "STRIDES"],
    "dwconv-bn-relu":       ["HW", "CHANNEL_SIZE", "KERNEL_SIZE", "STRIDES"],
    "dwconv-bn-relu6":      ["HW", "CHANNEL_SIZE", "KERNEL_SIZE", "STRIDES"],
    "dwconv-bn-hswish":     ["HW", "CHANNEL_SIZE", "KERNEL_SIZE", "STRIDES"],
    "dwconv-swish":         ["HW", "CHANNEL_SIZE", "KERNEL_SIZE", "STRIDES"],
    # pooling
    "maxpool":              ["HW", "CHANNEL_SIZE", "KERNEL_SIZE", "POOL_STRIDES"],
    "avgpool":              ["HW", "CHANNEL_SIZE", "KERNEL_SIZE", "POOL_STRIDES"],
    # others
    "fc":                   ["CIN", "COUT"],
    "concat":               ["HW", "CIN1", "CIN2", "CIN3", "CIN4"],
    "split":                ["HW", "CIN"],
    "channelshuffle":       ["HW", "CIN"],
    "se":                   ["HW", "CIN"],
    "global-avgpool":       ["HW", "CIN"],
    "bnrelu":               ["HW", "CIN"],
    "bn":                   ["HW", "CIN"],
    "hswish":               ["HW", "CIN"],
    "swish":                ["HW", "CIN"],
    "relu":                 ["HW", "CIN"],
    # In "addrelu" block and "add" block, the second feature "CIN" will always be the same as
    # the third feature
    "addrelu":              ["HW", "CIN", "CIN"],
    "add":                  ["HW", "CIN", "CIN"], 
    "gemm":                 ["HW", "CIN", "COUT"],
    "matmul":               ["C", "M", "N", "K"],
    "linearmatmul":         ["M", "N", "K"]
}

class BaseFeatureParser:
    def __init__(self, kernel_type):
        self.kernel_type = kernel_type
        self.needed_config = feature_for_kernel[kernel_type]

    def get_feature_by_config(self, config_dict):
        feature = [config_dict[data] for data in self.needed_config]
        return feature

    def get_config_by_feature(self, feature):
        assert len(self.needed_config) == len(feature)
        config = {k: v for k, v in zip(self.needed_config, feature)}
        return config


class FlopsParamParser(BaseFeatureParser):
    def get_feature_by_config(self, sampler, config_dict):
        feature = [config_dict[data] for data in self.needed_config]
        
        flop, param = sampler.get_flop_params(config_dict)
        flop /= 2e6
        param /= 1e6
        feature.extend([flop, param])
        # print(f"feature:{feature}")
        return feature

    def get_config_by_feature(self, feature):
        # remove flops and params num feature from feature vector
        feature = feature[:-2]
        assert len(self.needed_config) == len(feature)
        config = {k: v for k, v in zip(self.needed_config, feature)}
        return config