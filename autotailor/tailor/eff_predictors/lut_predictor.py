import json
import ast

class LUTPredictor(object):
    def __init__(self, lut_path_dict):
        self.lut_dict = {}
        for kernel_type, lut_path in lut_path_dict.items():
            self.lut_dict[kernel_type] = {}
            kernel_latency_dict = json.load(open(lut_path))
            for kernel_id, kernel_config in kernel_latency_dict.items():
                print(kernel_config)
                lat_str = kernel_config["latency"]
                cfg = kernel_config["config"]
                if kernel_type == "Reshape" or kernel_type == "Transpose" or kernel_type == "LayerNormalization":
                    cfg = kernel_config["config"][:-2]
                new_cfg = [] # transform all list to tuple
                for v in cfg:
                    if isinstance(v, list):
                        if isinstance(v[0], list):
                            new_cfg.append((tuple(v[0]), tuple(v[1])))
                        else:
                            new_cfg.append(tuple(v))
                    else:
                        new_cfg.append(v)
                new_cfg = tuple(new_cfg)
                lat = None
                if "+-" in lat_str:
                    lat = float(lat_str.split(" +- ")[0])
                self.lut_dict[kernel_type][new_cfg] = lat
    
    def predict_efficiency(self, op_dict):
        """
        Return None if existing illegal kernel
        """
        pred = 0
        for kernel_type, features in op_dict.items():
            if kernel_type not in self.lut_dict:
                continue
            lut = self.lut_dict[kernel_type]
            for feature in features:
                if kernel_type == "Reshape" or kernel_type == "Transpose" or kernel_type == "LayerNormalization":
                    feature = feature[:-2]
                if kernel_type == "Mul" and feature not in lut:
                    #FIXME
                    # print(f"unsupport mul type, shape {feature}")
                    continue
                # unit_pred = lut[feature]
                tup = ast.literal_eval(feature)
                unit_pred = lut[tup]
                # print(feature)
                # print(kernel_type)
                if unit_pred is None:
                    pred += 0
                else:
                    # FIXME: scale the gelu profiling offset
                    if kernel_type == "Gelu":
                        unit_pred = unit_pred * 1.5
                    pred += unit_pred
        return pred