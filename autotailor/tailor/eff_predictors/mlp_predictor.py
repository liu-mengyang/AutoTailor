import json

import torch
import torch.nn as nn

from .feature_parser import FeatureParser


# source code from LitePred[NSDI'24]
class MLP(nn.Module):
    def __init__(self,input_features=None):
        super().__init__()
        self.fc1 = nn.Linear(input_features,16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64,128)
        self.fc5 = nn.Linear(128,128)
        self.fc6 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 256)
        self.fc10 = nn.Linear(256, 256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 128)
        self.fc14 = nn.Linear(128, 64)
        self.fc15 = nn.Linear(64, 16)
        self.fc16 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x))
        x = self.relu(self.fc11(x))
        x = self.relu(self.fc12(x))
        x = self.leakyrelu(self.fc13(x))
        x = self.leakyrelu(self.fc14(x))
        x = self.relu(self.fc15(x))
        out = self.fc16(x)

        return out.squeeze(-1)


class MLPPredictor(object):
    def __init__(self, mlp_weight_path_dict):
        self.feature_parser = FeatureParser()
        
        self.model_dict = {}
        self.lut_dict = {}
        for op_type, mlp_weight_path in mlp_weight_path_dict.items():
            if op_type == "Conv" or op_type == "DepthConv" or op_type == "MatMul":
                print(self.feature_parser.num_features[op_type])
                self.model_dict[op_type] = MLP(input_features=self.feature_parser.num_features[op_type])
                self.model_dict[op_type].load_state_dict(torch.load(mlp_weight_path, map_location="cpu"))
            else:
                # not implement mlp, use lut
                self.lut_dict[op_type] = {}
                kernel_latency_dict = json.load(open(mlp_weight_path))
                for kernel_id, kernel_config in kernel_latency_dict.items():
                    lat_str = kernel_config["latency"]
                    cfg = kernel_config["config"]
                # FIXME
                    if op_type == "Reshape" or op_type == "Transpose" or op_type == "LayerNormalization":
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
                    self.lut_dict[op_type][new_cfg] = lat
    
    def predict_efficiency(self, op_dict):
        """
        Return None if existing illegal kernel
        """
        pred = 0
        for op_type, features in op_dict.items():
            if op_type in self.model_dict:
                model = self.model_dict[op_type]
                # if torch.cuda.is_available():
                #     model = model.cuda()
                model.eval()
                with torch.no_grad():
                    for feature in features:
                        feature = self.feature_parser.parse(op_type, feature)
                        data = torch.tensor(feature).float()
                        # if torch.cuda.is_available():
                        #     data = data.cuda()
                        unit_pred = float(model(data).cpu())
                        if unit_pred is None:
                            return None
                        else:
                            pred += unit_pred
            elif op_type in self.lut_dict:
                lut = self.lut_dict[op_type]
                for feature in features:
                    if op_type == "Reshape" or op_type == "Transpose" or op_type == "LayerNormalization":
                        feature = feature[:-2]
                    if op_type == "Mul" and feature not in lut:
                        #FIXME
                        #print(f"unsupport mul type, shape {feature}")
                        continue
                    unit_pred = lut[feature]
                    if unit_pred is None:
                        pred += 0
                    else:
                        pred += unit_pred
            else:
                continue
        return pred