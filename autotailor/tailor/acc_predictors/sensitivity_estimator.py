import json
from numbers import Real


def _decode_base_accuracy(value):
    """Decode current ``[accuracy]`` and legacy metric-list layouts."""
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], Real)
        and not isinstance(value[0], bool)
    ):
        return float(value[0])
    if (
        isinstance(value, list)
        and len(value) == 3
        and isinstance(value[2], list)
        and len(value[2]) == 1
        and isinstance(value[2][0], Real)
        and not isinstance(value[2][0], bool)
    ):
        return float(value[2][0])
    raise ValueError(
        "base_acc must be [accuracy] or legacy [[flops], [params], [accuracy]]"
    )


class SensitivityEstimator(object):
    def __init__(self, tailor, trans_weight_path, bottomup_mode=False, miniacc=None):
        self.bottomup_mode = bottomup_mode
        self.miniacc = miniacc
        self.tailor = tailor
        self.weights = self._load_weights(trans_weight_path)
        
    def _load_weights(self, weights_path):
        weights_dict = json.load(open(weights_path))
        
        self.base_acc = _decode_base_accuracy(weights_dict["base_acc"])
        if self.bottomup_mode:
            self.base_acc = self.miniacc
        weights = {}

        # Init delta acc
        acc_dict = weights_dict["acc"]
        for k, v in acc_dict.items():
            if isinstance(v, list):
                weights[k] = []
                for acc in v:
                    delta_acc = acc - self.base_acc
                    if self.bottomup_mode:
                        delta_acc = acc - self.miniacc
                    weights[k].append(delta_acc)
            elif isinstance(v, dict):
                weights[k] = {}
                for kk, vv in v.items():
                    weights[k][kk] = []
                    for acc in vv:
                        delta_acc = acc - self.base_acc
                        if self.bottomup_mode:
                            delta_acc = acc - self.miniacc
                        weights[k][kk].append(delta_acc)
        
        # Pad weight dict
        new_weights = {}

        for k, v in weights.items():
            if isinstance(v, list):
                new_weights[k] = []
                if k in self.tailor.global_vars:
                    # global dimension
                    pos = 0
                    max_v = max(self.tailor.global_vars[k])
                    for global_var in self.tailor.global_vars[k]:
                        if global_var == max_v:
                            new_weights[k].append(0.0)
                        else:
                            new_weights[k].append(v[pos])
                            pos += 1
                elif k in self.tailor.stage_vars:
                    # stage dimension
                    if "skipped" in k:
                        continue
                    pos = 0
                    max_v = max(self.tailor.stage_vars[k])
                    num_stages = len(self.tailor.supercode[k])
                    skip_stage_index = []
                    if f"{k}_skipped" in self.tailor.stage_vars:
                        for stage_index in self.tailor.stage_vars[f"{k}_skipped"]:
                            if stage_index < 0:
                                skip_stage_index.append(num_stages+stage_index)
                            else:
                                skip_stage_index.append(stage_index)
                    for stage_var in self.tailor.stage_vars[k]:
                        for stage_i in range(num_stages):
                            if stage_var == max_v or stage_i in skip_stage_index:
                                new_weights[k].append(0.0)
                            else:
                                new_weights[k].append(v[pos])
                                pos+=1
            elif isinstance(v, dict):
                # block dimension
                new_weights[k] = {}
                for kk, vv in v.items():
                    pos = 0
                    new_weights[k][kk] = []
                    num_stages = len(self.tailor.supercode[k][kk])
                    max_v = max(self.tailor.block_vars[k][kk])
                    for block_var in self.tailor.block_vars[k][kk]:
                        for stage_i in range(num_stages):
                            num_blocks = len(self.tailor.supercode[k][kk][stage_i])
                            if num_blocks == 0:
                                new_weights[k][kk].append([])
                                continue
                            temp_list = []
                            for block_i in range(num_blocks):
                                if block_var == max_v:
                                    temp_list.append(0.0)
                                else:
                                    temp_list.append(vv[pos])
                                    pos += 1
                            new_weights[k][kk].append(temp_list)
        
        return new_weights

    def predict_accuracy(self, code):
        self.tailor.transform(code)
        acc = self.base_acc
        for k, v in code.items():
            if k in self.tailor.global_vars:
                # global dim
                global_var_index = self.tailor.global_vars[k].index(v)
                delta_acc = self.weights[k][global_var_index]
                acc += delta_acc
            elif k in self.tailor.stage_vars:
                if "skipped" in k:
                    continue
                num_stages = len(self.tailor.supercode[k])
                for stage_i, stage_value in enumerate(v):
                    stage_var_index = self.tailor.stage_vars[k].index(stage_value)
                    delta_acc = self.weights[k][stage_var_index*num_stages+stage_i]
                    acc += delta_acc
            elif k in self.tailor.block_vars:
                for kk, vv in v.items():
                    num_stages = len(self.tailor.supercode[k][kk])
                    for stage_i, stage_value_list in enumerate(vv):
                        for block_i, block_v in enumerate(stage_value_list):
                            block_var_index = self.tailor.block_vars[k][kk].index(block_v)
                            delta_acc = self.weights[k][kk][block_var_index*num_stages+stage_i][block_i]
                            acc += delta_acc
        return acc
