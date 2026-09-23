


class ArchEncoder:
    def __init__(self, super_config, supercode):
        global_vars = super_config["global_vars"]
        stage_vars = super_config["stage_vars"]
        block_vars = super_config["block_vars"]
        
        # construct map of encoder
        self.map = {"global": {}, "stage":{}, "block": {}}
        # global level
        for k, v in global_vars.items():
            self.map["global"][k] = {}
            num_choices = len(v)
            for i in range(num_choices):
                code = [0] * num_choices
                code[i] = 1
                self.map["global"][k][v[i]] = code
                
        # stage level
        for k, v in stage_vars.items():
            if "skipped" in k:
                continue
            self.map["stage"][k] = {}
            num_choices = len(v)
            for i in range(num_choices):
                code = [0] * num_choices
                code[i] = 1
                self.map["stage"][k][v[i]] = code
        
        # block level
        for block_type, block_dict in block_vars.items():
            self.map["block"][block_type] = {}
            for k, v in block_dict.items():
                if "skipped" in k:
                    continue
                self.map["block"][block_type][k] = {}
                num_choices = len(v)
                for i in range(num_choices):
                    code = [0] * num_choices
                    code[i] = 1
                    self.map["block"][block_type][k][v[i]] = code
        
        self.n_dim = len(self.encode(supercode))
        
    def encode(self, subnet_code):
        code = []
        for k, v in subnet_code.items():
            if isinstance(v, dict):
                # block_level, k is the block type
                block_type = k
                # kk is transformation key
                for kk, vv in v.items():
                    for stage_vars in vv:
                        # for each stage
                        for vvv in stage_vars:
                            code += self.map["block"][k][kk][vvv]
            elif isinstance(v, list):
                # stage_level
                for vv in v:
                    code += self.map["stage"][k][vv]
            else:
                # global_level
                code += self.map["global"][k][v]
        return code