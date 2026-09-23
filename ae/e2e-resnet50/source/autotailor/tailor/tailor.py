import copy
import os
import sys
import random

from loguru import logger 

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)


def reduced_block_type(tir, stage_id, block_types):
    # find the block_type can be reduced in one stage
    stage = tir.stages[stage_id]
    for i in range(len(stage.flow)):
        block_type = stage.flow[len(stage.flow)-i-1].type
        if block_type in block_types:
            return block_type

    return None

class Tailor(object):
    def __init__(self, tir):
        self.tir = tir
        self.supernet_config_dict = tir.supernet_cfg_dict
    
        self.global_vars = self.supernet_config_dict["var"]["global_vars"]
        self.stage_vars = self.supernet_config_dict["var"]["stage_vars"]
        self.block_vars = self.supernet_config_dict["var"]["block_vars"]
    
        for stage_id, stage in self.tir.stages.items():
            stage.cnt_blocks()

        self.num_stage = len(self.tir.stages)
        logger.info(f"The number of stages: {self.num_stage}")

        self.num_dynamic_block_dict = {}
        
        for block_type in self.block_vars.keys():
            self.num_dynamic_block_dict[block_type] = []
            for stage_id, stage in self.tir.stages.items():
                num_block = 0
                for block_id, block in stage.flow.items():
                    if block.type == block_type:
                        num_block += 1
                self.num_dynamic_block_dict[block_type].append(num_block)
        logger.info(f"The number of dynamic blocks: {self.num_dynamic_block_dict}")
        
        self.block_type_map = []
        for stage_id, stage in self.tir.stages.items():
            stage_type_map = []
            for block_id, block in stage.flow.items():
                if block.type in self.block_vars.keys():
                    stage_type_map.append(block.type)
            self.block_type_map.append(stage_type_map)
        
        self.supercode = {}
        for glob_k, glob_v in self.global_vars.items():
            self.supercode[glob_k] = max(glob_v)
        
        
        for stage_k, stage_v in self.stage_vars.items():
            if "skipped" in stage_k:
                continue
            self.supercode[stage_k] = []
            for i in range(self.num_stage):
                self.supercode[stage_k].append(max(stage_v))
                
        # Skip some stage
        self.skipcode = {}
        for stage_k, stage_v in self.stage_vars.items():
            if "skipped" in stage_k:
                dim_name = stage_k.split("_skipped")[0]
                self.skipcode[dim_name] = stage_v
        
        for block_type, block_vars in self.block_vars.items():
            self.supercode[block_type] = {}
            for block_k, block_v in block_vars.items():
                self.supercode[block_type][block_k] = []
                for i in range(self.num_stage):
                    line_code = []
                    for j in range(self.num_dynamic_block_dict[block_type][i]):
                        line_code.append(max(block_v))
                    self.supercode[block_type][block_k].append(line_code)
        
        # make sure all dimension are under supernet
        for code_k, code_v in self.supercode.items():
            if code_k in self.block_vars.keys():
                for block_k, block_v in code_v.items():
                    for stage_id, stage in self.tir.stages.items():
                        block_cnt = 0
                        for block_id, block in stage.flow.items():
                            if block.type == code_k:
                                max_v = block.features[f"max_{block_k}"]
                                self.supercode[code_k][block_k][stage_id][block_cnt] = min(max_v, self.supercode[code_k][block_k][stage_id][block_cnt])
                                block_cnt += 1
            
        # get min depth info
        self.min_depth = []
        for stage_id, stage in self.tir.stages.items():
            depth = 0
            for block_id, block in stage.flow.items():
                if block.is_transformable():
                    depth += 1
            self.min_depth.append(depth)
        logger.info(f"Min depth: {self.min_depth}")
        logger.info(f"The supercode: {self.supercode}")
    
        self.dimension_index_dict = None
    
    
    def transform(self, code):
        for code_k, code_v in code.items():
            if code_k in self.global_vars.keys():
                # Global transformation
                self.tir.transform({code_k: code_v})
            elif code_k in self.stage_vars.keys():
                if "skipped" in code_k:
                    continue
                # Stage transformation
                cur_width = self.tir.inp_shape[1]
                cur_shape = self.tir.inp_shape
                for stage_id, stage in self.tir.stages.items():
                    stage.update({"in_channel": cur_width, "in_shape": cur_shape})
                    if code_k == "reduce_depth":
                        # FIXME: keep safe depth
                        reduced_block_types = []
                        minimal_reduce_depth = []

                        reduce_num = -min(self.stage_vars["reduce_depth"])
                        for j in range(self.num_stage):
                            reduced_block_types.append(reduced_block_type(self.tir, j, list(self.block_vars.keys())))
                        # print(f"reduced block types: {reduced_block_types}")
                        skipcode = {}
                        for stage_k, stage_v in self.stage_vars.items():
                            if "skipped" in stage_k:
                                dim_name = stage_k.split("_skipped")[0]
                                skipcode[dim_name] = stage_v
                        
                        # transform skipcode to positive values
                        for dim_name, skipids in skipcode.items():
                            for i in range(len(skipids)):
                                if skipcode[dim_name][i] < 0:
                                    skipcode[dim_name][i] += self.num_stage

                        for i in range(self.num_stage):
                            if i not in skipcode["reduce_depth"]:
                                stage_num_blocks_total = 0
                                # count # of blocks in this stage
                                block_types = list(self.block_vars.keys())
                                for block_type in block_types:
                                    stage_num_blocks_total += len(list(self.supercode[block_type].values())[0][i])
                                # print(i)
                                stage_num_blocks = len(list(self.supercode[reduced_block_types[i]].values())[0][i])
                                # print(f"{i}: {stage_num_blocks}")
                                reduce_num_temp = min(min(reduce_num, stage_num_blocks-1), stage_num_blocks_total-2)
                                minimal_reduce_depth.append(-reduce_num_temp)
                            else:
                                minimal_reduce_depth.append(0)
                        
                        stage.transform({code_k: max(code_v[stage_id], minimal_reduce_depth[stage_id])})
                    else:
                        stage.transform({code_k: code_v[stage_id]})

                    cur_width = stage.features["out_channel"]
                    cur_shape = stage.features["out_shape"]
            elif code_k in self.block_vars.keys():
                for block_k, block_v in code_v.items():
                    # Block transformation
                    cur_width = self.tir.inp_shape[1]
                    cur_shape = self.tir.inp_shape
                    for stage_id, stage in self.tir.stages.items():
                        block_cnt = 0
                        for block_id, block in stage.flow.items():
                            block.update({"in_channel": cur_width, "in_shape": cur_shape})
                            if block.type == code_k:
                                block.transform({block_k: block_v[stage_id][block_cnt]})
                                block_cnt += 1
                            cur_width = block.features["out_channel"]
                            cur_shape = block.features["out_shape"]
        
    def min_sample(self):
        if self.dimension_index_dict:
            enable_dimensions = self.dimension_index_dict["enable_dimensions"]
            dynamic_dimension = self.dimension_index_dict["dynamic_dimension"]
            dynamic_dimension_index = self.dimension_index_dict["dynamic_dimension_index"]
            
            code = {}
            # globa_level
            for glob_k, glob_v in self.global_vars.items():
                code[glob_k] = min(glob_v)
            # stage level
            for stage_k, stage_v in self.stage_vars.items():
                if "skipped" in stage_k:
                    continue
                code[stage_k] = []
                
                # get choice pool
                choice_pool = []
                
                if stage_k == dynamic_dimension:
                    for i in dynamic_dimension_index:
                        if i >= len(stage_v):
                            break
                        choice_pool.append(stage_v[i])
                elif stage_k not in enable_dimensions:
                    # use super value
                    for i in range(self.num_stage):
                        code[stage_k].append(max(stage_v))
                    continue
                else:
                    choice_pool = stage_v
                # choose min from choice pool
                for i in range(self.num_stage):
                    if stage_k == "reduce_depth":
                        code[stage_k].append(max(min(choice_pool),
                                            1-self.min_depth[i]))
                    else:
                        code[stage_k].append(min(choice_pool))
            for block_type, block_vars in self.block_vars.items():
                code[block_type] = {}
                for block_k, block_v in block_vars.items():
                    code[block_type][block_k] = []
                    
                    # get choice pool
                    choice_pool = []
                    
                    if block_k == dynamic_dimension:
                        for i in dynamic_dimension_index:
                            if i >= len(block_v):
                                break
                            choice_pool.append(block_v[i])
                    elif block_k not in enable_dimensions:
                        # use super value
                        for i in range(self.num_stage):
                            line_code = []
                            for j in range(self.num_dynamic_block_dict[block_type][i]):
                                line_code.append(max(block_v))
                            code[block_type][block_k].append(line_code)
                        continue
                    else:
                        choice_pool = block_v
                    
                    for i in range(self.num_stage):
                        line_code = []
                        for j in range(self.num_dynamic_block_dict[block_type][i]):
                            line_code.append(min(choice_pool))
                        code[block_type][block_k].append(line_code)
            
            # skip some stage
            for stage_k, stage_v in self.skipcode.items():
                for stage_id in stage_v:
                    if stage_k in self.block_vars:
                        # skip blocks
                        for block_id in range(self.num_dynamic_block_dict[stage_k][stage_id]):
                            for block_k in code[stage_k]:
                                code[stage_k][block_k][stage_id][block_id] = self.supercode[stage_k][block_k][stage_id][block_id]
                    else:
                        # skip stages
                        code[stage_k][stage_id] = max(self.stage_vars[stage_k])
        else:
            code = {}
            for glob_k, glob_v in self.global_vars.items():
                code[glob_k] = min(glob_v)
            for stage_k, stage_v in self.stage_vars.items():
                if "skipped" in stage_k:
                    continue
                code[stage_k] = []
                for i in range(self.num_stage):
                    if stage_k == "reduce_depth":
                        code[stage_k].append(max(min(stage_v),
                                            1-self.min_depth[i]))
                    else:
                        code[stage_k].append(min(stage_v))
            for block_type, block_vars in self.block_vars.items():
                code[block_type] = {}
                for block_k, block_v in block_vars.items():
                    code[block_type][block_k] = []
                    for i in range(self.num_stage):
                        line_code = []
                        for j in range(self.num_dynamic_block_dict[block_type][i]):
                            line_code.append(min(block_v))
                        code[block_type][block_k].append(line_code)
            
            # skip some stage
            for stage_k, stage_v in self.skipcode.items():
                for stage_id in stage_v:
                    if stage_k in self.block_vars:
                        # skip blocks
                        for block_id in range(self.num_dynamic_block_dict[stage_k][stage_id]):
                            for block_k in code[stage_k]:
                                code[stage_k][block_k][stage_id][block_id] = self.supercode[stage_k][block_k][stage_id][block_id]
                    else:
                        # skip stages
                        code[stage_k][stage_id] = max(self.stage_vars[stage_k])
        
        return code
    
    def random_sample(self):
        code = {}
        for glob_k, glob_v in self.global_vars.items():
            code[glob_k] = random.choice(glob_v)
        for stage_k, stage_v in self.stage_vars.items():
            if "skipped" in stage_k:
                continue
            code[stage_k] = []
            for i in range(self.num_stage):
                if stage_k == "reduce_depth":
                    code[stage_k].append(max(random.choice(stage_v),
                                         1-self.min_depth[i]))
                else:
                    code[stage_k].append(random.choice(stage_v))
                
        for block_type, block_vars in self.block_vars.items():
            code[block_type] = {}
            for block_k, block_v in block_vars.items():
                code[block_type][block_k] = []
                for i in range(self.num_stage):
                    line_code = []
                    for j in range(self.num_dynamic_block_dict[block_type][i]):
                        line_code.append(random.choice(block_v))
                    code[block_type][block_k].append(line_code)
        
        # skip some stage
        for stage_k, stage_v in self.skipcode.items():
            for stage_id in stage_v:
                if stage_k in self.block_vars:
                    # skip blocks
                    for block_id in range(self.num_dynamic_block_dict[stage_k][stage_id]):
                        for block_k in code[stage_k]:
                            code[stage_k][block_k][stage_id][block_id] = self.supercode[stage_k][block_k][stage_id][block_id]
                else:
                    # skip stages
                    code[stage_k][stage_id] = max(self.stage_vars[stage_k])
        
        return code
    
    def compound_sample(self):
        # refer from CompOFA [ICLR'21]
        
        code = {}
        # keep resolution elastic
        for glob_k, glob_v in self.global_vars.items():
            code[glob_k] = random.choice(glob_v)
            
        # other dimensions are coupling
        ## find the maximum number of dimensions
        comp_idx = []
        max_num_dimensions = 0
        for stage_k, stage_v in self.stage_vars.items():
            if "skipped" in stage_k:
                continue
            num_dimensions = len(stage_v)
            if num_dimensions > max_num_dimensions:
                max_num_dimensions = num_dimensions
        for block_type, block_vars in self.block_vars.items():
            for block_k, block_v in block_vars.items():
                num_dimensions = len(block_v)
                if num_dimensions > max_num_dimensions:
                    max_num_dimensions = num_dimensions
        
        ## random sampling for each stage
        for i in range(self.num_stage):
            random_idx = random.choice(range(max_num_dimensions))
            comp_idx.append(random_idx)
        
        for stage_k, stage_v in self.stage_vars.items():
            if "skipped" in stage_k:
                continue
            code[stage_k] = []
            for i in range(self.num_stage):
                scaled_random_idx = int(comp_idx[i] * (len(stage_v) / max_num_dimensions))
                if stage_k == "reduce_depth":
                    code[stage_k].append(max(stage_v[scaled_random_idx],
                                         1-self.min_depth[i]))
                else:
                    code[stage_k].append(stage_v[scaled_random_idx])
                
        for block_type, block_vars in self.block_vars.items():
            code[block_type] = {}
            for block_k, block_v in block_vars.items():
                code[block_type][block_k] = []
                for i in range(self.num_stage):
                    scaled_random_idx = int(comp_idx[i] * (len(block_v) / max_num_dimensions))
                    line_code = []
                    for j in range(self.num_dynamic_block_dict[block_type][i]):
                        line_code.append(block_v[scaled_random_idx])
                    code[block_type][block_k].append(line_code)
        
        # skip some stage
        for stage_k, stage_v in self.skipcode.items():
            for stage_id in stage_v:
                if stage_k in self.block_vars:
                    # skip blocks
                    for block_id in range(self.num_dynamic_block_dict[stage_k][stage_id]):
                        for block_k in code[stage_k]:
                            code[stage_k][block_k][stage_id][block_id] = self.supercode[stage_k][block_k][stage_id][block_id]
                else:
                    # skip stages
                    code[stage_k][stage_id] = max(self.stage_vars[stage_k])
        return code
    
    def dummy_sample(self, num_subnets):
        # generate one batch of subnets with the most similar architecture to
        # the supernet
        # (currently only enabling depth transformations)
        code = copy.deepcopy(self.supercode)
        batch_subnet_codes = [self.supercode]
        stage_idx = 1
        var_idx = 1
        
        for i in range(num_subnets-1):
            # find an appropriate stage id
            while True:
                if stage_idx <= len(code["reduce_depth"]):
                    if -stage_idx in self.skipcode["reduce_depth"] or len(code["reduce_depth"])-stage_idx in self.skipcode["reduce_depth"]:
                        # skip this stage
                        stage_idx += 1
                    else:
                        break
                else:
                    # cannot find
                    raise NotImplementedError
            code["reduce_depth"][-stage_idx] = self.stage_vars["reduce_depth"][var_idx]
            if var_idx == len(self.stage_vars["reduce_depth"])-1:
                # reduce the previous stage
                stage_idx += 1
                var_idx = 1
            else:
                var_idx += 1
            batch_subnet_codes.append(copy.deepcopy(code))
                
        return batch_subnet_codes
        
    
    def sample_subnet(self, compound=False):
        if compound:
            return self.compound_sample()
        # TODO: merge this into random_sample()
        elif self.dimension_index_dict:
            enable_dimensions = self.dimension_index_dict["enable_dimensions"]
            dynamic_dimension = self.dimension_index_dict["dynamic_dimension"]
            dynamic_dimension_index = self.dimension_index_dict["dynamic_dimension_index"]
            
            code = {}
            # globa_level
            for glob_k, glob_v in self.global_vars.items():
                code[glob_k] = random.choice(glob_v)
            # stage level
            for stage_k, stage_v in self.stage_vars.items():
                if "skipped" in stage_k:
                    continue
                code[stage_k] = []
                
                # get choice pool
                choice_pool = []
                
                if stage_k == dynamic_dimension:
                    for i in dynamic_dimension_index:
                        choice_pool.append(stage_v[i])
                elif stage_k not in enable_dimensions:
                    # use super value
                    for i in range(self.num_stage):
                        code[stage_k].append(max(stage_v))
                    continue
                else:
                    choice_pool = stage_v
                # random choose from choice pool
                for i in range(self.num_stage):
                    if stage_k == "reduce_depth":
                        code[stage_k].append(max(random.choice(choice_pool),
                                            1-self.min_depth[i]))
                    else:
                        code[stage_k].append(random.choice(choice_pool))
            for block_type, block_vars in self.block_vars.items():
                code[block_type] = {}
                for block_k, block_v in block_vars.items():
                    code[block_type][block_k] = []
                    
                    # get choice pool
                    choice_pool = []
                    
                    if block_k == dynamic_dimension:
                        for i in dynamic_dimension_index:
                            choice_pool.append(block_v[i])
                    elif block_k not in enable_dimensions:
                        # use super value
                        for i in range(self.num_stage):
                            line_code = []
                            for j in range(self.num_dynamic_block_dict[block_type][i]):
                                line_code.append(max(block_v))
                            code[block_type][block_k].append(line_code)
                        continue
                    else:
                        choice_pool = block_v
                    
                    for i in range(self.num_stage):
                        line_code = []
                        for j in range(self.num_dynamic_block_dict[block_type][i]):
                            line_code.append(random.choice(choice_pool))
                        code[block_type][block_k].append(line_code)
            
            # skip some stage
            for stage_k, stage_v in self.skipcode.items():
                for stage_id in stage_v:
                    if stage_k in self.block_vars:
                        # skip blocks
                        for block_id in range(self.num_dynamic_block_dict[stage_k][stage_id]):
                            for block_k in code[stage_k]:
                                code[stage_k][block_k][stage_id][block_id] = self.supercode[stage_k][block_k][stage_id][block_id]
                    else:
                        # skip stages
                        code[stage_k][stage_id] = max(self.stage_vars[stage_k])
            
            return code
        else:
            return self.random_sample()
        
    def adapt(self,
              efficiency_metric="flops",
              efficiency_value=600000,
              accuracy_metric="flops",
              mode="evolution",
              acc_predictor_weight=None,
              ):
        optimizer = OptimizerFramework(self.tir,
                                       efficiency_metric,
                                       efficiency_value,
                                       efficiency_predictor,
                                       acc_predictor,
                                       optimizer="beam_evolution")
    
    @property
    def dynamic_blocks(self):
        for stage_id, stage in self.tir.stages.items():
            logger.info(f'##### STAGE {stage_id} #####')
            logger.info(f"Super depths: {stage.super_depth}")
            for block_id, block in stage.flow.items():
                block.features['activated'] = True # activate all blocks for sampling
                if block.features['trans_type'] == 'active':
                    logger.info(block.type)