import copy
import os
import sys
import random
import time

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)

from autotailor.tailor.eff_predictors.flops_counter import FLOPsCounter
from autotailor.tailor.eff_predictors.block_lut_predictor import BlockLUTPredictor

__all__ = ["OptimizerFramework"]

class OptimizerFramework:
    def __init__(
            self,
            tailor,
            efficiency_constraint,
            efficiency_predictor,
            accuracy_predictor,
            optimizer="beam_evolution",
    ):
        self.efficiency_constraint = efficiency_constraint

        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.tailor = tailor

        self.config_space = self._get_config_space()
        self.profiler = self._profiler_factory()

        random.seed(time.time())

        if optimizer == "random":
            from autotailor.tailor.optimizer.random import Random
            self.optimizer = Random(self.config_space, self.profiler, self.efficiency_constraint)
        elif optimizer == "beam_search":
            # from autotailor.tailor.optimizer.beam_search import BeamSearch
            # self.optimizer = BeamSearch(self.config_space, self.profiler, self.efficiency_constraint)
            raise NotImplementedError
        elif optimizer == "bayesian_optimization":
            # from autotailor.tailor.ofa.optimizer.bayesian_optimization import BayesianOptimization
            # self.optimizer = BayesianOptimization(self.config_space, self.profiler, self.efficiency_constraint)
            raise NotImplementedError
        elif optimizer == "evolution_finder":
            from autotailor.tailor.optimizer.evolution_finder import EvolutionFinder
            self.optimizer = EvolutionFinder(self.config_space, self.profiler, self.efficiency_constraint)
        elif optimizer == "beam_evolution":
            from autotailor.tailor.optimizer.beam_evolution import BeamEvolution
            self.optimizer = BeamEvolution(self.config_space, self.profiler, self.efficiency_constraint)

    # Flatten the config space: {"config_key": [value1, value2, ...]}
    def _get_config_space(self):
        supercode = self.tailor.supercode
        config_space = {}
        # glob dimensions
        for k, v in self.tailor.global_vars.items():
            config_space[k] = v
        
        # stage dimensions
        for k, v in self.tailor.stage_vars.items():
            if "skipped" in k:
                continue
            skip_v = []
            if k in self.tailor.skipcode:
                num_stage = len(supercode[k])
                for stage_i in self.tailor.skipcode[k]:
                    if stage_i < 0:
                        # handle tail order
                        skip_v.append(num_stage+stage_i)
                    else:
                        skip_v.append(stage_i)
            
            for v_i in range(len(supercode[k])):
                if v_i not in skip_v:
                    config_space[f"{k}-{v_i}"] = v
                else:
                    config_space[f"{k}-{v_i}"] = []

        # block dimensions
        for block_type in self.tailor.block_vars:
            for k, v in self.tailor.block_vars[block_type].items():
                for v_i in range(len(supercode[block_type][k])):
                    for v_j in range(len(supercode[block_type][k][v_i])):
                        if supercode[block_type][k][v_i][v_j]:
                            config_space[f"{block_type}-{k}-{v_i}-{v_j}"] = v
                        else:
                            config_space[f"{block_type}-{k}-{v_i}-{v_j}"] = []
                
        return config_space
    

    # Unflatten the config to code, appliable to the tailor
    def _config_to_code(self, config):
        code = copy.deepcopy(self.tailor.supercode)
        
        for k, v in config.items():
            splitted_k = k.split("-")
            num_unit = len(splitted_k)
            if num_unit == 1:
                # global dimension
                code[k] = v
            elif num_unit == 2:
                # stage dimension
                dim_name = splitted_k[0]
                stage_i = int(splitted_k[1])
                if v:
                    code[dim_name][stage_i] = v
            elif num_unit == 4:
                # block dimension
                block_type = splitted_k[0]
                dim_name = splitted_k[1]
                stage_i = int(splitted_k[2])
                block_i = int(splitted_k[3])
                if v:
                    code[block_type][dim_name][stage_i][block_i] = v
            else:
                raise NotImplementedError(f"Not support dimension {k}")
        return code
    
    # Helper function for optimizer
    # Profile the accuracy and latency of a given config
    def _profiler_factory(self):
        def profiler(config):
            # print(config)
            code = self._config_to_code(config)
            # print(code)
            self.tailor.transform(code)
            if isinstance(self.efficiency_predictor, FLOPsCounter):
                efficiency = self.efficiency_predictor.predict_efficiency(code)
            elif isinstance(self.efficiency_predictor, BlockLUTPredictor):
                print(code)
                block_dict = {}
                for stage_id, stage in self.tailor.tir.stages.items():
                    for block_id, block in stage.flow.items():
                        if block.is_active():
                            block_type = block.name
                            block_feature = block.info
                            if block_type not in block_dict:
                                block_dict[block_type] = [block_feature]
                            else:
                                block_dict[block_type].append(block_feature)
                efficiency = self.efficiency_predictor.predict_efficiency(block_dict)
            else:
                op_dict = self.tailor.tir.get_ops(drop_dup=False)
                efficiency = self.efficiency_predictor.predict_efficiency(op_dict)
                # print(efficiency)
            accuracy = self.accuracy_predictor.predict_accuracy(code)
            return accuracy, efficiency
        return profiler
    
    # Return (accuracy, efficiency) of the best architecture
    def optimize(self):
        best_config, best_accuracy, best_efficiency, acc_list = self.optimizer.optimize()
        best_code = self._config_to_code(best_config)
        return best_code, best_accuracy, best_efficiency, acc_list