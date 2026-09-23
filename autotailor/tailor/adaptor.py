import os
import sys

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)

from autotailor.tailor.optimizer_framework import OptimizerFramework


class Adaptor(object):
    def __init__(self, tailor):
        self.tailor = tailor
    
    def adapt(self,
              efficiency_metric="flops",
              efficiency_constraint=600000,
              data_dict=None,
              mode="beam_evolution",
              accuracy_metric="flops",
              trans_weight_path=None,
              arch_encoder=None,
              mlp_model=None,
              lat_mlp_model=None
              ):
        efficiency_predictor = None
        accuracy_predictor = None
        if efficiency_metric == "flops":
            from autotailor.tailor.eff_predictors.flops_counter import FLOPsCounter as FLOPsEffPredictor
            efficiency_predictor = FLOPsEffPredictor(self.tailor)
        elif efficiency_metric == "lut":
            from autotailor.tailor.eff_predictors.lut_predictor import LUTPredictor
            efficiency_predictor = LUTPredictor(data_dict)
        elif efficiency_metric == "block":
            from autotailor.tailor.eff_predictors.block_lut_predictor import BlockLUTPredictor
            efficiency_predictor = BlockLUTPredictor(data_dict)
        elif efficiency_metric == "mlp":
            from autotailor.tailor.eff_predictors.mlp_predictor import MLPPredictor
            efficiency_predictor = MLPPredictor(data_dict)
        elif efficiency_metric == "ofamlp":
            from autotailor.tailor.acc_predictors.mlp_estimator import MLPEstimator
            efficiency_predictor = MLPEstimator(arch_encoder, lat_mlp_model, device="cuda:0")
        else:
            raise KeyError(f"{efficiency_metric} is not an supported efficiency metric")
            
        if accuracy_metric == "flops":
            from autotailor.tailor.acc_predictors.flops_counter import FLOPsCounter as FLOPsAccPredictor
            accuracy_predictor = FLOPsAccPredictor(self.tailor)
        elif accuracy_metric == "sensitivity":
            import json
            from autotailor.tailor.acc_predictors.sensitivity_provenance import require_training_sensitivity
            with open(trans_weight_path) as stream:
                require_training_sensitivity(json.load(stream))
            from autotailor.tailor.acc_predictors.sensitivity_estimator import SensitivityEstimator
            accuracy_predictor = SensitivityEstimator(self.tailor, trans_weight_path)
        elif accuracy_metric == "mlp":
            from autotailor.tailor.acc_predictors.mlp_estimator import MLPEstimator
            accuracy_predictor = MLPEstimator(arch_encoder, mlp_model, device="cuda:0")
        else:
            raise KeyError(f"{accuracy_metric} is not an supported accuracy metric")
        
        optimizer = OptimizerFramework(self.tailor,
                                       efficiency_constraint,
                                       efficiency_predictor,
                                       accuracy_predictor,
                                       optimizer=mode)
        
        best_code, best_acc, best_eff, acc_list = optimizer.optimize()
        return best_code, best_acc, best_eff, acc_list
