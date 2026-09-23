import argparse
import json
import os
import sys
from pathlib import Path

AUTOTAILOR_HOME = os.environ.get("AUTOTAILOR_HOME", str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, AUTOTAILOR_HOME)


from autotailor.tailor.acc_predictors.evaluation_data import heldout_indices



def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("onnx_path", type=str)
    arg_parser.add_argument("supernet_config_path", type=str)
    arg_parser.add_argument("accuracy_metric", type=str)
    arg_parser.add_argument("dataset_path", type=str)
    arg_parser.add_argument("--group_size", type=int, default=200)
    arg_parser.add_argument("--group_num", type=int, default=100)
    arg_parser.add_argument("--weight_path", type=str, default=None)
    arg_parser.add_argument("--log_path", type=str, default=None)
    arg_parser.add_argument("--resolution", type=int, default=224)
    
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--device", default="cuda:0")
    args = arg_parser.parse_args()
    if args.accuracy_metric not in {"flops", "sensitivity", "mlp"}:
        arg_parser.error("accuracy_metric must be flops, sensitivity, or mlp")
    if args.accuracy_metric in {"mlp", "sensitivity"} and not args.weight_path:
        arg_parser.error("Predictor evaluation requires --weight_path")
    if args.group_size < 1 or args.group_num < 1:
        arg_parser.error("--group_size and --group_num must be positive")
    return args


def main():
    args = parse_args()
    import toml
    import numpy as np
    from loguru import logger

    import onnx
    import onnx_graphsurgeon as gs
    from autotailor.tir.tailor_ir import TailorIR
    from autotailor.tailor.tailor import Tailor
    from autotailor.tir.graph_utils import shape_inference
    import autotailor.tir.globvar as globvar
    
    if args.log_path:
        logger.add(args.log_path)
    
    onnx_path = args.onnx_path
    supernet_config_dict = toml.load(args.supernet_config_path)
    sensitivity_weight_path = args.weight_path

    onnx_model = onnx.load(onnx_path)
    globvar.shape_dict = shape_inference(onnx_model)
    globvar.onnx_graph = gs.import_onnx(onnx_model)
    globvar.onnx_graph.fold_constants()
    globvar.onnx_graph.cleanup().toposort()

    tir = TailorIR(supernet_config_dict)
    tir.parse_graph(inp_shape=(1,3,args.resolution, args.resolution))

    tailor = Tailor(tir)
    accuracy_metric = args.accuracy_metric
    if accuracy_metric == "flops":
        from autotailor.tailor.acc_predictors.flops_counter import FLOPsCounter as FLOPsAccPredictor
        accuracy_predictor = FLOPsAccPredictor(tailor)
    elif accuracy_metric == "sensitivity":
        from autotailor.tailor.acc_predictors.sensitivity_provenance import require_training_sensitivity
        with open(sensitivity_weight_path) as stream:
            require_training_sensitivity(json.load(stream))
        from autotailor.tailor.acc_predictors.sensitivity_estimator import SensitivityEstimator
        accuracy_predictor = SensitivityEstimator(tailor, sensitivity_weight_path)
    elif accuracy_metric == "mlp":
        from autotailor.tailor.acc_predictors.arch_encoder import ArchEncoder
        from autotailor.tailor.acc_predictors.mlp_predictor import MLPPredictor
        arch_encoder = ArchEncoder(supernet_config_dict["var"], tailor.supercode)
        accuracy_predictor = MLPPredictor(arch_encoder, checkpoint_path=args.weight_path, device=args.device)

    # Load dataset
    dataset = json.load(open(args.dataset_path))
    codes = dataset["code"]
    accs = dataset["acc"]
    if args.accuracy_metric == "mlp":
        indices = heldout_indices(arch_encoder, codes, accs, accuracy_predictor.split_metadata)
        print(f"Held-out architectures: {len(indices)}; excluded training rows: {len(codes) - len(indices)}")
        codes, accs = [codes[i] for i in indices], [accs[i] for i in indices]
    if not codes or len(codes) != len(accs):
        raise ValueError("Expected nonempty, equally sized architecture and accuracy lists")
    if args.group_size > len(codes):
        raise ValueError(f"--group_size exceeds the {len(codes)} eligible architectures")
    rng = np.random.default_rng(args.seed)
    
    repeat_num = 10
    rmse_lst = []
    avg_gap_ratio_lst = []
    top1_lst = []
    top5_lst = []
    for i in range(repeat_num):
        predictions = []
        gap = []
        gap_ratio = []
        for i, code in enumerate(codes):
            predict_acc = accuracy_predictor.predict_accuracy(code)
            predictions.append(predict_acc)
            if args.accuracy_metric != "flops":
                gap.append(abs(predict_acc - accs[i]))
                gap_ratio.append(gap[-1] / accs[i])
        
        # print(predictions)
        # count rmse, grouped top1, top5
        if args.accuracy_metric != "flops":
            sum_gap = 0
            for v in gap:
                sum_gap += v*v
            rmse = (sum_gap/len(gap))**0.5
            avg_gap_ratio = sum(gap_ratio)/len(gap_ratio)
            rmse_lst.append(rmse)
            avg_gap_ratio_lst.append(avg_gap_ratio)

        group_num = args.group_num
        group_size = args.group_size
        group_datasets = []
        # random grouping
        for i in range(group_num):
            indexes = list(range(len(codes)))
            group_idxs = rng.choice(indexes, size=group_size, replace=False)
            group_datasets.append(list(group_idxs))
        top1 = 0
        top5 = 0
        for group in group_datasets:
            group_accs = [accs[i] for i in group]
            group_preds = [predictions[i] for i in group]
            best_acc_idx = np.argmax(np.array(group_accs))
            best_pred_idx = np.argmax(np.array(group_preds))
            if best_acc_idx == best_pred_idx:
                top1 += 1
            if best_acc_idx in np.argsort(np.array(group_preds))[-5:]:
                top5 += 1
        top1_lst.append(top1)
        top5_lst.append(top5)
    
    rmse = sum(rmse_lst)/len(rmse_lst) if rmse_lst else None
    avg_gap_ratio = sum(avg_gap_ratio_lst)/len(avg_gap_ratio_lst) if avg_gap_ratio_lst else None
    top1 = sum(top1_lst)/len(top1_lst)
    top5 = sum(top5_lst)/len(top5_lst)
    if args.accuracy_metric != "flops":
        print(f"RMSE: {rmse}")
        print(f"Avg gap ratio: {avg_gap_ratio}")
    print(f"Grouped Top1: {top1/group_num}, Top5: {top5/group_num}")
    

if __name__ == "__main__":
    main()
