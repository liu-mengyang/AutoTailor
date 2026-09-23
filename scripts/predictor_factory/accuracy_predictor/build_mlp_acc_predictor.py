import argparse
import copy
import json
import math
import os
import os.path as osp
import sys
from pathlib import Path
from pprint import pprint

AUTOTAILOR_HOME = os.environ.get("AUTOTAILOR_HOME", str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, AUTOTAILOR_HOME)

import torch


from autotailor.tailor.acc_predictors.arch_encoder import ArchEncoder
from autotailor.tailor.acc_predictors.mlp_trainer import build_acc_data_loader, AccPredictorTrainer, RegDataset
from autotailor.tailor.acc_predictors.mlp_predictor import MLPPredictor
from autotailor.tailor.acc_predictors.evaluation_data import heldout_indices

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("onnx_path", type=str)
    arg_parser.add_argument("supernet_config_path", type=str)
    arg_parser.add_argument("dataset_path", type=str)
    arg_parser.add_argument("save_name", type=str)
    arg_parser.add_argument("-r", "--resolution", type=int, default=224)
    arg_parser.add_argument("-g", type=int, default=0)
    arg_parser.add_argument("-ckpt", type=str, default=None)
    arg_parser.add_argument("-e", action="store_true", default=False)
    arg_parser.add_argument("-no_split", action="store_true", default=False)
    arg_parser.add_argument("-num_s", type=int, default=None)
    
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--epochs", type=int, default=200)
    arg_parser.add_argument("--workers", type=int, default=16)
    args = arg_parser.parse_args()
    if args.e and (not args.ckpt or args.no_split):
        arg_parser.error("-e requires -ckpt and cannot be combined with -no_split")
    if args.ckpt and not args.e:
        arg_parser.error("Warm-start training has no safe split contract; train afresh or use -e")
    if args.num_s is not None and args.num_s < 1:
        arg_parser.error("-num_s must be positive")
    if args.epochs < 1 or args.workers < 0:
        arg_parser.error("--epochs must be positive and --workers nonnegative")
    return args


def main():
    args = parse_args()
    import toml

    import onnx
    import onnx_graphsurgeon as gs
    from autotailor.tir.tailor_ir import TailorIR
    from autotailor.tailor.tailor import Tailor
    from autotailor.tir.graph_utils import shape_inference
    import autotailor.tir.globvar as globvar

    onnx_path = args.onnx_path
    supernet_config_dict = toml.load(args.supernet_config_path)
    dataset = json.load(open(args.dataset_path))
    # print(len(dataset["code"]))
    # print(len(dataset["acc"]))
    # assert len(dataset["code"]) == len(dataset["acc"])
    
    onnx_model = onnx.load(onnx_path)
    globvar.shape_dict = shape_inference(onnx_model)
    globvar.onnx_graph = gs.import_onnx(onnx_model)
    globvar.onnx_graph.fold_constants()
    globvar.onnx_graph.cleanup().toposort()

    tir = TailorIR(supernet_config_dict)
    tir.parse_graph(inp_shape=(1,3,args.resolution,args.resolution))

    tailor = Tailor(tir)
    
    arch_encoder = ArchEncoder(supernet_config_dict["var"], tailor.supercode)
    codes = dataset["code"][:args.num_s]
    accs = dataset["acc"][:args.num_s]
    if "time" in dataset:
        times = dataset["time"][:args.num_s]
        print(f"Data collection for {len(codes)} subnets costs {sum(times)/3600} hours")
    device = f"cuda:{args.g}"
    torch.manual_seed(args.seed)
    predictor = MLPPredictor(arch_encoder, checkpoint_path=args.ckpt, device=device)
    if args.e:
        indices = heldout_indices(
            arch_encoder, codes, accs, predictor.split_metadata, saved_split=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            RegDataset(
                torch.tensor([arch_encoder.encode(codes[i]) for i in indices], dtype=torch.float32),
                torch.tensor([accs[i] / 100.0 for i in indices], dtype=torch.float32),
            ),
            batch_size=256, shuffle=False, num_workers=args.workers,
        )
        trainer = AccPredictorTrainer(
            predictor, None, valid_loader, args.save_name, args.epochs, device=device,
        )
        print(json.dumps(trainer.evaluate(), sort_keys=True))
    else:
        train_loader, valid_loader, base_acc = build_acc_data_loader(
            arch_encoder, codes, accs,
            n_training_sample=len(codes) if args.no_split else None,
            seed=args.seed, n_workers=args.workers,
        )
        trainer = AccPredictorTrainer(
            predictor, train_loader, valid_loader, args.save_name, args.epochs, device=device,
        )
        trainer.train(base_acc)
        if valid_loader is not None:
            print(json.dumps(trainer.evaluate(), sort_keys=True))
        else:
            print("Trained on all architectures; no held-out metric is available")
        trainer.save()

if __name__ == "__main__":
    main()
