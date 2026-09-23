"""Collect sensitivity labels exclusively from audited ImageNet training images.

The evaluator configuration's train/val folders are two *training-image views*:
train for BN calibration, val for sensitivity labels. Symlink targets are checked
against --imagenet-train-root before any network evaluation. Official validation
images are never a permitted source for this collector.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AUTOTAILOR_HOME", str(ROOT))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from autotailor.tailor.acc_predictors.sensitivity_provenance import (
    SCHEMA, audit_training_sources, canonical_hash, file_hash,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("supernet_config_path", type=Path)
    parser.add_argument("save_path", type=Path)
    parser.add_argument("--evaluator_config_path", type=Path, required=True,
                        help="Config whose valdir contains disjoint train-image train/val views")
    parser.add_argument("--imagenet-train-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Trained AutoTailor shared-weight checkpoint")
    parser.add_argument("-n", "--net", choices=["ofa_supernet_mbv3_w10", "ofa_supernet_mbv3_w12", "ofa_supernet_proxyless", "ofa_supernet_resnet50"])
    parser.add_argument("-g", "--gpu", default="0", help="Physical GPU made visible as cuda:0")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-r", "--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe-limit", type=int, default=0, help="Diagnostic only; 0 collects every probe")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.probe_limit < 0:
        parser.error("Invalid batch size, worker count, or probe limit")
    if args.checkpoint and args.net:
        parser.error("Use either an AutoTailor checkpoint or external OFA weights")
    if args.checkpoint and args.save_path.resolve() == args.checkpoint.resolve():
        parser.error("Output must not overwrite the checkpoint")
    if args.save_path.resolve() in {args.onnx_path.resolve(), args.supernet_config_path.resolve(), args.evaluator_config_path.resolve()}:
        parser.error("Output must not overwrite an input")
    return args


def generate_codes(tailor):
    codes = {}
    # glob dimensions
    for k in tailor.global_vars:
        codes[k] = []
        max_v = max(tailor.global_vars[k])
        for v in tailor.global_vars[k]:
            if v != max_v:
                code = copy.deepcopy(tailor.supercode)
                code[k] = v
                codes[k].append(code)

    # stage dimension
    for k in tailor.stage_vars:
        if "skipped" in k:
            continue
        codes[k] = []
        max_v = max(tailor.stage_vars[k])
        for v in tailor.stage_vars[k]:
            if v != max_v:
                for v_i in range(len(tailor.supercode[k])):
                    # skip some stage
                    skip_v = []
                    if k in tailor.skipcode:
                        num_stage = len(tailor.supercode[k])
                        for stage_i in tailor.skipcode[k]:
                            if stage_i < 0:
                                # handle tail order
                                skip_v.append(num_stage+stage_i)
                            else:
                                skip_v.append(stage_i)
                    if v_i not in skip_v:
                        code = copy.deepcopy(tailor.supercode)
                        code[k][v_i] = v
                        codes[k].append(code)

    # block dimension
    for block_type in tailor.block_vars:
        codes[block_type] = {}
        for k in tailor.block_vars[block_type]:
            codes[block_type][k] = []
            max_v = max(tailor.block_vars[block_type][k])
            for v in tailor.block_vars[block_type][k]:
                if v != max_v:
                    for v_list_i in range(len(tailor.supercode[block_type][k])):
                        for v_i in range(len(tailor.supercode[block_type][k][v_list_i])):
                            code = copy.deepcopy(tailor.supercode)
                            code[block_type][k][v_list_i][v_i] = v
                            codes[block_type][k].append(code)

    return codes


def load_kernel_transform_matrix(ofa_super_model, tir):
    from autotailor.tir.blocks import BottleneckBlock, BottleneckResidualBlock, DepthConvOp
    for state_id, stage in tir.stages.items():
        for block_id, block in stage.flow.items():
            dwop = None
            if isinstance(block, BottleneckBlock):
                for op in block.flow:
                    if isinstance(op, DepthConvOp):
                        dwop = op
                        break
            elif isinstance(block, BottleneckResidualBlock):
                for op in block.main_path:
                    if isinstance(op, DepthConvOp):
                        dwop = op
                        break
            if dwop:
                op_name = dwop.name
                split_op_info = op_name.split("/")
                torch_k = f"{split_op_info[1]}.{split_op_info[2]}.{split_op_info[3]}.{split_op_info[4]}"
                for k, v in ofa_super_model.state_dict().items():
                    if torch_k in k:
                        matrix_k = k.split(".")[-1]
                        if "matrix" in matrix_k:
                            split_ks = matrix_k.split("_")[0].split("to")
                            src_k = split_ks[0]
                            tar_k = split_ks[1]
                            dwop.transform_matrix_dict[f"{src_k}to{tar_k}"] = v.cuda()
                            continue




def iter_probes(codes):
    for dimension, entries in codes.items():
        if isinstance(entries, list):
            for index, code in enumerate(entries):
                yield f"{dimension}/{index}", code
        else:
            for parameter, rows in entries.items():
                for index, code in enumerate(rows):
                    yield f"{dimension}/{parameter}/{index}", code


def nested_values(codes, results, field):
    output = {}
    for dimension, entries in codes.items():
        if isinstance(entries, list):
            output[dimension] = [results[f"{dimension}/{i}"][field] for i in range(len(entries))]
        else:
            output[dimension] = {
                parameter: [results[f"{dimension}/{parameter}/{i}"][field] for i in range(len(rows))]
                for parameter, rows in entries.items()
            }
    return output


def write_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import onnx
    import onnx_graphsurgeon as gs
    import toml
    import torch
    from torchvision import datasets, transforms
    from autotailor.tir.tailor_ir import TailorIR
    from autotailor.tir.graph_utils import shape_inference
    from autotailor.tailor.tailor import Tailor
    import autotailor.tir.globvar as globvar
    from autotailor.tailor.acc_predictors.sensitivity_collection import calibrate_batch_norm, evaluate_top1

    if args.save_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.save_path}; use a new output or --resume")
    config = json.loads(args.evaluator_config_path.read_text())
    view = Path(config["valdir"])
    calibration_root, sensitivity_root = view / "train", view / "val"
    calibration = datasets.ImageFolder(str(calibration_root))
    sensitivity = datasets.ImageFolder(str(sensitivity_root))
    if calibration.classes != sensitivity.classes or len(sensitivity.classes) != 1000:
        raise ValueError("Both partitions must use the same 1,000 ImageNet class indices")
    provenance = audit_training_sources(
        args.imagenet_train_root, [p for p, _ in calibration.samples],
        [p for p, _ in sensitivity.samples],
    )
    print(f"Audited {len(calibration)} BN images and {len(sensitivity)} sensitivity images, all from train", flush=True)
    inputs = {key: {"path": str(path.resolve()), "sha256": file_hash(path)} for key, path in [
        ("onnx", args.onnx_path), ("config", args.supernet_config_path),
        ("evaluator_config", args.evaluator_config_path),
    ]}
    if args.checkpoint:
        inputs["checkpoint"] = {"path": str(args.checkpoint.resolve()), "sha256": file_hash(args.checkpoint)}
    protocol = {"seed": args.seed, "batch_size": args.batch_size, "workers": args.workers,
                "torch_version": str(torch.__version__),
                "bn_source": "disjoint ImageNet train", "accuracy_source": "ImageNet train",
                "image_count": len(sensitivity), "external_ofa_net": args.net,
                "initial_resolution": args.resolution,
                "collector_sha256": file_hash(Path(__file__)),
                "collection_helper_sha256": file_hash(ROOT / "autotailor/tailor/acc_predictors/sensitivity_collection.py"),
                "provenance_helper_sha256": file_hash(ROOT / "autotailor/tailor/acc_predictors/sensitivity_provenance.py")}
    globvar.super_weights = {}
    onnx_model = onnx.load(str(args.onnx_path))
    globvar.shape_dict = shape_inference(onnx_model, input_shape=(1, 3, args.resolution, args.resolution))
    globvar.onnx_graph = gs.import_onnx(onnx_model)
    globvar.onnx_graph.fold_constants()
    globvar.onnx_graph.cleanup().toposort()
    tir = TailorIR(toml.load(args.supernet_config_path))
    tir.parse_graph(inp_shape=(1, 3, args.resolution, args.resolution))
    tailor = Tailor(tir)
    if args.checkpoint:
        from autotailor.tailor.acc_predictors.sensitivity_checkpoint import load_shared_checkpoint
        protocol["checkpoint_load"] = load_shared_checkpoint(args.checkpoint, globvar.super_weights)
        protocol["checkpoint_loader_sha256"] = file_hash(ROOT / "autotailor/tailor/acc_predictors/sensitivity_checkpoint.py")
        tir.bind_weight()
        print(f"Loaded trained supernet: {protocol['checkpoint_load']}", flush=True)
    if args.net:
        # Preserve the existing OFA kernel-transform loading path, recording its weights.
        ofa_model = torch.hub.load("mit-han-lab/once-for-all", args.net, pretrained=True)
        import hashlib
        digest = hashlib.sha256()
        for name, tensor in sorted(ofa_model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        inputs["ofa_state_dict_sha256"] = digest.hexdigest()
        load_kernel_transform_matrix(ofa_model, tailor.tir)
    codes = generate_codes(tailor)
    probes = [("base", copy.deepcopy(tailor.supercode)), *iter_probes(codes)]
    resume_id = canonical_hash([inputs, provenance, protocol, probes])
    if args.save_path.exists():
        result = json.loads(args.save_path.read_text())
        if result.get("resume_fingerprint") != resume_id:
            raise ValueError("Cannot resume: weights, images, probe codes, or collection protocol changed")
    else:
        result = {"schema_version": SCHEMA, "status": "running", "inputs": inputs,
                  "data_provenance": provenance, "protocol": protocol,
                  "resume_fingerprint": resume_id, "code": codes,
                  "base_code": copy.deepcopy(tailor.supercode), "probe_results": {}}
    results = result["probe_results"]
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    evaluated = 0
    for name, code in probes:
        if name in results:
            if results[name]["code_sha256"] != canonical_hash(code):
                raise ValueError("Resume probe architecture mismatch")
            continue
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        start = time.perf_counter()
        tailor.transform(code)
        model = copy.deepcopy(tailor.tir.build()).cuda().eval()
        flops, params = tailor.tir.count_flops_params()
        resolution = int(code["resolution"])
        calibration.transform = transforms.Compose([
            transforms.RandomResizedCrop(resolution), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        calibration_loader = torch.utils.data.DataLoader(
            calibration, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        calibrated = calibrate_batch_norm(model, calibration_loader, "cuda:0")
        if calibrated not in (0, len(calibration)):
            raise RuntimeError("Incomplete BN calibration")
        sensitivity.transform = transforms.Compose([
            transforms.Resize(math.ceil(resolution / 0.875)), transforms.CenterCrop(resolution),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        accuracy_loader = torch.utils.data.DataLoader(
            sensitivity, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        )
        accuracy = evaluate_top1(model, accuracy_loader, "cuda:0")
        if accuracy["image_count"] != len(sensitivity):
            raise RuntimeError("Incomplete sensitivity-image evaluation")
        results[name] = {**accuracy, "bn_calibration_image_count": calibrated,
                         "flops": float(flops), "params": int(params),
                         "time": time.perf_counter() - start, "code_sha256": canonical_hash(code)}
        write_result(args.save_path, result)
        print(f"{name}: {results[name]['accuracy']:.4f}% ({len(results)}/{len(probes)})", flush=True)
        del model, calibration_loader, accuracy_loader
        torch.cuda.empty_cache()
        evaluated += 1
        if args.probe_limit and evaluated >= args.probe_limit and len(results) < len(probes):
            result["status"] = "partial_diagnostic"
            write_result(args.save_path, result)
            return
    result["base_acc"] = [results["base"]["accuracy"]]
    for field in ["accuracy", "flops", "params", "time"]:
        result["acc" if field == "accuracy" else field] = nested_values(codes, results, field)
    result["status"] = "complete"
    write_result(args.save_path, result)
    print(f"Wrote {len(results)} training-only sensitivity measurements to {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
