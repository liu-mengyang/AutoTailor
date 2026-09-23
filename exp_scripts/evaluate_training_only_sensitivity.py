#!/usr/bin/env python3
"""Evaluate frozen sensitivity selections with the collection's exact BN protocol.

Model, train-image partitions, batch size, workers and execution seed come from
the completed sensitivity artifact. The final labels come only from official
ImageNet validation. No selection or sensitivity fitting occurs in this script.
"""

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AUTOTAILOR_HOME", str(ROOT))

from autotailor.tailor.acc_predictors.sensitivity_provenance import (
    audit_training_sources, canonical_hash, file_hash, require_training_sensitivity,
)
from scripts.predictor_factory.accuracy_predictor.collect_acc_sensitivity import write_result


def validation_records(root, classes, expected_count=50000):
    labels_path = root / "LOC_val_solution.csv"
    image_root = (root / "ILSVRC/Data/CLS-LOC/val").resolve(strict=True)
    if image_root.name != "val" or image_root.parent.name != "CLS-LOC":
        raise ValueError("Expected original CLS-LOC/val image directory")
    mapped = [line.split()[0] for line in (root / "LOC_synset_mapping.txt").read_text().splitlines() if line.strip()]
    if mapped != classes:
        raise ValueError("Official class order differs from the collection's training ImageFolder")
    class_ids = {name: i for i, name in enumerate(classes)}
    records, seen = [], set()
    with labels_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["ImageId", "PredictionString"]:
            raise ValueError("Unexpected official validation label columns")
        for row in reader:
            image_id = row["ImageId"]
            tokens = row["PredictionString"].split()
            if (image_id in seen or not image_id.startswith("ILSVRC2012_val_")
                    or not tokens or len(tokens) % 5
                    or any(label != tokens[0] for label in tokens[0::5])):
                raise ValueError("Invalid or duplicate official validation record")
            path = (image_root / f"{image_id}.JPEG").resolve(strict=True)
            if not path.is_file() or not path.is_relative_to(image_root):
                raise ValueError("Validation image escapes the original validation split")
            records.append((str(path), class_ids[tokens[0]]))
            seen.add(image_id)
    if len(records) != expected_count:
        raise ValueError(f"Expected all {expected_count} validation images, got {len(records)}")
    return sorted(records), image_root


class ValidationDataset:
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from torchvision.datasets.folder import default_loader

        path, target = self.records[index]
        return self.transform(default_loader(path)), target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--imagenet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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

    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    sensitivity = json.loads(args.sensitivity.read_text())
    provenance = require_training_sensitivity(sensitivity)
    selection = json.loads(args.selection.read_text())
    if selection["inputs"]["sensitivity"]["sha256"] != file_hash(args.sensitivity):
        raise ValueError("Selections were not frozen using these sensitivity weights")
    protocol = sensitivity["protocol"]
    if protocol.get("external_ofa_net"):
        raise ValueError("This evaluator supports ONNX-only models; external OFA matrices need a matched evaluator")
    for name in ["onnx", "config", "evaluator_config", *(["checkpoint"] if "checkpoint" in sensitivity["inputs"] else [])]:
        record = sensitivity["inputs"][name]
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError(f"Collection input {name} changed before final evaluation")
    if protocol["collection_helper_sha256"] != file_hash(ROOT / "autotailor/tailor/acc_predictors/sensitivity_collection.py"):
        raise ValueError("BN/evaluation helper differs from collection")
    if protocol["torch_version"] != str(torch.__version__):
        raise ValueError("PyTorch version differs from collection")
    view = Path(json.loads(Path(sensitivity["inputs"]["evaluator_config"]["path"]).read_text())["valdir"])
    calibration = datasets.ImageFolder(str(view / "train"))
    label_partition = datasets.ImageFolder(str(view / "val"))
    live_audit = audit_training_sources(provenance["original_training_root"],
                                       [path for path, _ in calibration.samples],
                                       [path for path, _ in label_partition.samples])
    if live_audit != provenance or calibration.classes != label_partition.classes:
        raise ValueError("Collection image partitions changed before final evaluation")
    records, validation_root = validation_records(args.imagenet_root, calibration.classes)
    if set(path for path, _ in records) & set(provenance["sensitivity"]["source_paths"]):
        raise ValueError("Sensitivity and final evaluation images overlap")
    entries = selection["selected_candidates"]
    hashes = [canonical_hash(entry["code"]) for entry in entries]
    if not entries or len(set(hashes)) != len(hashes) or any(
        digest != entry["code_sha256"] for digest, entry in zip(hashes, entries)
    ):
        raise ValueError("Invalid or duplicate selected architectures")
    contract = {"sensitivity_sha256": file_hash(args.sensitivity),
                "selection_sha256": file_hash(args.selection),
                "evaluator_sha256": file_hash(Path(__file__)),
                "validation_labels_sha256": file_hash(args.imagenet_root / "LOC_val_solution.csv"),
                "validation_records_sha256": canonical_hash(records),
                "validation_root": str(validation_root), "collection_protocol": protocol}
    fingerprint = canonical_hash(contract)
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result.get("resume_fingerprint") != fingerprint:
            raise ValueError("Final evaluation resume inputs/protocol changed")
    else:
        result = {"schema_version": "training-only-sensitivity-final-evaluation-v1", "status": "running",
                  "contract": contract, "resume_fingerprint": fingerprint,
                  "model_inputs": sensitivity["inputs"], "data_provenance": provenance,
                  "validation_image_count": len(records), "subnets": [],
                  "scope": "selected architectures only; no inference about full-pool ranking"}
    completed = {row["code_sha256"] for row in result["subnets"]}
    if not completed <= set(hashes) or len(completed) != len(result["subnets"]):
        raise ValueError("Invalid completed architectures in resume artifact")
    model_path = Path(sensitivity["inputs"]["onnx"]["path"])
    config_path = Path(sensitivity["inputs"]["config"]["path"])
    globvar.super_weights = {}
    onnx_model = onnx.load(str(model_path))
    resolution = protocol["initial_resolution"]
    globvar.shape_dict = shape_inference(onnx_model, input_shape=(1, 3, resolution, resolution))
    globvar.onnx_graph = gs.import_onnx(onnx_model)
    globvar.onnx_graph.fold_constants()
    globvar.onnx_graph.cleanup().toposort()
    tir = TailorIR(toml.load(config_path))
    tir.parse_graph(inp_shape=(1, 3, resolution, resolution))
    tailor = Tailor(tir)
    if "checkpoint" in sensitivity["inputs"]:
        from autotailor.tailor.acc_predictors.sensitivity_checkpoint import load_shared_checkpoint
        if protocol.get("checkpoint_loader_sha256") != file_hash(ROOT / "autotailor/tailor/acc_predictors/sensitivity_checkpoint.py"):
            raise ValueError("Checkpoint loader differs from collection")
        loaded = load_shared_checkpoint(sensitivity["inputs"]["checkpoint"]["path"], globvar.super_weights)
        if loaded != protocol.get("checkpoint_load"):
            raise ValueError("Checkpoint metadata differs from collection")
        tir.bind_weight()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    for entry in entries:
        if entry["code_sha256"] in completed:
            continue
        seed = protocol["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        code = entry["code"]
        tailor.transform(code)
        model = copy.deepcopy(tailor.tir.build()).cuda().eval()
        resolution = int(code["resolution"])
        calibration.transform = transforms.Compose([
            transforms.RandomResizedCrop(resolution), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        bn_loader = torch.utils.data.DataLoader(calibration, batch_size=protocol["batch_size"],
            shuffle=False, num_workers=protocol["workers"], generator=torch.Generator().manual_seed(seed))
        calibrated = calibrate_batch_norm(model, bn_loader, "cuda:0")
        if calibrated not in (0, len(calibration)):
            raise ValueError("Incomplete BN calibration")
        transform = transforms.Compose([
            transforms.Resize(math.ceil(resolution / 0.875)), transforms.CenterCrop(resolution),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        loader = torch.utils.data.DataLoader(ValidationDataset(records, transform), batch_size=protocol["batch_size"],
                                             shuffle=False, num_workers=protocol["workers"])
        accuracy = evaluate_top1(model, loader, "cuda:0")
        if accuracy["image_count"] != 50000:
            raise ValueError("Incomplete final validation")
        result["subnets"].append({"candidate_id": entry["candidate_id"], "code": code,
                                  "code_sha256": entry["code_sha256"], "accuracy": accuracy,
                                  "bn_calibration_images": calibrated, "seconds": time.perf_counter() - started})
        write_result(args.output, result)
        print(f"{entry['candidate_id']}: {accuracy['accuracy']:.3f}% ({len(result['subnets'])}/{len(entries)})", flush=True)
        del model, bn_loader, loader
        torch.cuda.empty_cache()
    result["status"] = "complete"
    write_result(args.output, result)


if __name__ == "__main__":
    main()
