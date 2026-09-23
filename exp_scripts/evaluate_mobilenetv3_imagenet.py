#!/usr/bin/env python3
"""Evaluate transformed OFA-MobileNetV3 subnets on ImageNet validation.

The experiment intentionally keeps batch-normalization calibration and final
accuracy measurement on different ImageNet source splits.  BN statistics use
only the ``bn_calibration`` rows in the deterministic training-view manifest;
accuracy uses the official ILSVRC validation images and Kaggle ground truth.

The TailorIR network is initialized from the repository ONNX model.  OFA's
official pretrained supernet is loaded only to copy the learned kernel
transformation matrices that are not represented in ONNX initializers.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("AUTOTAILOR_HOME", str(REPO_ROOT))

import autotailor.tir.globvar as globvar
from autotailor.tailor.tailor import Tailor
from autotailor.tir.blocks import BottleneckBlock, BottleneckResidualBlock, DepthConvOp
from autotailor.tir.graph_utils import shape_inference
from autotailor.tir.tailor_ir import TailorIR


SCHEMA_VERSION = "mobilenetv3-imagenet-accuracy-v1"
DEFAULT_SEED = 20260902
DEFAULT_IMAGENET_ROOT = Path(
    "/datasets/imagenet"
)
OFA_LOADER_CHECKPOINT = (
    REPO_ROOT / ".torch/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0"
)
SUBNET_NAMES = ("max", "min", "seeded_random", "seeded_compound")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ImageFile.LOAD_TRUNCATED_IMAGES = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=REPO_ROOT / "models/onnx/ofa_supernet_mbv3_w10.onnx",
    )
    parser.add_argument(
        "--supernet-config",
        type=Path,
        default=REPO_ROOT / "configs/supernet/ofa_mobilenetv3w10.toml",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=REPO_ROOT / "exp/rebuttal/imagenet_train10k_view/train",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=REPO_ROOT / "exp/rebuttal/imagenet_train10k_view/manifest.csv",
    )
    parser.add_argument(
        "--calibration-metadata",
        type=Path,
        default=REPO_ROOT / "exp/rebuttal/imagenet_train10k_view/metadata.json",
    )
    parser.add_argument(
        "--validation-images",
        type=Path,
        default=DEFAULT_IMAGENET_ROOT / "ILSVRC/Data/CLS-LOC/val",
    )
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=DEFAULT_IMAGENET_ROOT / "LOC_val_solution.csv",
    )
    parser.add_argument(
        "--synset-mapping",
        type=Path,
        default=DEFAULT_IMAGENET_ROOT / "LOC_synset_mapping.txt",
    )
    parser.add_argument(
        "--ofa-checkpoint",
        type=Path,
        default=OFA_LOADER_CHECKPOINT,
        help="Checkpoint populated by the official OFA torch-hub loader",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/rebuttal/mobilenetv3_accuracy/results.json",
    )
    parser.add_argument(
        "--codes-json",
        type=Path,
        default=None,
        help=(
            "Optional selection manifest containing selected_candidates entries; "
            "when omitted, evaluate the four built-in audit SubNets"
        ),
    )
    parser.add_argument(
        "--subnets",
        nargs="+",
        default=None,
        help=(
            "Names to evaluate. Defaults to all built-in audit SubNets or all "
            "candidates from --codes-json"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibration-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--val-limit",
        type=int,
        default=0,
        help="Diagnostic-only deterministic validation subset size; 0 uses all 50,000",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.calibration_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.workers < 0 or args.val_limit < 0:
        parser.error("--workers and --val-limit must be non-negative")
    if args.subnets is not None and len(set(args.subnets)) != len(args.subnets):
        parser.error("--subnets must not contain duplicates")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ensure_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class TensorAverage:
    def __init__(self) -> None:
        self.sum: torch.Tensor | int = 0
        self.count = 0

    @property
    def average(self) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError("average requested before any observations")
        return self.sum / self.count

    def update(self, value: torch.Tensor, count: int) -> None:
        self.sum = self.sum + value * count
        self.count += count


def set_running_statistics(model: nn.Module, data_loader: DataLoader, device: torch.device) -> int:
    """OFA-compatible BN recalibration with exact processed-image accounting."""
    means: dict[str, TensorAverage] = {}
    variances: dict[str, TensorAverage] = {}
    forward_model = copy.deepcopy(model).to(device).eval()

    for name, module in forward_model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        means[name] = TensorAverage()
        variances[name] = TensorAverage()

        def replacement(
            bn: nn.BatchNorm2d,
            mean_estimator: TensorAverage,
            variance_estimator: TensorAverage,
        ):
            def forward(inputs: torch.Tensor) -> torch.Tensor:
                batch_mean = inputs.mean(dim=(0, 2, 3))
                batch_variance = (inputs - batch_mean[None, :, None, None]).square().mean(
                    dim=(0, 2, 3)
                )
                mean_estimator.update(batch_mean.detach(), inputs.shape[0])
                variance_estimator.update(batch_variance.detach(), inputs.shape[0])
                feature_count = batch_mean.shape[0]
                return F.batch_norm(
                    inputs,
                    batch_mean,
                    batch_variance,
                    bn.weight[:feature_count],
                    bn.bias[:feature_count],
                    False,
                    0.0,
                    bn.eps,
                )

            return forward

        module.forward = replacement(module, means[name], variances[name])

    processed = 0
    with torch.inference_mode():
        for images, _ in tqdm(data_loader, desc="BN calibration", leave=False):
            processed += images.shape[0]
            forward_model(images.to(device, non_blocking=True))

    for name, module in model.named_modules():
        if name not in means:
            continue
        if means[name].count == 0:
            raise RuntimeError(f"BN layer {name} received no observations")
        feature_count = means[name].average.shape[0]
        if not isinstance(module, nn.BatchNorm2d):
            raise TypeError(f"expected BatchNorm2d at {name}")
        module.running_mean[:feature_count].copy_(means[name].average)
        module.running_var[:feature_count].copy_(variances[name].average)
    return processed


class FlatImageNetValidation(Dataset):
    def __init__(self, records: list[tuple[Path, int]], transform: Any) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.records[index]
        image = default_loader(str(path))
        return self.transform(image), target


def read_synsets(path: Path) -> list[str]:
    synsets = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                synsets.append(stripped.split(maxsplit=1)[0])
    if len(synsets) != 1000 or len(set(synsets)) != 1000:
        raise ValueError(f"expected 1,000 unique synsets in {path}, found {len(synsets)}")
    return synsets


def read_validation_records(
    images_root: Path,
    labels_path: Path,
    synsets: list[str],
) -> tuple[list[tuple[Path, int]], dict[str, Any]]:
    class_index = {synset: index for index, synset in enumerate(synsets)}
    labels: dict[str, int] = {}
    box_annotation_count = 0
    with labels_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ImageId", "PredictionString"]:
            raise ValueError(f"unexpected validation-label fields: {reader.fieldnames}")
        for row in reader:
            image_id = row["ImageId"]
            fields = row["PredictionString"].split()
            if not fields or len(fields) % 5 != 0:
                raise ValueError(f"malformed localization ground truth for {image_id}")
            box_synsets = fields[0::5]
            if any(synset not in class_index for synset in box_synsets):
                raise ValueError(f"invalid ground truth for {image_id}")
            if any(synset != box_synsets[0] for synset in box_synsets):
                raise ValueError(f"multiple class labels found for {image_id}")
            try:
                [float(value) for offset in range(1, 5) for value in fields[offset::5]]
            except ValueError as error:
                raise ValueError(f"invalid box coordinates for {image_id}") from error
            if image_id in labels:
                raise ValueError(f"duplicate validation label row for {image_id}")
            labels[image_id] = class_index[fields[0]]
            box_annotation_count += len(box_synsets)

    records = []
    for image_id in sorted(labels):
        image_path = images_root / f"{image_id}.JPEG"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        records.append((image_path, labels[image_id]))
    if len(records) != 50000:
        raise ValueError(f"expected 50,000 validation rows, found {len(records)}")
    label_audit = {
        "row_count": len(records),
        "box_annotation_count": box_annotation_count,
        "prediction_string_group_size": 5,
        "first_token_is_valid_class_for_every_row": True,
        "all_box_labels_match_first_token_per_row": True,
        "all_rows_have_parseable_box_coordinates": True,
        "image_ids_are_unique": True,
    }
    return records, label_audit


def audit_calibration_partition(
    root: Path,
    manifest_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text())
    rows = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["class", "purpose", "source", "link"]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected calibration manifest fields: {reader.fieldnames}")
        rows = [row for row in reader if row["purpose"] == "bn_calibration"]

    dataset = datasets.ImageFolder(str(root))
    dataset_paths = {str(Path(path).resolve()) for path, _ in dataset.samples}
    manifest_paths = {str(Path(row["source"]).resolve()) for row in rows}
    resolved_links = {str(Path(row["link"]).resolve()) for row in rows}
    if dataset_paths != manifest_paths or resolved_links != manifest_paths:
        raise ValueError("calibration ImageFolder does not exactly match bn_calibration manifest rows")
    if len(rows) != 2000 or len(dataset.classes) != 1000:
        raise ValueError(
            f"expected 2,000 calibration images over 1,000 classes, got "
            f"{len(rows)} over {len(dataset.classes)}"
        )
    if not all("/CLS-LOC/train/" in row["source"] for row in rows):
        raise ValueError("calibration manifest contains a non-training-split source")
    if metadata.get("partitions_are_disjoint") is not True:
        raise ValueError("calibration metadata does not assert disjoint partitions")

    return {
        "image_count": len(rows),
        "class_count": len(dataset.classes),
        "purpose": "bn_calibration",
        "source_split": metadata.get("source_split"),
        "manifest_sha256": sha256_file(manifest_path),
        "metadata_sha256": sha256_file(metadata_path),
        "source_paths_sha256": canonical_sha256(sorted(manifest_paths)),
        "sensitivity_partition_used_for_accuracy": False,
    }


def load_tailor(onnx_path: Path, config_path: Path) -> Tailor:
    globvar.super_weights = {}
    onnx_model = onnx.load(str(onnx_path))
    globvar.shape_dict = shape_inference(onnx_model)
    globvar.onnx_graph = gs.import_onnx(onnx_model)
    globvar.onnx_graph.fold_constants()
    globvar.onnx_graph.cleanup().toposort()
    tir = TailorIR(toml.load(config_path))
    tir.parse_graph()
    return Tailor(tir)


def load_kernel_transform_matrices(tailor: Tailor) -> dict[str, Any]:
    # The official loader uses a repository-relative .torch checkpoint path.
    previous_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        ofa_model = torch.hub.load(
            "mit-han-lab/once-for-all",
            "ofa_supernet_mbv3_w10",
            pretrained=True,
            trust_repo=True,
        )
    finally:
        os.chdir(previous_cwd)

    copied: list[str] = []
    state = ofa_model.state_dict()
    for stage in tailor.tir.stages.values():
        for block in stage.flow.values():
            depthwise = None
            if isinstance(block, BottleneckBlock):
                depthwise = next(
                    (op for op in block.flow if isinstance(op, DepthConvOp)), None
                )
            elif isinstance(block, BottleneckResidualBlock):
                depthwise = next(
                    (op for op in block.main_path if isinstance(op, DepthConvOp)), None
                )
            if depthwise is None:
                continue
            name_fields = depthwise.name.split("/")
            torch_key_fragment = ".".join(name_fields[1:5])
            for key, value in state.items():
                if torch_key_fragment not in key or "matrix" not in key.split(".")[-1]:
                    continue
                matrix_name = key.split(".")[-1].split("_")[0]
                depthwise.transform_matrix_dict[matrix_name] = value.detach().to(
                    depthwise.super_weights.device
                )
                copied.append(key)
    if not copied:
        raise RuntimeError("no OFA kernel transformation matrices matched TailorIR blocks")
    return {
        "matrix_tensor_count": len(copied),
        "matrix_keys_sha256": canonical_sha256(sorted(copied)),
    }


def deterministic_codes(tailor: Tailor, seed: int) -> dict[str, dict[str, Any]]:
    random.seed(seed)
    random_code = tailor.random_sample()
    random.seed(seed + 1)
    compound_code = tailor.compound_sample()
    return {
        "max": copy.deepcopy(tailor.supercode),
        "min": tailor.min_sample(),
        "seeded_random": random_code,
        "seeded_compound": compound_code,
    }


def selected_codes(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    candidates = payload.get("selected_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{path}: selected_candidates must be a non-empty list")
    codes: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if not isinstance(item, dict):
            raise TypeError(f"{path}: every selected candidate must be an object")
        index = item.get("candidate_index")
        code = item.get("code")
        expected_hash = item.get("code_sha256")
        if not isinstance(index, int) or index < 0 or not isinstance(code, dict):
            raise ValueError(f"{path}: malformed selected candidate {item!r}")
        name = f"candidate_{index:04d}"
        if name in codes:
            raise ValueError(f"{path}: duplicate candidate index {index}")
        actual_hash = canonical_sha256(code)
        if expected_hash != actual_hash:
            raise ValueError(
                f"{path}: candidate {index} code hash mismatch: "
                f"{expected_hash!r} != {actual_hash}"
            )
        codes[name] = code
    return codes


def calibration_loader(
    root: Path,
    resolution: int,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    dataset = datasets.ImageFolder(str(root), transform=transform)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def validation_loader(
    records: list[tuple[Path, int]],
    resolution: int,
    batch_size: int,
    workers: int,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(math.ceil(resolution / 0.875)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return DataLoader(
        FlatImageNetValidation(records, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    top1_correct = 0
    top5_correct = 0
    image_count = 0
    model.eval()
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="validation", leave=False):
            logits = model(images.to(device, non_blocking=True))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.ndim != 2 or logits.shape[1] != 1000:
                raise ValueError(f"unexpected logits shape {tuple(logits.shape)}")
            targets = targets.to(device, non_blocking=True)
            predictions = logits.topk(5, dim=1).indices
            top1_correct += predictions[:, 0].eq(targets).sum().item()
            top5_correct += predictions.eq(targets[:, None]).any(dim=1).sum().item()
            image_count += targets.numel()
    return {
        "image_count": image_count,
        "top1_correct": int(top1_correct),
        "top5_correct": int(top5_correct),
        "top1_percent": 100.0 * top1_correct / image_count,
        "top5_percent": 100.0 * top5_correct / image_count,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    files = [
        args.onnx,
        args.supernet_config,
        args.calibration_manifest,
        args.calibration_metadata,
        args.validation_labels,
        args.synset_mapping,
        args.ofa_checkpoint,
    ]
    if args.codes_json is not None:
        files.append(args.codes_json)
    ensure_files(files)
    if args.ofa_checkpoint.resolve() != OFA_LOADER_CHECKPOINT.resolve():
        raise ValueError(
            "--ofa-checkpoint must resolve to the repository-relative path read by "
            f"OFA's loader: {OFA_LOADER_CHECKPOINT.resolve()}"
        )
    if not args.calibration_root.is_dir() or not args.validation_images.is_dir():
        raise FileNotFoundError("calibration or validation image directory is missing")

    calibration_audit = audit_calibration_partition(
        args.calibration_root,
        args.calibration_manifest,
        args.calibration_metadata,
    )
    synsets = read_synsets(args.synset_mapping)
    calibration_class_order = sorted(
        path.name for path in args.calibration_root.iterdir() if path.is_dir()
    )
    if synsets != calibration_class_order:
        raise ValueError(
            "LOC synset order differs from sorted ImageFolder training-class order"
        )
    class_order_sha256 = canonical_sha256(synsets)
    all_validation_records, validation_label_audit = read_validation_records(
        args.validation_images, args.validation_labels, synsets
    )
    if args.val_limit and args.val_limit < len(all_validation_records):
        selection_rng = random.Random(args.seed + 2)
        selected_indices = sorted(
            selection_rng.sample(range(len(all_validation_records)), args.val_limit)
        )
        validation_records = [all_validation_records[index] for index in selected_indices]
        scope = "diagnostic_subset"
    else:
        validation_records = all_validation_records
        scope = "official_full_validation"

    validation_ids = [path.stem for path, _ in validation_records]
    if any("/CLS-LOC/val/" not in str(path) for path, _ in validation_records):
        raise ValueError("validation records contain a non-validation-split source")

    tailor = load_tailor(args.onnx, args.supernet_config)
    matrix_info = load_kernel_transform_matrices(tailor)
    if args.codes_json is None:
        codes = deterministic_codes(tailor, args.seed)
        requested_subnets = args.subnets or list(SUBNET_NAMES)
        unknown_subnets = sorted(set(requested_subnets) - set(SUBNET_NAMES))
    else:
        codes = selected_codes(args.codes_json)
        requested_subnets = args.subnets or list(codes)
        unknown_subnets = sorted(set(requested_subnets) - set(codes))
    if unknown_subnets:
        raise ValueError(f"unknown requested SubNets: {unknown_subnets}")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "seed": args.seed,
        "scope": scope,
        "protocol": {
            "calibration": "OFA-compatible BN reset on 2,000 training images",
            "calibration_augmentation": (
                "seeded RandomResizedCrop, RandomHorizontalFlip, ColorJitter"
            ),
            "accuracy": "single-crop FP32 inference on official ILSVRC validation",
            "accuracy_preprocess": "Resize(ceil(resolution/0.875)), CenterCrop",
            "batch_size": args.batch_size,
            "calibration_batch_size": args.calibration_batch_size,
            "workers": args.workers,
            "shared_bn_calibration_seed": args.seed + 100,
        },
        "inputs": {
            "experiment_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "onnx": {"path": str(args.onnx.resolve()), "sha256": sha256_file(args.onnx)},
            "supernet_config": {
                "path": str(args.supernet_config.resolve()),
                "sha256": sha256_file(args.supernet_config),
            },
            "ofa_checkpoint": {
                "path": str(args.ofa_checkpoint.resolve()),
                "sha256": sha256_file(args.ofa_checkpoint),
            },
            "calibration": calibration_audit,
            "validation": {
                "images_root": str(args.validation_images.resolve()),
                "labels_path": str(args.validation_labels.resolve()),
                "labels_sha256": sha256_file(args.validation_labels),
                "synset_mapping_path": str(args.synset_mapping.resolve()),
                "synset_mapping_sha256": sha256_file(args.synset_mapping),
                "class_index_order_sha256": class_order_sha256,
                "class_index_matches_sorted_training_directories": True,
                "label_schema_audit": validation_label_audit,
                "available_image_count": len(all_validation_records),
                "evaluated_image_count": len(validation_records),
                "evaluated_image_ids_sha256": canonical_sha256(validation_ids),
            },
        },
        "split_checks": {
            "calibration_source_is_imagenet_train": True,
            "accuracy_source_is_imagenet_validation": True,
            "calibration_and_accuracy_source_splits_disjoint": True,
            "sensitivity_partition_used_for_accuracy": False,
        },
        "kernel_transform_matrices": matrix_info,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "onnx": onnx.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "requested_subnets": requested_subnets,
        "subnets": [],
        "limitations": [
            (
                "This evaluates selected candidates from a fixed candidate pool, not "
                "a complete architecture-space search."
                if args.codes_json is not None
                else "This is a four-subnet accuracy audit, not a complete architecture-space search."
            ),
            "BN calibration augmentations are deterministic for the recorded software and seed.",
            "Kernel transformation matrices come from OFA's public pretrained w1.0 supernet.",
        ],
    }
    if args.codes_json is not None:
        result["inputs"]["selection_manifest"] = {
            "path": str(args.codes_json.resolve()),
            "sha256": sha256_file(args.codes_json),
        }
    write_json(args.output, result)

    for subnet_index, name in enumerate(requested_subnets):
        code = codes[name]
        # Paired SubNets must see identical stochastic BN-calibration crops.
        subnet_seed = args.seed + 100
        seed_everything(subnet_seed)
        started = time.perf_counter()
        tailor.transform(code)
        # TailorIR's built modules reference persistent supernet parameters.
        # Isolate evaluation, BN, and device state from later transformations by
        # transferring an independent static net.  Kernel matrices already match
        # each depthwise superweight device in load_kernel_transform_matrices().
        model = copy.deepcopy(tailor.tir.build()).to(device).eval()
        resolution = int(code["resolution"])
        try:
            flops, parameter_count = tailor.tir.count_flops_params()
            flops = float(flops)
            parameter_count = int(parameter_count)
        except Exception as error:  # The accuracy result remains useful if ptflops drifts.
            flops = None
            parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
            flop_error = f"{type(error).__name__}: {error}"
        else:
            flop_error = None

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        bn_loader = calibration_loader(
            args.calibration_root,
            resolution,
            args.calibration_batch_size,
            args.workers,
            subnet_seed,
        )
        calibration_started = time.perf_counter()
        calibrated_images = set_running_statistics(model, bn_loader, device)
        calibration_seconds = time.perf_counter() - calibration_started
        if calibrated_images != calibration_audit["image_count"]:
            raise RuntimeError(
                f"calibrated on {calibrated_images}, expected {calibration_audit['image_count']}"
            )

        val_loader = validation_loader(
            validation_records,
            resolution,
            args.batch_size,
            args.workers,
        )
        evaluation_started = time.perf_counter()
        accuracy = evaluate(model, val_loader, device)
        evaluation_seconds = time.perf_counter() - evaluation_started
        subnet_result = {
            "name": name,
            "sampling_seed": (
                args.seed
                if name == "seeded_random"
                else args.seed + 1
                if name == "seeded_compound"
                else None
            ),
            "execution_seed": subnet_seed,
            "resolution": resolution,
            "code": code,
            "code_sha256": canonical_sha256(code),
            "parameter_count": parameter_count,
            "flops": flops,
            "flop_count_error": flop_error,
            "bn_calibration_images": calibrated_images,
            "accuracy": accuracy,
            "timing_seconds": {
                "bn_calibration": calibration_seconds,
                "validation": evaluation_seconds,
                "total": time.perf_counter() - started,
            },
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        }
        result["subnets"].append(subnet_result)
        write_json(args.output, result)
        del model, bn_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["status"] = "complete"
    result["checks"] = {
        "all_requested_subnets_completed": len(result["subnets"])
        == len(requested_subnets),
        "all_subnets_used_all_calibration_images": all(
            item["bn_calibration_images"] == calibration_audit["image_count"]
            for item in result["subnets"]
        ),
        "all_subnets_used_requested_validation_images": all(
            item["accuracy"]["image_count"] == len(validation_records)
            for item in result["subnets"]
        ),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
