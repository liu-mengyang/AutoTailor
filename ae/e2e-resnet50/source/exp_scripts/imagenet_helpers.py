#!/usr/bin/env python3
"""Evaluate selected OFA-ResNet50 SubNets on held-out ImageNet validation.

Every selected architecture is transformed from the same repository ONNX
supernet.  Batch-normalization statistics are calibrated on the deterministic
2,000-image training partition, while accuracy is measured on the disjoint
official 50,000-image ILSVRC validation set.  Selection manifests may contain
the same architecture more than once; evaluation is deduplicated by the
canonical architecture hash while all selection provenance is retained.

The output is written atomically after every architecture.  Re-running the
same command resumes an interrupted result only after verifying the complete
input/protocol fingerprint and every already-completed architecture.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
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
from autotailor.tir.graph_utils import shape_inference
from autotailor.tir.tailor_ir import TailorIR


SCHEMA_VERSION = "ofa-resnet50-imagenet-selected-accuracy-v1"
DEFAULT_SEED = 20260903
DEFAULT_ONNX = REPO_ROOT / "models/onnx/ofa_supernet_resnet50.onnx"
DEFAULT_SUPERNET_CONFIG = REPO_ROOT / "configs/supernet/ofa_resnet50.toml"
DEFAULT_IMAGENET_ROOT = Path(
    "/home/lmy/.cache/kagglehub/competitions/imagenet-object-localization-challenge"
)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ImageFile.LOAD_TRUNCATED_IMAGES = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codes-json",
        type=Path,
        required=True,
        help="Selection manifest with a non-empty selected_candidates array",
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument(
        "--supernet-config", type=Path, default=DEFAULT_SUPERNET_CONFIG
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
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "results/rebuttal/resnet50_factorial/resnet50_accuracy_f0.json"
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
        help="Diagnostic-only deterministic validation subset; 0 uses all 50,000",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.calibration_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.workers < 0 or args.val_limit < 0:
        parser.error("--workers and --val-limit must be non-negative")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def set_running_statistics(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> int:
    """OFA-compatible BN recalibration with exact image accounting."""
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
                batch_variance = (
                    inputs - batch_mean[None, :, None, None]
                ).square().mean(dim=(0, 2, 3))
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

    if not means:
        raise RuntimeError("transformed network contains no BatchNorm2d layers")

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
        if not isinstance(module, nn.BatchNorm2d):
            raise TypeError(f"expected BatchNorm2d at {name}")
        feature_count = means[name].average.shape[0]
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
    return records, {
        "row_count": len(records),
        "box_annotation_count": box_annotation_count,
        "prediction_string_group_size": 5,
        "first_token_is_valid_class_for_every_row": True,
        "all_box_labels_match_first_token_per_row": True,
        "all_rows_have_parseable_box_coordinates": True,
        "image_ids_are_unique": True,
    }


def audit_calibration_partition(
    root: Path,
    manifest_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text())
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["class", "purpose", "source", "link"]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected calibration manifest fields: {reader.fieldnames}")
        rows = [row for row in reader if row["purpose"] == "bn_calibration"]

    dataset = datasets.ImageFolder(str(root))
    dataset_paths = {str(Path(path).resolve()) for path, _ in dataset.samples}
    manifest_paths = {str(Path(row["source"]).resolve()) for row in rows}
    resolved_links = {
        str(
            (
                Path(row["link"])
                if Path(row["link"]).is_absolute()
                else REPO_ROOT / row["link"]
            ).resolve()
        )
        for row in rows
    }
    if dataset_paths != manifest_paths or resolved_links != manifest_paths:
        raise ValueError(
            "calibration ImageFolder does not exactly match bn_calibration manifest rows"
        )
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
        "root": str(root.resolve()),
        "image_count": len(rows),
        "class_count": len(dataset.classes),
        "purpose": "bn_calibration",
        "source_split": metadata.get("source_split"),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "metadata_path": str(metadata_path.resolve()),
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


def _candidate_metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in item.items() if key != "code"}


def selected_codes(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text())
    candidates = payload.get("selected_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{path}: selected_candidates must be a non-empty list")

    unique: dict[str, dict[str, Any]] = {}
    candidate_id_hashes: dict[str, str] = {}
    evaluation_index_hashes: dict[int, str] = {}
    raw_hashes: list[str] = []
    for position, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise TypeError(f"{path}: selected_candidates[{position}] must be an object")
        code = item.get("code")
        expected_hash = item.get("code_sha256")
        if not isinstance(code, dict):
            raise TypeError(f"{path}: selected_candidates[{position}].code must be an object")
        actual_hash = canonical_sha256(code)
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            raise ValueError(
                f"{path}: selected_candidates[{position}].code_sha256 must be "
                "64 lowercase hex characters"
            )
        if expected_hash != actual_hash:
            raise ValueError(
                f"{path}: selected_candidates[{position}] code hash mismatch: "
                f"{expected_hash} != {actual_hash}"
            )

        candidate_id = item.get("candidate_id")
        if candidate_id is None and isinstance(item.get("candidate_index"), int):
            candidate_id = f"candidate_{item['candidate_index']:04d}"
        if candidate_id is None:
            candidate_id = f"selection_{position:04d}"
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"{path}: selected_candidates[{position}] has invalid candidate_id")
        prior_id_hash = candidate_id_hashes.setdefault(candidate_id, actual_hash)
        if prior_id_hash != actual_hash:
            raise ValueError(f"{path}: candidate_id {candidate_id!r} maps to multiple codes")

        evaluation_index = item.get("evaluation_index")
        if evaluation_index is not None:
            if not isinstance(evaluation_index, int) or evaluation_index < 0:
                raise ValueError(
                    f"{path}: selected_candidates[{position}].evaluation_index must be non-negative"
                )
            prior_index_hash = evaluation_index_hashes.setdefault(
                evaluation_index, actual_hash
            )
            if prior_index_hash != actual_hash:
                raise ValueError(
                    f"{path}: evaluation_index {evaluation_index} maps to multiple codes"
                )

        selected_by = item.get("selected_by", [])
        if not isinstance(selected_by, list) or any(
            not isinstance(record, dict) for record in selected_by
        ):
            raise TypeError(
                f"{path}: selected_candidates[{position}].selected_by must be an array of objects"
            )

        raw_hashes.append(actual_hash)
        if actual_hash not in unique:
            unique[actual_hash] = {
                "name": f"architecture_{actual_hash[:12]}",
                "code": copy.deepcopy(code),
                "code_sha256": actual_hash,
                "selection_records": [],
            }
        unique[actual_hash]["selection_records"].append(_candidate_metadata(item))

    architectures = list(unique.values())
    short_names = [item["name"] for item in architectures]
    if len(short_names) != len(set(short_names)):
        raise ValueError("12-character architecture hash prefixes are not unique")
    return architectures, {
        "selected_candidate_entry_count": len(candidates),
        "unique_architecture_count": len(architectures),
        "deduplicated_entry_count": len(candidates) - len(architectures),
        "raw_code_hashes_sha256": canonical_sha256(raw_hashes),
        "unique_code_hashes_sha256": canonical_sha256(
            [item["code_sha256"] for item in architectures]
        ),
    }


def validate_full_code_shape(code: Any, supercode: Any, location: str = "code") -> None:
    if isinstance(supercode, dict):
        if not isinstance(code, dict):
            raise TypeError(f"{location} must be an object")
        if set(code) != set(supercode):
            missing = sorted(set(supercode) - set(code))
            extra = sorted(set(code) - set(supercode))
            raise ValueError(
                f"{location} keys differ from supercode; "
                f"missing={missing}, extra={extra}"
            )
        for key in supercode:
            validate_full_code_shape(code[key], supercode[key], f"{location}.{key}")
        return
    if isinstance(supercode, list):
        if not isinstance(code, list):
            raise TypeError(f"{location} must be an array")
        if len(code) != len(supercode):
            raise ValueError(
                f"{location} has length {len(code)}, expected full-shaped length {len(supercode)}"
            )
        for index, (value, template) in enumerate(zip(code, supercode)):
            validate_full_code_shape(value, template, f"{location}[{index}]")
        return
    if isinstance(supercode, bool):
        if not isinstance(code, bool):
            raise TypeError(f"{location} must be boolean")
    elif isinstance(supercode, int):
        if not isinstance(code, int) or isinstance(code, bool):
            raise TypeError(f"{location} must be an integer")
    elif isinstance(supercode, float):
        if not isinstance(code, (int, float)) or isinstance(code, bool):
            raise TypeError(f"{location} must be numeric")
    elif type(code) is not type(supercode):
        raise TypeError(f"{location} has unexpected scalar type {type(code).__name__}")


def validate_code_values(code: dict[str, Any], tailor: Tailor) -> None:
    for name, choices in tailor.global_vars.items():
        if code[name] not in choices:
            raise ValueError(f"code.{name}={code[name]!r} is outside {choices!r}")
    for name, choices in tailor.stage_vars.items():
        if name.endswith("_skipped"):
            continue
        for stage_index, value in enumerate(code[name]):
            if value not in choices:
                raise ValueError(
                    f"code.{name}[{stage_index}]={value!r} is outside {choices!r}"
                )
    for block_type, dimensions in tailor.block_vars.items():
        for name, choices in dimensions.items():
            for stage_index, stage_values in enumerate(code[block_type][name]):
                for block_index, value in enumerate(stage_values):
                    if value not in choices:
                        raise ValueError(
                            f"code.{block_type}.{name}[{stage_index}][{block_index}]="
                            f"{value!r} is outside {choices!r}"
                        )


def validate_canonical_code(code: dict[str, Any], tailor: Tailor) -> None:
    """Reject semantically inert values that would create duplicate code hashes."""
    for dimension, choices in tailor.stage_vars.items():
        if dimension.endswith("_skipped"):
            continue
        stage_count = len(tailor.supercode[dimension])
        skipped = {
            value if value >= 0 else stage_count + value
            for value in tailor.stage_vars.get(f"{dimension}_skipped", [])
        }
        maximum = max(choices)
        for stage_index in skipped:
            if code[dimension][stage_index] != maximum:
                raise ValueError(
                    f"code.{dimension}[{stage_index}] is skipped and must equal "
                    f"the canonical maximum {maximum!r}"
                )

    if "reduce_depth" not in code:
        return
    for stage_index, reduction in enumerate(code["reduce_depth"]):
        masked_tail = max(0, -int(reduction))
        if masked_tail == 0:
            continue
        for block_type, dimensions in tailor.block_vars.items():
            for dimension, choices in dimensions.items():
                stage_values = code[block_type][dimension][stage_index]
                if not stage_values:
                    continue
                if masked_tail >= len(stage_values):
                    raise ValueError(
                        f"code.reduce_depth[{stage_index}] masks every "
                        f"{block_type} block"
                    )
                maximum = max(choices)
                if any(value != maximum for value in stage_values[-masked_tail:]):
                    raise ValueError(
                        f"code.{block_type}.{dimension}[{stage_index}] has a "
                        "non-maximal value on a depth-masked tail block"
                    )


def calibration_loader(
    root: Path,
    resolution: int,
    batch_size: int,
    workers: int,
    seed: int,
    pin_memory: bool,
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
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def validation_loader(
    records: list[tuple[Path, int]],
    resolution: int,
    batch_size: int,
    workers: int,
    pin_memory: bool,
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
        pin_memory=pin_memory,
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
    if image_count == 0:
        raise RuntimeError("validation loader produced no images")
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


def software_info(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "onnx": onnx.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }


def load_or_initialize_result(
    output: Path,
    initial: dict[str, Any],
    requested_hashes: list[str],
) -> tuple[dict[str, Any], bool]:
    if not output.exists():
        write_json(output, initial)
        return initial, False
    if not output.is_file():
        raise ValueError(f"resume output is not a regular file: {output}")
    result = json.loads(output.read_text())
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"refusing to resume incompatible output schema at {output}")
    if result.get("resume_fingerprint") != initial["resume_fingerprint"]:
        raise ValueError(
            f"refusing to resume {output}: current inputs/protocol differ from the saved run"
        )
    if result.get("requested_code_sha256") != requested_hashes:
        raise ValueError(f"refusing to resume {output}: requested architecture order differs")
    subnets = result.get("subnets")
    if not isinstance(subnets, list):
        raise TypeError(f"{output}: subnets must be an array")
    completed_hashes: set[str] = set()
    for index, subnet in enumerate(subnets):
        if not isinstance(subnet, dict):
            raise TypeError(f"{output}: subnets[{index}] must be an object")
        code = subnet.get("code")
        code_hash = subnet.get("code_sha256")
        if not isinstance(code, dict) or canonical_sha256(code) != code_hash:
            raise ValueError(f"{output}: subnets[{index}] has a corrupt architecture hash")
        if code_hash not in requested_hashes:
            raise ValueError(f"{output}: subnets[{index}] was not requested by this run")
        if code_hash in completed_hashes:
            raise ValueError(f"{output}: duplicate completed architecture {code_hash}")
        if subnet.get("bn_calibration_images") != initial["inputs"]["calibration"]["image_count"]:
            raise ValueError(f"{output}: subnets[{index}] has incomplete BN calibration")
        expected_validation_count = initial["inputs"]["validation"][
            "evaluated_image_count"
        ]
        if subnet.get("accuracy", {}).get("image_count") != expected_validation_count:
            raise ValueError(f"{output}: subnets[{index}] has incomplete validation")
        completed_hashes.add(code_hash)

    if result.get("status") == "complete":
        if completed_hashes != set(requested_hashes):
            raise ValueError(
                f"{output}: complete status does not cover every requested architecture"
            )
        return result, True
    if result.get("status") not in {"running", "interrupted"}:
        raise ValueError(f"{output}: unsupported resumable status {result.get('status')!r}")
    result["status"] = "running"
    result["resume_count"] = int(result.get("resume_count", 0)) + 1
    result.pop("last_error", None)
    result["updated_at_utc"] = utc_now()
    result.setdefault("run_history", []).append(
        {
            "started_at_utc": result["updated_at_utc"],
            "resume": True,
            "software": initial["software"],
        }
    )
    write_json(output, result)
    return result, False


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    ensure_files(
        [
            args.onnx,
            args.supernet_config,
            args.codes_json,
            args.calibration_manifest,
            args.calibration_metadata,
            args.validation_labels,
            args.synset_mapping,
        ]
    )
    if args.onnx.resolve() != DEFAULT_ONNX.resolve():
        raise ValueError(
            "--onnx must resolve to the shared repository model "
            f"{DEFAULT_ONNX.resolve()}"
        )
    if args.supernet_config.resolve() != DEFAULT_SUPERNET_CONFIG.resolve():
        raise ValueError(
            "--supernet-config must resolve to the full OFA-ResNet50 configuration "
            f"{DEFAULT_SUPERNET_CONFIG.resolve()}"
        )
    if not args.calibration_root.is_dir() or not args.validation_images.is_dir():
        raise FileNotFoundError("calibration or validation image directory is missing")

    selection_candidates, selection_audit = selected_codes(args.codes_json)
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
    for candidate in selection_candidates:
        validate_full_code_shape(candidate["code"], tailor.supercode)
        validate_code_values(candidate["code"], tailor)
        validate_canonical_code(candidate["code"], tailor)
        resolution = candidate["code"].get("resolution")
        if not isinstance(resolution, int) or resolution <= 0:
            raise ValueError(f"{candidate['name']}: resolution must be a positive integer")

    current_software = software_info(device)
    script_path = Path(__file__).resolve()
    requested_hashes = [item["code_sha256"] for item in selection_candidates]
    inputs = {
        "experiment_script": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
        "onnx": {
            "path": str(args.onnx.resolve()),
            "sha256": sha256_file(args.onnx),
            "shared_for_all_architectures": True,
            "pretrained_source": "official OFA ResNet50 ONNX weights",
        },
        "supernet_config": {
            "path": str(args.supernet_config.resolve()),
            "sha256": sha256_file(args.supernet_config),
            "full_design_space": True,
        },
        "selection_manifest": {
            "path": str(args.codes_json.resolve()),
            "sha256": sha256_file(args.codes_json),
            **selection_audit,
        },
        "calibration": calibration_audit,
        "validation": {
            "images_root": str(args.validation_images.resolve()),
            "labels_path": str(args.validation_labels.resolve()),
            "labels_sha256": sha256_file(args.validation_labels),
            "synset_mapping_path": str(args.synset_mapping.resolve()),
            "synset_mapping_sha256": sha256_file(args.synset_mapping),
            "class_index_order_sha256": canonical_sha256(synsets),
            "class_index_matches_sorted_training_directories": True,
            "label_schema_audit": validation_label_audit,
            "available_image_count": len(all_validation_records),
            "evaluated_image_count": len(validation_records),
            "evaluated_image_ids_sha256": canonical_sha256(validation_ids),
        },
    }
    protocol = {
        "calibration": "OFA-compatible BN reset on 2,000 ImageNet training images",
        "calibration_augmentation": (
            "seeded RandomResizedCrop, RandomHorizontalFlip, ColorJitter"
        ),
        "accuracy": "single-crop FP32 inference on official ILSVRC validation",
        "accuracy_preprocess": "Resize(ceil(resolution/0.875)), CenterCrop",
        "accuracy_aggregation": (
            "sum exact per-batch correct counts and targets.numel, including the "
            "final partial batch"
        ),
        "batch_size": args.batch_size,
        "calibration_batch_size": args.calibration_batch_size,
        "workers": args.workers,
        "shared_bn_calibration_seed": args.seed + 100,
        "val_limit": args.val_limit,
    }
    resume_contract = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "scope": scope,
        "protocol": protocol,
        "inputs": inputs,
        "software": current_software,
        "requested_code_sha256": requested_hashes,
    }
    started_at = utc_now()
    initial: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": started_at,
        "updated_at_utc": started_at,
        "resume_count": 0,
        "resume_fingerprint": canonical_sha256(resume_contract),
        "seed": args.seed,
        "scope": scope,
        "protocol": protocol,
        "inputs": inputs,
        "split_checks": {
            "calibration_source_is_imagenet_train": True,
            "accuracy_source_is_imagenet_validation": True,
            "calibration_and_accuracy_source_splits_disjoint": True,
            "sensitivity_partition_used_for_accuracy": False,
        },
        "software": current_software,
        "requested_architecture_count": len(selection_candidates),
        "requested_code_sha256": requested_hashes,
        "subnets": [],
        "run_history": [
            {
                "started_at_utc": started_at,
                "resume": False,
                "software": current_software,
            }
        ],
        "limitations": [
            "This evaluates selected candidates from fixed pools, not a new exhaustive search.",
            "BN calibration augmentations are deterministic for the recorded software and seed.",
        ],
    }
    result, already_complete = load_or_initialize_result(
        args.output, initial, requested_hashes
    )
    if already_complete:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    completed_hashes = {item["code_sha256"] for item in result["subnets"]}
    pin_memory = device.type == "cuda"
    try:
        for candidate in selection_candidates:
            if candidate["code_sha256"] in completed_hashes:
                continue
            code = candidate["code"]
            subnet_seed = args.seed + 100
            seed_everything(subnet_seed)
            started = time.perf_counter()
            transform_started = time.perf_counter()
            tailor.transform(code)
            model = copy.deepcopy(tailor.tir.build()).to(device).eval()
            transform_seconds = time.perf_counter() - transform_started
            resolution = int(code["resolution"])
            try:
                flops, parameter_count = tailor.tir.count_flops_params()
                flops = float(flops)
                parameter_count = int(parameter_count)
            except Exception as error:
                flops = None
                parameter_count = int(
                    sum(parameter.numel() for parameter in model.parameters())
                )
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
                pin_memory,
            )
            calibration_started = time.perf_counter()
            calibrated_images = set_running_statistics(model, bn_loader, device)
            calibration_seconds = time.perf_counter() - calibration_started
            if calibrated_images != calibration_audit["image_count"]:
                raise RuntimeError(
                    f"calibrated on {calibrated_images}, expected "
                    f"{calibration_audit['image_count']}"
                )

            val_loader = validation_loader(
                validation_records,
                resolution,
                args.batch_size,
                args.workers,
                pin_memory,
            )
            evaluation_started = time.perf_counter()
            accuracy = evaluate(model, val_loader, device)
            evaluation_seconds = time.perf_counter() - evaluation_started
            result["subnets"].append(
                {
                    "name": candidate["name"],
                    "selection_records": candidate["selection_records"],
                    "execution_seed": subnet_seed,
                    "resolution": resolution,
                    "code": code,
                    "code_sha256": candidate["code_sha256"],
                    "parameter_count": parameter_count,
                    "flops": flops,
                    "flop_count_error": flop_error,
                    "bn_calibration_images": calibrated_images,
                    "accuracy": accuracy,
                    "timing_seconds": {
                        "transform_and_build": transform_seconds,
                        "bn_calibration": calibration_seconds,
                        "validation": evaluation_seconds,
                        "total": time.perf_counter() - started,
                    },
                    "peak_cuda_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else None
                    ),
                    "completed_at_utc": utc_now(),
                }
            )
            completed_hashes.add(candidate["code_sha256"])
            result["updated_at_utc"] = utc_now()
            write_json(args.output, result)
            del model, bn_loader, val_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except BaseException as error:
        result["status"] = "interrupted"
        result["updated_at_utc"] = utc_now()
        result["last_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        write_json(args.output, result)
        raise

    result["status"] = "complete"
    result["updated_at_utc"] = utc_now()
    result["checks"] = {
        "all_requested_architectures_completed": completed_hashes
        == set(requested_hashes),
        "no_architecture_evaluated_more_than_once": len(result["subnets"])
        == len({item["code_sha256"] for item in result["subnets"]}),
        "all_codes_are_full_shaped": True,
        "all_codes_are_canonical": True,
        "all_subnets_use_shared_onnx": True,
        "all_subnets_used_all_calibration_images": all(
            item["bn_calibration_images"] == calibration_audit["image_count"]
            for item in result["subnets"]
        ),
        "all_subnets_used_requested_validation_images": all(
            item["accuracy"]["image_count"] == len(validation_records)
            for item in result["subnets"]
        ),
        "official_full_validation_has_exactly_50000_images": (
            scope != "official_full_validation" or len(validation_records) == 50000
        ),
    }
    if not all(result["checks"].values()):
        raise RuntimeError(f"final audit checks failed: {result['checks']}")
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
