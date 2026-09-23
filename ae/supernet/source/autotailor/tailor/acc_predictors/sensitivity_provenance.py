"""Fail-closed image provenance checks for sensitivity collection."""

import hashlib
import json
from pathlib import Path


SCHEMA = "training-only-sensitivity-v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_training_sources(training_root, calibration_paths, sensitivity_paths):
    """Check actual image targets, not a directory name or a metadata assertion.

    training_root must be the original ImageNet train directory, not a view.
    The returned source lists make the exact collection membership reviewable.
    """
    root = Path(training_root).resolve(strict=True)
    if root.name != "train" or root.parent.name != "CLS-LOC":
        raise ValueError("--imagenet-train-root must identify the original CLS-LOC/train directory")
    groups = {}
    inode_groups = {}
    for name, paths in [("bn_calibration", calibration_paths), ("sensitivity", sensitivity_paths)]:
        resolved = [Path(path).resolve(strict=True) for path in paths]
        if not resolved:
            raise ValueError(f"{name} image partition is empty")
        if any(not path.is_file() or not path.is_relative_to(root) for path in resolved):
            raise ValueError(f"{name} contains images outside the original ImageNet training split")
        if any(path.name.startswith("ILSVRC2012_val_") for path in resolved):
            raise ValueError(f"{name} contains official-validation image identifiers")
        sources = sorted(str(path) for path in resolved)
        inodes = {(path.stat().st_dev, path.stat().st_ino) for path in resolved}
        if len(set(sources)) != len(sources) or len(inodes) != len(sources):
            raise ValueError(f"{name} contains duplicate source images")
        groups[name] = sources
        inode_groups[name] = inodes
    if set(groups["bn_calibration"]) & set(groups["sensitivity"]) or (
        inode_groups["bn_calibration"] & inode_groups["sensitivity"]
    ):
        raise ValueError("BN calibration and sensitivity image partitions overlap")
    return {
        "schema": SCHEMA,
        "accuracy_source_split": "imagenet_train",
        "original_training_root": str(root),
        "official_validation_used": False,
        "calibration_and_sensitivity_disjoint": True,
        **{
            name: {"image_count": len(sources), "source_paths": sources,
                   "source_paths_sha256": canonical_hash(sources)}
            for name, sources in groups.items()
        },
    }


def require_training_sensitivity(payload):
    provenance = payload.get("data_provenance", {})
    if (
        provenance.get("schema") != SCHEMA
        or provenance.get("accuracy_source_split") != "imagenet_train"
        or provenance.get("official_validation_used") is not False
        or provenance.get("calibration_and_sensitivity_disjoint") is not True
        or payload.get("status") != "complete"
    ):
        raise ValueError("Sensitivity weights lack completed training-image provenance; recollect them on ImageNet train")
    groups = {}
    root = Path(provenance.get("original_training_root", ""))
    if not root.is_absolute() or root.name != "train" or root.parent.name != "CLS-LOC":
        raise ValueError("Invalid original ImageNet training root in sensitivity provenance")
    for name in ["bn_calibration", "sensitivity"]:
        group = provenance.get(name, {})
        sources = group.get("source_paths", [])
        if (not sources or len(sources) != group.get("image_count")
                or len(set(sources)) != len(sources)
                or canonical_hash(sources) != group.get("source_paths_sha256")):
            raise ValueError(f"Invalid {name} source manifest in sensitivity weights")
        if any(not Path(path).is_relative_to(root) or Path(path).name.startswith("ILSVRC2012_val_") for path in sources):
            raise ValueError("Sensitivity provenance contains non-training image paths")
        groups[name] = set(sources)
    if groups["bn_calibration"] & groups["sensitivity"]:
        raise ValueError("Sensitivity provenance image partitions overlap")
    return provenance
