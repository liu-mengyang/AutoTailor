#!/usr/bin/env python3
"""Build a deterministic ImageFolder view from the ImageNet training split.

The generated ``train`` and ``val`` directories are both sourced from the
original training split.  They are named this way only to match AutoTailor's
existing BN-calibration and accuracy-evaluation interfaces.  ``train`` is the
BN-statistics partition and ``val`` is the sensitivity-measurement partition.
The two partitions contain disjoint, class-balanced symbolic links.
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_train", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--bn-per-class", type=int, default=2)
    parser.add_argument("--sensitivity-per-class", type=int, default=8)
    return parser.parse_args()


def class_seed(seed, class_name):
    digest = hashlib.sha256(f"{seed}:{class_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def source_images(class_dir):
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def create_link(source, destination):
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"conflicting link: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"refusing to replace: {destination}")
    destination.symlink_to(source.resolve())


def main():
    args = parse_args()
    if args.bn_per_class < 0 or args.sensitivity_per_class < 1:
        raise ValueError("partition sizes must be non-negative and sensitivity must be non-zero")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"output root must be absent or empty to prevent stale partition links: "
            f"{args.output_root}"
        )

    classes = sorted(path for path in args.source_train.iterdir() if path.is_dir())
    if len(classes) != 1000:
        raise ValueError(f"expected 1,000 ImageNet classes, found {len(classes)}")

    total_per_class = args.bn_per_class + args.sensitivity_per_class
    records = []
    for class_dir in classes:
        images = source_images(class_dir)
        if len(images) < total_per_class:
            raise ValueError(
                f"{class_dir.name} has {len(images)} images, needs {total_per_class}"
            )
        rng = random.Random(class_seed(args.seed, class_dir.name))
        selected = rng.sample(images, total_per_class)
        partitions = (
            ("train", "bn_calibration", selected[: args.bn_per_class]),
            ("val", "sensitivity", selected[args.bn_per_class :]),
        )
        for directory_name, purpose, paths in partitions:
            class_output = args.output_root / directory_name / class_dir.name
            class_output.mkdir(parents=True, exist_ok=True)
            for source in paths:
                destination = class_output / source.name
                create_link(source, destination)
                records.append(
                    {
                        "class": class_dir.name,
                        "purpose": purpose,
                        "source": str(source.resolve()),
                        "link": str(destination),
                    }
                )

    manifest_path = args.output_root / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "purpose", "source", "link"])
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "source_split": str(args.source_train.resolve()),
        "seed": args.seed,
        "class_count": len(classes),
        "bn_calibration_images": args.bn_per_class * len(classes),
        "sensitivity_images": args.sensitivity_per_class * len(classes),
        "total_images": total_per_class * len(classes),
        "partitions_are_disjoint": True,
        "directory_interface": {
            "train": "BN calibration images from the original ImageNet training split",
            "val": "sensitivity images from the original ImageNet training split",
        },
    }
    with (args.output_root / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
