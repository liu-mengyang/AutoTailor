from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
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