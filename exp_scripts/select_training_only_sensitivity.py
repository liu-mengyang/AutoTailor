#!/usr/bin/env python3
"""Freeze a training-only sensitivity selection before reading final accuracies.

Optionally freeze historical-table selections as a diagnostic comparator. The
historical comparator has unverified image/checkpoint provenance and must not
be reported as a leakage-free method. All budgets depend only on candidate FLOPs.
"""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autotailor.tailor.acc_predictors.sensitivity_provenance import (
    canonical_hash, file_hash, require_training_sensitivity,
)


def leaves(value, path=()):
    if isinstance(value, dict):
        return {key: item for name, child in value.items() for key, item in leaves(child, (*path, name)).items()}
    if isinstance(value, list):
        return {key: item for index, child in enumerate(value) for key, item in leaves(child, (*path, index)).items()}
    return {path: value}


def build_table(payload, base_code):
    raw = payload["base_acc"]
    base = float(raw[0] if len(raw) == 1 else raw[2][0])
    maximum = leaves(base_code)
    table, profile_hashes = {}, {canonical_hash(base_code)}
    for dimension, entries in payload["code"].items():
        groups = [(entries, payload["acc"][dimension])] if isinstance(entries, list) else [
            (rows, payload["acc"][dimension][name]) for name, rows in entries.items()
        ]
        for codes, accuracies in groups:
            if len(codes) != len(accuracies):
                raise ValueError("Sensitivity probe and label counts differ")
            for code, accuracy in zip(codes, accuracies):
                flat = leaves(code)
                if flat.keys() != maximum.keys():
                    raise ValueError("Sensitivity probe and maximal architecture differ in shape")
                changed = [path for path, value in flat.items() if value != maximum[path]]
                if len(changed) != 1:
                    raise ValueError("Expected one modification per sensitivity probe")
                key = (changed[0], flat[changed[0]])
                delta = float(accuracy) - base
                if not math.isfinite(delta):
                    raise ValueError("Nonfinite sensitivity")
                if key in table:
                    raise ValueError("Duplicate sensitivity probe")
                table[key] = delta
                profile_hashes.add(canonical_hash(code))
    return base, maximum, table, profile_hashes


def score(code, base, maximum, table):
    flat = leaves(code)
    if flat.keys() != maximum.keys():
        raise ValueError("Candidate and sensitivity architecture shapes differ")
    return base + sum(table[(path, value)] for path, value in flat.items() if value != maximum[path])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--historical-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = json.loads(args.sensitivity.read_text())
    require_training_sensitivity(payload)
    base_code = payload["base_code"]
    tables = {"training_only_sensitivity": build_table(payload, base_code)}
    inputs = {"sensitivity": {"path": str(args.sensitivity.resolve()), "sha256": file_hash(args.sensitivity)},
              "candidates": {"path": str(args.candidates.resolve()), "sha256": file_hash(args.candidates)}}
    if args.historical_reference:
        tables["historical_reference_diagnostic"] = build_table(json.loads(args.historical_reference.read_text()), base_code)
        inputs["historical_reference"] = {"path": str(args.historical_reference.resolve()), "sha256": file_hash(args.historical_reference)}
    candidate_payload = json.loads(args.candidates.read_text())
    codes, flops = candidate_payload["code"], candidate_payload["flops"]
    if len(codes) != len(flops) or not codes:
        raise ValueError("Invalid candidate FLOP records")
    forbidden = set.union(*(row[3] for row in tables.values()))
    seen, candidates = set(), []
    for index, (code, value) in enumerate(zip(codes, flops)):
        digest = canonical_hash(code)
        if digest in forbidden or digest in seen:
            continue
        seen.add(digest)
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid candidate FLOPs")
        candidates.append({"candidate_id": f"candidate:{index:04d}", "source_index": index,
                           "code": code, "code_sha256": digest, "flops": value,
                           "scores": {name: score(code, *table[:3]) for name, table in tables.items()}})
    if not candidates:
        raise ValueError("No architecture-held-out candidates remain")
    ordered_flops = sorted(row["flops"] for row in candidates)
    selections, selected = [], {}
    for quantile in [0.1, 0.3, 0.5, 0.7, 0.9]:
        budget = ordered_flops[math.ceil(quantile * (len(ordered_flops) - 1))]
        feasible = [row for row in candidates if row["flops"] <= budget]
        for method in [*tables, "flops_proxy"]:
            winner = min(feasible, key=lambda row: (
                -(row["flops"] if method == "flops_proxy" else row["scores"][method]), row["source_index"],
            ))
            selected[winner["code_sha256"]] = winner
            selections.append({"quantile": quantile, "flops_budget": budget, "method": method,
                               "candidate_id": winner["candidate_id"], "code_sha256": winner["code_sha256"],
                               "feasible_count": len(feasible)})
    output = {"schema_version": "training-only-sensitivity-selection-v1", "inputs": inputs,
              "protocol": "Freeze training-only sensitivity and FLOP-constrained selection before final ImageNet accuracy",
              "candidate_accuracy_labels_used_for_selection": False,
              "historical_reference_provenance": "unverified; diagnostic only",
              "candidate_count": len(candidates), "excluded_probe_or_duplicate_count": len(codes) - len(candidates),
              "selections": selections, "selected_candidates": list(selected.values()), "candidate_scores": candidates,
              "model_inputs": payload["inputs"], "data_provenance": payload["data_provenance"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Frozen {len(selections)} selections ({len(selected)} distinct architectures) to {args.output}")


if __name__ == "__main__":
    main()
