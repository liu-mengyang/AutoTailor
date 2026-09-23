#!/usr/bin/env python3
"""Join frozen selections to fresh, completed 50K-image validation measurements."""

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autotailor.tailor.acc_predictors.sensitivity_provenance import file_hash, require_training_sensitivity


def summarize(sensitivity, selection, validation):
    require_training_sensitivity(sensitivity)
    if validation.get("status") != "complete" or validation.get("validation_image_count") != 50000:
        raise ValueError("Final ImageNet evaluation is not complete")
    measured = {}
    for row in validation["subnets"]:
        if row["code_sha256"] in measured or row["accuracy"]["image_count"] != 50000:
            raise ValueError("Duplicate or incomplete measured architecture")
        if row["bn_calibration_images"] != sensitivity["data_provenance"]["bn_calibration"]["image_count"]:
            raise ValueError("Final evaluation did not use the collection BN image count")
        measured[row["code_sha256"]] = row["accuracy"]["accuracy"]
    if set(measured) != {row["code_sha256"] for row in selection["selected_candidates"]}:
        raise ValueError("Final evaluation does not cover exactly the frozen selection union")
    budgets = {}
    for row in selection["selections"]:
        budget = budgets.setdefault(row["quantile"], {"quantile": row["quantile"],
                                                      "flops_budget": row["flops_budget"], "methods": {}})
        budget["methods"][row["method"]] = {
            "candidate_id": row["candidate_id"], "code_sha256": row["code_sha256"],
            "top1_percent": measured[row["code_sha256"]],
        }
    rows = sorted(budgets.values(), key=lambda row: row["quantile"])
    for row in rows:
        clean = row["methods"]["training_only_sensitivity"]
        historical = row["methods"].get("historical_reference_diagnostic")
        if historical:
            row["training_only_minus_historical_pp"] = clean["top1_percent"] - historical["top1_percent"]
            row["selection_changed"] = clean["code_sha256"] != historical["code_sha256"]
        row["training_only_minus_flops_proxy_pp"] = clean["top1_percent"] - row["methods"]["flops_proxy"]["top1_percent"]
    methods = rows[0]["methods"]
    return {
        "scope": "fresh 50K-image accuracy of frozen selections; not full-pool ranking or causal leakage effect",
        "budget_count": len(rows), "unique_evaluated_architectures": len(measured),
        "sensitivity_measurements": len(sensitivity["probe_results"]),
        "sensitivity_images_per_measurement": sensitivity["protocol"]["image_count"],
        "bn_calibration_images": sensitivity["data_provenance"]["bn_calibration"]["image_count"],
        "image_partitions_disjoint": True,
        "mean_top1_percent_by_method": {method: statistics.mean(row["methods"][method]["top1_percent"] for row in rows) for method in methods},
        "selection_changes_vs_historical": sum(row.get("selection_changed", False) for row in rows),
        "per_budget": rows,
        "limitations": ["Historical sensitivity checkpoint/BN provenance is unknown; differences cannot be attributed solely to leakage.",
                        "This covers only the frozen candidate set and FLOP budgets, not every model or published result."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sensitivity = json.loads(args.sensitivity.read_text())
    selection = json.loads(args.selection.read_text())
    validation = json.loads(args.validation.read_text())
    if (validation["contract"]["sensitivity_sha256"] != file_hash(args.sensitivity)
            or validation["contract"]["selection_sha256"] != file_hash(args.selection)
            or selection["inputs"]["sensitivity"]["sha256"] != file_hash(args.sensitivity)):
        raise ValueError("Input artifact hashes do not match the frozen evaluation contract")
    output = summarize(sensitivity, selection, validation)
    output["inputs"] = {name: {"path": str(path), "sha256": file_hash(path)} for name, path in [
        ("sensitivity", args.sensitivity), ("selection", args.selection), ("validation", args.validation)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
