"""Architecture-level split provenance for accuracy-regression checkpoints.

These checks establish architecture holdout, not image-level split provenance.
Image labels must still be collected on appropriately separated image sets.
"""

import hashlib
import json
import math


SCHEMA = "accuracy-predictor-split-v1"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def dataset_identity(encoder, codes, accuracies):
    if not codes or len(codes) != len(accuracies):
        raise ValueError("Expected nonempty, equally sized architecture and accuracy lists")
    if any(not math.isfinite(float(value)) for value in accuracies):
        raise ValueError("Accuracy labels must be finite")
    features = [encoder.encode(code) for code in codes]
    code_ids = [fingerprint(code) for code in codes]
    feature_ids = [fingerprint(list(map(float, row))) for row in features]
    dataset_id = fingerprint([code_ids, feature_ids, list(map(float, accuracies))])
    return features, code_ids, feature_ids, dataset_id


def heldout_indices(encoder, codes, accuracies, metadata, *, saved_split=False):
    """Restore the saved validation set, or exclude training rows from new data.

    Legacy weights remain usable for inference, but cannot certify a holdout.
    No random split of an already trained checkpoint can repair missing lineage.
    """
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        raise ValueError("Checkpoint has no split provenance; retrain before held-out evaluation")
    _, code_ids, feature_ids, dataset_id = dataset_identity(encoder, codes, accuracies)
    trained_codes = set(metadata["training_architecture_sha256"])
    trained_features = set(metadata["training_feature_sha256"])
    if not trained_codes or not trained_features:
        raise ValueError("Checkpoint has empty training provenance")
    if saved_split:
        if dataset_id != metadata["dataset_sha256"]:
            raise ValueError("Dataset or encoder differs from the checkpoint's saved split")
        indices = metadata["validation_indices"]
    else:
        indices = [
            i for i in range(len(codes))
            if code_ids[i] not in trained_codes and feature_ids[i] not in trained_features
        ]
    if not indices:
        raise ValueError("No held-out architectures remain (a no-split checkpoint has no internal validation set)")
    if len(indices) != len(set(indices)) or any(
        not isinstance(i, int) or i < 0 or i >= len(codes) for i in indices
    ):
        raise ValueError("Invalid saved validation indices")
    if any(code_ids[i] in trained_codes or feature_ids[i] in trained_features for i in indices):
        raise ValueError("Evaluation architectures overlap checkpoint training data")
    if len({code_ids[i] for i in indices}) != len(indices) or len(
        {feature_ids[i] for i in indices}
    ) != len(indices):
        raise ValueError("Duplicate evaluation architectures; canonicalize and deduplicate first")
    return indices
