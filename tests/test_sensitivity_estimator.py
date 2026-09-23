import json

from autotailor.tailor.acc_predictors.sensitivity_estimator import (
    SensitivityEstimator,
    _decode_base_accuracy,
)


def test_decode_current_base_accuracy():
    assert _decode_base_accuracy([77.25]) == 77.25


def test_decode_legacy_base_accuracy():
    legacy = [[1_000_000_000.0], [25_000_000.0], [76.5]]

    assert _decode_base_accuracy(legacy) == 76.5


def test_decode_base_accuracy_rejects_ambiguous_schema():
    invalid_values = [77.25, [], [77.25, 76.5], [[1.0], [2.0], []], [True]]
    for value in invalid_values:
        try:
            _decode_base_accuracy(value)
        except ValueError as error:
            assert "base_acc" in str(error)
        else:
            raise AssertionError(f"accepted ambiguous base_acc schema: {value!r}")


def test_block_weights_do_not_depend_on_stage_key_order(tmp_path):
    class BlockOnlyTailor:
        global_vars = {}
        stage_vars = {}
        block_vars = {"Block": {"kernel_size": [3, 5, 7]}}
        supercode = {"Block": {"kernel_size": [[], [7, 7]]}}

        @staticmethod
        def transform(code):
            return code

    weights_path = tmp_path / "sorted_weights.json"
    weights_path.write_text(
        json.dumps(
            {
                "base_acc": [80.0],
                "acc": {
                    "Block": {
                        "kernel_size": [78.0, 79.0, 77.0, 78.0],
                    }
                },
            },
            sort_keys=True,
        )
    )
    estimator = SensitivityEstimator(BlockOnlyTailor(), weights_path)
    code = {"Block": {"kernel_size": [[], [3, 5]]}}

    assert estimator.predict_accuracy(code) == 76.0
