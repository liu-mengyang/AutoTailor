import copy
from pathlib import Path
import tempfile
import unittest

from exp_scripts.evaluate_training_only_sensitivity import validation_records
from exp_scripts.summarize_sensitivity_leakage_validation import summarize
from autotailor.tailor.acc_predictors.sensitivity_provenance import audit_training_sources


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.val = self.root / "ILSVRC/Data/CLS-LOC/val"
        self.val.mkdir(parents=True)
        self.classes = ["n00000001", "n00000002"]
        (self.root / "LOC_synset_mapping.txt").write_text("n00000001 first\nn00000002 second\n")
        self.labels = self.root / "LOC_val_solution.csv"
        self.labels.write_text("ImageId,PredictionString\nILSVRC2012_val_00000001,n00000002 0 0 1 1 n00000002 1 1 2 2\n")
        self.image = self.val / "ILSVRC2012_val_00000001.JPEG"
        self.image.write_bytes(b"image")

    def test_official_labels_use_class_order_and_accept_multiple_boxes(self):
        rows, _ = validation_records(self.root, self.classes, expected_count=1)
        self.assertEqual(rows, [(str(self.image.resolve()), 1)])
        with self.assertRaisesRegex(ValueError, "class order"):
            validation_records(self.root, self.classes[::-1], expected_count=1)

    def test_partial_and_duplicate_validation_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "50000"):
            validation_records(self.root, self.classes)
        self.labels.write_text(self.labels.read_text() + self.labels.read_text().splitlines()[1] + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validation_records(self.root, self.classes, expected_count=2)

    def test_symlink_into_train_is_rejected(self):
        train = self.val.parent / "train"
        train.mkdir()
        source = train / "training.JPEG"
        source.write_bytes(b"train")
        self.image.unlink()
        self.image.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "escapes"):
            validation_records(self.root, self.classes, expected_count=1)

    def test_summary_joins_by_hash_and_requires_complete_matched_measurements(self):
        train = self.val.parent / "train"
        train.mkdir()
        paths = [train / name for name in ["bn.JPEG", "sensitivity.JPEG"]]
        for path in paths:
            path.write_bytes(b"train")
        sensitivity = {"status": "complete", "data_provenance": audit_training_sources(train, paths[:1], paths[1:]),
                       "probe_results": [], "protocol": {"image_count": 1}}
        selection = {"selected_candidates": [{"code_sha256": "clean"}, {"code_sha256": "old"}],
                     "selections": [{"quantile": .5, "flops_budget": 1, "method": method,
                                     "candidate_id": code, "code_sha256": code} for method, code in [
                         ("training_only_sensitivity", "clean"), ("historical_reference_diagnostic", "old"),
                         ("flops_proxy", "old")]]}
        validation = {"status": "complete", "validation_image_count": 50000,
                      "subnets": [{"code_sha256": code, "bn_calibration_images": 1,
                                   "accuracy": {"accuracy": acc, "image_count": 50000}}
                                  for code, acc in [("old", 70), ("clean", 71)]]}
        result = summarize(sensitivity, selection, validation)
        self.assertEqual(result["per_budget"][0]["training_only_minus_historical_pp"], 1)
        for field, value in [("bn_calibration_images", 2), ("code_sha256", "unknown")]:
            bad = copy.deepcopy(validation)
            bad["subnets"][0][field] = value
            with self.assertRaises(ValueError):
                summarize(sensitivity, selection, bad)
        validation["subnets"][0]["accuracy"]["image_count"] = 49999
        with self.assertRaises(ValueError):
            summarize(sensitivity, selection, validation)


if __name__ == "__main__":
    unittest.main()
