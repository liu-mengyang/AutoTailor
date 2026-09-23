import copy
import json
import os
import contextlib
import io
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from autotailor.tailor.acc_predictors.sensitivity_collection import calibrate_batch_norm, evaluate_top1
from autotailor.tailor.acc_predictors.sensitivity_provenance import (
    audit_training_sources, require_training_sensitivity,
)
from scripts.predictor_factory.accuracy_predictor.collect_acc_sensitivity import (
    generate_codes, iter_probes, nested_values,
)


class SensitivityProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.train = self.root / "CLS-LOC/train"
        self.val = self.root / "CLS-LOC/val"
        self.train.mkdir(parents=True)
        self.val.mkdir()
        self.paths = []
        for i in range(4):
            path = self.train / f"n00000001_{i}.JPEG"
            path.write_bytes(bytes([i]))
            self.paths.append(path)

    def audit(self, calibration=None, labels=None):
        return audit_training_sources(
            self.train, self.paths[:2] if calibration is None else calibration,
            self.paths[2:] if labels is None else labels,
        )

    def test_train_view_named_val_is_allowed_if_sources_are_training(self):
        view = self.root / "view/val"
        view.mkdir(parents=True)
        links = []
        for source in self.paths[2:]:
            link = view / source.name
            link.symlink_to(source)
            links.append(link)
        audit = self.audit(labels=links)
        self.assertEqual(audit["sensitivity"]["source_paths"], sorted(str(path.resolve()) for path in self.paths[2:]))
        require_training_sensitivity({"status": "complete", "data_provenance": audit})

    def test_official_validation_source_is_rejected(self):
        path = self.val / "ILSVRC2012_val_00000001.JPEG"
        path.write_bytes(b"test")
        with self.assertRaisesRegex(ValueError, "outside"):
            self.audit(labels=[path])

    def test_symlink_named_as_training_cannot_hide_validation(self):
        path = self.val / "renamed.JPEG"
        path.write_bytes(b"test")
        disguised = self.train / "pretend_train.JPEG"
        disguised.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.audit(labels=[disguised])

    def test_copied_official_validation_identifier_is_rejected(self):
        path = self.train / "ILSVRC2012_val_00000001.JPEG"
        path.write_bytes(b"test")
        with self.assertRaisesRegex(ValueError, "official-validation"):
            self.audit(labels=[path])

    def test_duplicate_sources_and_cross_partition_overlap_are_rejected(self):
        for calibration, labels in [([self.paths[0]] * 2, self.paths[2:]),
                                    (self.paths[:2], self.paths[1:])]:
            with self.assertRaises(ValueError):
                self.audit(calibration, labels)

    def test_hardlinks_cannot_hide_overlap(self):
        link = self.train / "same-image.JPEG"
        os.link(self.paths[0], link)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.audit(labels=[link])

    def test_unaudited_or_partial_weights_cannot_be_used_for_search(self):
        audit = self.audit()
        for payload in [{"base_acc": [80.0]}, {"status": "partial_diagnostic", "data_provenance": audit}]:
            with self.assertRaisesRegex(ValueError, "recollect"):
                require_training_sensitivity(payload)

    def test_metadata_must_match_exact_source_lists(self):
        audit = self.audit()
        audit["sensitivity"]["source_paths"][0] = str(self.val / "fake.JPEG")
        with self.assertRaises(ValueError):
            require_training_sensitivity({"status": "complete", "data_provenance": audit})

    def test_calibration_and_final_accuracy_do_not_share_bn_updates(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2, bias=False))
        images = torch.tensor([[0., 2.], [2., 4.], [4., 6.]])
        labels = torch.tensor([0, 1, 0])
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=2)
        self.assertEqual(calibrate_batch_norm(model, loader, "cpu"), 3)
        self.assertTrue(torch.allclose(model[0].running_mean, torch.tensor([2., 4.])))
        self.assertTrue(torch.allclose(model[0].running_var, torch.tensor([2/3, 2/3])))
        frozen = copy.deepcopy(model[0].state_dict())
        result = evaluate_top1(model, loader, "cpu")
        self.assertEqual(result["image_count"], 3)
        for name, tensor in frozen.items():
            self.assertTrue(torch.equal(tensor, model[0].state_dict()[name]))

    def test_accuracy_counts_short_last_batch_exactly(self):
        images = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
        labels = torch.tensor([0, 1, 1])
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=2)
        result = evaluate_top1(torch.nn.Identity(), loader, "cpu")
        self.assertEqual(result["correct"], 2)
        self.assertAlmostEqual(result["accuracy"], 200/3)

    def test_probe_order_preserves_legacy_weight_layout(self):
        class Tailor:
            global_vars = {"resolution": [128, 224]}
            stage_vars = {"reduce_depth": [-1, 0], "reduce_depth_skipped": [-1]}
            skipcode = {"reduce_depth": [-1]}
            block_vars = {"Block": {"width": [0.5, 1.0]}}
            supercode = {"resolution": 224, "reduce_depth": [0, 0], "Block": {"width": [[1., 1.], [1.]]}}
        codes = generate_codes(Tailor())
        probes = list(iter_probes(codes))
        self.assertEqual(len(probes), 5)
        self.assertEqual(probes[1][1]["reduce_depth"], [-1, 0])
        results = {name: {"accuracy": i} for i, (name, _) in enumerate(probes)}
        nested = nested_values(codes, results, "accuracy")
        self.assertEqual(nested, {"resolution": [0], "reduce_depth": [1], "Block": {"width": [2, 3, 4]}})

    def test_candidate_accuracy_labels_cannot_change_frozen_selections(self):
        from exp_scripts.select_training_only_sensitivity import main
        base = {"width": 2, "depth": 2, "resolution": 224}
        payload = {
            "status": "complete", "data_provenance": self.audit(), "base_code": base,
            "base_acc": [80.0], "inputs": {},
            "code": {"width": [{**base, "width": 1}], "depth": [{**base, "depth": 1}],
                     "resolution": [{**base, "resolution": 128}]},
            "acc": {"width": [79.0], "depth": [78.0], "resolution": [77.0]},
        }
        sensitivity = self.root / "sensitivity.json"
        sensitivity.write_text(json.dumps(payload))
        candidates = self.root / "candidates.json"
        codes = [{**base, "width": 1, "depth": 1}, {**base, "depth": 1, "resolution": 128}]
        outputs = []
        for index, labels in enumerate([[0.0, 100.0], [100.0, 0.0]]):
            candidates.write_text(json.dumps({"code": codes, "flops": [2., 1.], "acc": labels}))
            output = self.root / f"selection{index}.json"
            with patch.object(sys, "argv", ["select", "--sensitivity", str(sensitivity), "--candidates", str(candidates), "--output", str(output)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    main()
            outputs.append(json.loads(output.read_text()))
        self.assertEqual(outputs[0]["selections"], outputs[1]["selections"])
        self.assertEqual(outputs[0]["selected_candidates"], outputs[1]["selected_candidates"])


if __name__ == "__main__":
    unittest.main()
