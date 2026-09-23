import copy
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from autotailor.tailor.acc_predictors.evaluation_data import heldout_indices
from autotailor.tailor.acc_predictors.mlp_predictor import MLPPredictor
from autotailor.tailor.acc_predictors.mlp_trainer import (
    AccPredictorTrainer, RegDataset, build_acc_data_loader,
)


class Encoder:
    n_dim = 1

    def encode(self, code):
        return [code["x"] / 10.0]


class AccuracyEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder()
        self.codes = [{"x": i} for i in range(10)]
        self.accs = [50.0 + i for i in range(10)]

    def loaders(self, **kwargs):
        return build_acc_data_loader(
            self.encoder, self.codes, self.accs, n_workers=0, **kwargs,
        )

    def test_validation_label_changes_do_not_change_training(self):
        train, valid, base = self.loaders(seed=4)
        changed = self.accs.copy()
        for i in train.split_metadata["validation_indices"]:
            changed[i] = 99.0
        train2, _, base2 = build_acc_data_loader(
            self.encoder, self.codes, changed, n_workers=0, seed=4,
        )
        self.assertEqual(base, base2)
        self.assertTrue(torch.equal(train.dataset.targets, train2.dataset.targets))
        self.assertTrue(torch.equal(train.dataset.inputs, train2.dataset.inputs))
        self.assertAlmostEqual(base, train.dataset.targets.mean().item())
        states = []
        for loader, offset in [(train, base), (train2, base2)]:
            torch.manual_seed(9)
            model = MLPPredictor(self.encoder, hidden_size=4, n_layers=1, device="cpu")
            trainer = AccPredictorTrainer(model, loader, None, "test", 2, device="cpu")
            trainer.train(offset)
            states.append(copy.deepcopy(model.state_dict()))
        for key in states[0]:
            self.assertTrue(torch.equal(states[0][key], states[1][key]), key)

    def test_nondefault_split_is_disjoint(self):
        train, valid, _ = self.loaders(n_training_sample=9)
        self.assertEqual(len(train.dataset), 9)
        self.assertEqual(len(valid.dataset), 1)
        self.assertFalse(set(train.split_metadata["training_indices"]) & set(
            train.split_metadata["validation_indices"]
        ))

    def test_no_split_has_no_validation(self):
        train, valid, _ = self.loaders(n_training_sample=10)
        self.assertIsNone(valid)
        with self.assertRaisesRegex(ValueError, "No held-out"):
            heldout_indices(self.encoder, self.codes, self.accs, train.split_metadata)

    def test_saved_split_survives_global_rng_changes_and_checkpoint_reload(self):
        train, valid, base = self.loaders(seed=3)
        torch.manual_seed(456)
        second, _, _ = self.loaders(seed=3)
        self.assertEqual(train.split_metadata, second.split_metadata)
        model = MLPPredictor(self.encoder, hidden_size=4, n_layers=1, device="cpu")
        trainer = AccPredictorTrainer(model, train, valid, "test", 1, device="cpu")
        trainer.train(base)
        with tempfile.TemporaryDirectory() as directory:
            trainer.save_dir = directory
            checkpoint = trainer.save()
            loaded = MLPPredictor(
                self.encoder, hidden_size=4, n_layers=1, device="cpu", checkpoint_path=checkpoint,
            )
            self.assertEqual(loaded.split_metadata, train.split_metadata)
            self.assertEqual(loaded.predict_accuracy(self.codes[0]), model.predict_accuracy(self.codes[0]))
            restored = heldout_indices(
                self.encoder, self.codes, self.accs, loaded.split_metadata, saved_split=True,
            )
            self.assertEqual(restored, train.split_metadata["validation_indices"])

    def test_external_evaluation_excludes_all_training_architectures(self):
        train, _, _ = self.loaders()
        indices = heldout_indices(self.encoder, self.codes, self.accs, train.split_metadata)
        self.assertEqual(set(indices), set(train.split_metadata["validation_indices"]))

    def test_legacy_checkpoint_can_predict_but_cannot_claim_holdout(self):
        model = MLPPredictor(self.encoder, hidden_size=4, n_layers=1, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pth"
            torch.save(model.state_dict(), path)
            loaded = MLPPredictor(
                self.encoder, hidden_size=4, n_layers=1, device="cpu", checkpoint_path=path,
            )
            self.assertIsInstance(loaded.predict_accuracy(self.codes[0]), float)
            with self.assertRaisesRegex(ValueError, "no split provenance"):
                heldout_indices(self.encoder, self.codes, self.accs, loaded.split_metadata)

    def test_changed_dataset_cannot_restore_original_holdout(self):
        train, _, _ = self.loaders()
        with self.assertRaisesRegex(ValueError, "differs"):
            heldout_indices(
                self.encoder, self.codes[::-1], self.accs[::-1], train.split_metadata,
                saved_split=True,
            )

    def test_forged_validation_overlap_is_rejected(self):
        train, _, _ = self.loaders()
        metadata = copy.deepcopy(train.split_metadata)
        metadata["validation_indices"] = metadata["training_indices"][:1]
        with self.assertRaisesRegex(ValueError, "overlap"):
            heldout_indices(self.encoder, self.codes, self.accs, metadata, saved_split=True)

    def test_duplicate_codes_or_features_are_rejected(self):
        for duplicate in [self.codes[0], {"x": 0, "ignored": 1}]:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                build_acc_data_loader(
                    self.encoder, self.codes + [duplicate], self.accs + [51.0], n_workers=0,
                )

    def test_invalid_datasets_are_rejected(self):
        for codes, labels in [([], []), (self.codes, [1.0]), (self.codes, [float("nan")] * 10)]:
            with self.assertRaises(ValueError):
                build_acc_data_loader(self.encoder, codes, labels, n_workers=0)
        for count in [0, -1, 11]:
            with self.assertRaises(ValueError):
                self.loaders(n_training_sample=count)

    def test_metrics_weight_final_batch_by_samples_and_restore_mode(self):
        model = torch.nn.Identity()
        model.train()
        dataset = RegDataset(torch.tensor([[0.0], [0.0], [1.0]]), torch.zeros(3))
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        trainer = AccPredictorTrainer(model, None, loader, "test", 0, device="cpu")
        metrics = trainer.evaluate()
        self.assertAlmostEqual(metrics["rmse_pp"], 100 / 3 ** 0.5)
        self.assertAlmostEqual(metrics["acc5"], 2 / 3)
        self.assertEqual(metrics["n"], 3)
        self.assertTrue(model.training)

    def test_training_loader_cannot_be_evaluation_loader(self):
        train, _, _ = self.loaders()
        with self.assertRaisesRegex(ValueError, "Training loader"):
            AccPredictorTrainer(torch.nn.Identity(), train, train, "test", 0)

    def test_unsafe_cli_invocations_are_rejected_before_loading_models(self):
        from scripts.predictor_factory.accuracy_predictor import build_mlp_acc_predictor as build
        from scripts.predictor_factory.accuracy_predictor import eval_acc_predictor as evaluate
        cases = [
            (build, ["model", "config", "dataset", "out", "-e"]),
            (build, ["model", "config", "dataset", "out", "-ckpt", "old.pth"]),
            (build, ["model", "config", "dataset", "out", "-e", "-ckpt", "x", "-no_split"]),
            (evaluate, ["model", "config", "mlp", "dataset"]),
        ]
        for module, arguments in cases:
            with self.subTest(arguments=arguments), patch.object(sys, "argv", ["test", *arguments]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    module.parse_args()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
