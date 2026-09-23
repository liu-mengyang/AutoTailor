from pathlib import Path
import tempfile
import unittest

import torch

from autotailor.tailor.acc_predictors.sensitivity_checkpoint import load_shared_checkpoint


class SharedCheckpointTests(unittest.TestCase):
    def test_load_updates_existing_parameter_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            parameter = torch.nn.Parameter(torch.zeros(2))
            target = {"conv": {"weight": parameter}}
            torch.save({"epoch": 119, "state_dict": {"conv": {"weight": torch.tensor([2., 3.])}}}, path)
            metadata = load_shared_checkpoint(path, target)
            self.assertIs(target["conv"]["weight"], parameter)
            torch.testing.assert_close(parameter, torch.tensor([2., 3.]))
            self.assertEqual(metadata, {"epoch": 119, "operator_count": 1, "tensor_count": 1, "strict": True})

    def test_invalid_checkpoint_cannot_partially_change_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            target = {"conv": {"weight": torch.zeros(2), "bias": torch.zeros(1)}}
            invalid = [
                {},
                {"conv": {"weight": torch.ones(2)}},
                {"conv": {"weight": torch.ones(2), "bias": torch.ones(2)}},
                {"conv": {"weight": torch.ones(2), "bias": torch.tensor([float('nan')])}},
            ]
            for state in invalid:
                torch.save({"state_dict": state}, path)
                with self.assertRaises(ValueError):
                    load_shared_checkpoint(path, target)
                self.assertEqual(target["conv"]["weight"].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
