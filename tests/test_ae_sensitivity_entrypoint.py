import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SensitivityEntrypointTests(unittest.TestCase):
    def run_entrypoint(self, args, role="gpu"):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            env = dict(os.environ, AE_ROLE=role, PATH=directory + os.pathsep + os.environ["PATH"])
            return subprocess.run(["bash", "ae/accuracy.sh", *args], env=env,
                                  cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)

    def test_routes_collection_and_selection_to_matched_evaluator(self):
        result = self.run_entrypoint(["sensitivity", "/results/weights.json", "/results/selected.json"])
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertEqual(args[0], "exp_scripts/evaluate_training_only_sensitivity.py")
        self.assertEqual(args[args.index("--sensitivity") + 1], "/results/weights.json")
        self.assertEqual(args[args.index("--selection") + 1], "/results/selected.json")
        self.assertEqual(args[args.index("--imagenet-root") + 1], "/datasets/imagenet")

    def test_missing_selection_is_rejected(self):
        self.assertEqual(self.run_entrypoint(["sensitivity", "weights.json"]).returncode, 2)

    def test_phone_role_is_rejected(self):
        self.assertEqual(self.run_entrypoint(["sensitivity", "weights.json", "selected.json"], "phone").returncode, 2)
