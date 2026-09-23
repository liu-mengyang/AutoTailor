"""Exercise both predictor entry points with isolated imports of each source tree."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECK = r'''
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from autotailor.tailor import adaptor
from autotailor.tailor.acc_predictors import sensitivity_provenance as provenance

# Confirm that imports resolve to the requested source tree, not the other copy.
assert Path(adaptor.__file__).resolve().is_relative_to(Path.cwd())
assert Path(provenance.__file__).resolve().is_relative_to(Path.cwd())
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train = root / 'CLS-LOC/train'
    train.mkdir(parents=True)
    images = [train / f'n00000001_{i}.JPEG' for i in range(2)]
    for image in images:
        image.write_bytes(b'image')
    valid = {
        'status': 'complete',
        'data_provenance': provenance.audit_training_sources(train, images[:1], images[1:]),
        'base_acc': [80.0], 'acc': {},
    }
    legacy = {'base_acc': [80.0], 'acc': {}}
    partial = copy.deepcopy(valid); partial['status'] = 'partial'
    validation = copy.deepcopy(valid)
    validation['data_provenance']['official_validation_used'] = True
    overlap = copy.deepcopy(valid)
    overlap['data_provenance']['sensitivity'] = copy.deepcopy(overlap['data_provenance']['bn_calibration'])
    tailor = SimpleNamespace(global_vars={}, stage_vars={}, block_vars={}, supercode={})
    weights = root / 'weights.json'
    for payload in [legacy, partial, validation, overlap]:
        weights.write_text(json.dumps(payload))
        with patch.object(adaptor, 'OptimizerFramework') as optimizer:
            try:
                adaptor.Adaptor(tailor).adapt(accuracy_metric='sensitivity', trans_weight_path=weights)
            except ValueError:
                pass
            else:
                raise AssertionError('Unsafe sensitivity file reached search')
            optimizer.assert_not_called()
    weights.write_text(json.dumps(valid))
    with patch.object(adaptor, 'OptimizerFramework') as optimizer:
        optimizer.return_value.optimize.return_value = ({}, 80.0, 0.0, [])
        result = adaptor.Adaptor(tailor).adapt(accuracy_metric='sensitivity', trans_weight_path=weights)
        optimizer.assert_called_once()
        assert result[1] == 80.0
'''


class AESupernetSensitivityTests(unittest.TestCase):
    def test_training_only_guard_in_both_source_trees(self):
        for source in [ROOT, ROOT / 'ae/supernet/source']:
            with self.subTest(source=source.relative_to(ROOT)):
                result = subprocess.run(
                    [sys.executable, '-c', CHECK], cwd=source,
                    env={**os.environ, 'AUTOTAILOR_HOME': str(source), 'PYTHONPATH': str(source)},
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
