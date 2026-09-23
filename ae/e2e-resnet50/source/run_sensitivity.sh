#!/bin/bash
set -euo pipefail
src=/results/e2e-resnet50/source
out=/results/e2e-resnet50
assets=/assets/e2e-resnet50/autotailor
export PYTHONPATH=$src AUTOTAILOR_HOME=$src
python "$src/exp_scripts/prepare_view.py"
mkdir -p "$out/sensitivity-cache"
cd "$out/sensitivity-cache"
python "$src/scripts/predictor_factory/accuracy_predictor/collect_acc_sensitivity.py" "$assets/timm_resnet50.a1_in1k.onnx" "$src/configs/supernet/timm_resnet50.toml" "$out/sensitivity.json" --evaluator_config_path "$out/train10k/evaluator.json" --imagenet-train-root /datasets/imagenet/ILSVRC/Data/CLS-LOC/train --checkpoint "$assets/resnet50_fix_best.pth.tar" -b 128 --workers 8 --resume
