#!/usr/bin/env bash
set -euo pipefail
[[ "${AE_ROLE:-}" == gpu ]] || { echo 'Run through the GPU container on the GPU server.' >&2; exit 2; }
if [[ "${1:-}" == sensitivity ]]; then
 [[ $# -ge 3 ]] || { echo 'Usage: bash ae/accuracy.sh sensitivity COLLECTION.json SELECTION.json [evaluator options]' >&2; exit 2; }
 collection=$2; selection=$3; shift 3
 exec python exp_scripts/evaluate_training_only_sensitivity.py \
   --sensitivity "$collection" --selection "$selection" \
   --imagenet-root /datasets/imagenet --gpu 0 \
   --output /results/sensitivity-selected-val50k.json "$@"
fi
[[ $# -ge 2 ]] || { echo 'Usage: bash ae/accuracy.sh resnet50 /assets/selection.json [evaluator options]' >&2; exit 2; }
model=$1; codes=$2; shift 2
case "$model" in
 resnet50) onnx=ofa_supernet_resnet50.onnx; cfg=ofa_resnet50.toml;;
 *) echo 'Unsupported model' >&2; exit 2;;
esac
view=/results/imagenet_train10k_view
# Build new container-local symlinks so author-machine paths cannot escape the dataset mount.
if [[ ! -f "$view/metadata.json" ]]; then
 python exp_scripts/build_stratified_imagenet_view.py /datasets/imagenet/ILSVRC/Data/CLS-LOC/train "$view"
fi
exec python "exp_scripts/evaluate_${model}_imagenet.py" --codes-json "$codes"  --onnx "/assets/models/onnx/$onnx" --supernet-config "configs/supernet/$cfg"  --calibration-root "$view/train" --calibration-manifest "$view/manifest.csv" --calibration-metadata "$view/metadata.json"  --validation-images /datasets/imagenet/ILSVRC/Data/CLS-LOC/val  --validation-labels /datasets/imagenet/LOC_val_solution.csv --synset-mapping /datasets/imagenet/LOC_synset_mapping.txt  --output "/results/${model}-accuracy.json" "$@"
