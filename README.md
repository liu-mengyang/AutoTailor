# AutoTailor: Automatic and Efficient Adaptive Model Deployment for Diverse Edge Devices

Artifact Evaluation for ACM SIGOPS ATC 2026.

AutoTailor builds adaptive SuperNets from computation graphs and selects models for edge devices using accuracy and latency predictors.

## Artifact components

| Component | Location | Relation to the paper |
|---|---|---|
| Graph compiler and TailorIR | `autotailor/tir/`, `onnx2torch/` | Converts model computation graphs into adaptive SuperNets. |
| Predictors and model adaptation | `autotailor/predictor_factory/`, `autotailor/tailor/` | Implements accuracy and latency prediction and constrained SubNet selection. |
| Model configurations and training | `configs/`, `scripts/trainer/` | Defines the supported model families and SuperNet training procedures; prepared weights avoid retraining during AE. |
| Model deployment | `infra/` | Converts models and executes them on edge devices for latency profiling. |
| Baselines | `baselines/` | Contains AdaptiveNet and the paper's NestDNN* implementation for comparison. |
| Evaluation and plotting | `ae/supernet/`, `exp_scripts/` | Evaluates SubNet accuracy and on-device latency and plots accuracy–latency and accuracy–compute tradeoffs. |
| Reproduction environment | `docker/`, `ae/reproduce.sh` | Packages dependencies and coordinates evaluation on the provided servers through one command. |

## Reproducibility platform and access

Please contact us through the artifact-evaluation review system to request access to our servers. We will provide login details and the repository location. Docker environments are prepared on the servers; reviewers do not need to supply a GPU or phone.

On-device latency profiling uses a USB-connected OnePlus ACE 6 phone (Qualcomm SM8750, Android 16). Host-side preparation and accuracy evaluation use a server with an Intel Xeon Gold 5520+ CPU, 376 GiB RAM, and two NVIDIA A100 80 GB GPUs.

The phone used in the original paper is an older Samsung A54. We provide a newer phone for AE, so absolute latency and accuracy–latency tradeoffs may differ from the paper. The supplied measurement workflow uses NCNN FP32 on the phone.

## Models and data

To make evaluation practical within the available time, we will place our trained SuperNets and baseline models on the evaluation servers. The reproduction script uses the prepared model paths automatically; reviewers do not need to locate, download, or train weights.

## Reproduce results

Log in to the provided GPU server and enter the repository directory supplied with your access details. Run:

```bash
bash ae/reproduce.sh
```

The script checks the environment, evaluates SubNets, transfers the exported models to the phone server, measures on-device latency, and generates the accuracy–latency and accuracy–compute plots.

Each invocation creates a new run directory and prints its location. The `figures/` directory contains:

- `checkpoint-pareto.pdf` and `checkpoint-pareto.png`: measured frontiers.
- `measurements.csv`: accuracy, compute, and latency for each candidate and phone.
- `summary.json`: frontier points and evaluation settings.
