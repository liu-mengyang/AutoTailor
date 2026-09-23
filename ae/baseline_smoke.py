#!/usr/bin/env python3
"""Functional checks with random weights only; no baseline accuracy is claimed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/opt/baselines')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', choices=['all', 'adaptivenet', 'nestdnn-star'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path, default=Path('/results/baselines'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.baseline == 'all':
        for name in ['adaptivenet', 'nestdnn-star']:
            subprocess.run([sys.executable, __file__, name, '--root', str(root), '--output-root', str(args.output_root), '--device', args.device], check=True)
        return
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    report = {'baseline': args.baseline, 'torch': torch.__version__,
              'scope': 'Random-weight structural/forward smoke only, not trained baseline performance'}
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('GPU not visible; run this image on the GPU server with --gpus')
        report['gpu'] = torch.cuda.get_device_name(0)
        report['cuda_check'] = torch.ones(4, device='cuda').sum().item()
    report['structural_test_device'] = 'cpu'
    if args.baseline == 'adaptivenet':
        sys.path.insert(0, str(root / 'adaptivenet/ondevice'))
        from mytimm.models import create_model
        model = create_model('resnet50', pretrained=False).eval()
        model.get_skip_blocks()
        code = model.generate_main_subnet()
        model.apply_subnet(code)
        model.eval()
        with torch.no_grad():
            output = model(torch.ones(1, 3, 64, 64))
        report['blocks'] = len(code)
    else:
        sys.path.insert(0, str(root / 'nestdnn-star'))
        import torch_pruning as tp
        import torchvision
        if not Path(tp.__file__).resolve().is_relative_to(root / 'nestdnn-star'):
            raise RuntimeError('Did not import the pinned NestDNN* source')
        model = torchvision.models.resnet18(weights=None).eval()
        x = torch.ones(1, 3, 32, 32)
        before = sum(p.numel() for p in model.parameters())
        pruner = tp.pruner.MetaPruner(model, x,
            importance=tp.importance.GroupTaylorImportance(),
            pruning_ratio=0.25, ignored_layers=[model.fc])
        model(x).sum().backward()
        pruner.step()
        after = sum(p.numel() for p in model.parameters())
        if not after < before:
            raise RuntimeError('Pruning did not reduce parameter count')
        with torch.no_grad():
            output = model(x)
        report.update(parameters_before=before, parameters_after=after)
    if tuple(output.shape) != (1, 1000) or not torch.isfinite(output).all():
        raise RuntimeError('Unexpected or non-finite model output')
    report.update(output_shape=list(output.shape), status='passed')
    print(json.dumps(report, indent=2))
    out = args.output_root; out.mkdir(parents=True, exist_ok=True)
    (out / (args.baseline + '-smoke.json')).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
