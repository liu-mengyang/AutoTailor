#!/usr/bin/env python3
"""Checkpoint-backed TIMM ResNet50 sampled frontier: exact ImageNet counts and exports."""
import argparse, copy, hashlib, json, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['AUTOTAILOR_HOME'] = str(ROOT)
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from imagenet_helpers import (load_tailor, read_synsets, read_validation_records,
    FlatImageNetValidation, set_running_statistics, evaluate, MEAN, STD, write_json)
import autotailor.tir.globvar as globvar


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=Path('/assets/compound-supernet'))
    p.add_argument('--dataset', type=Path, default=Path('/datasets/imagenet'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    # Original parser writes generated ONNX blocks relative to cwd.
    cache = a.output / 'parser-cache'; cache.mkdir(exist_ok=True); os.chdir(cache)
    torch.set_num_threads(4)
    seed = 20260923
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = a.assets / 'resnetCompound_comp_swq_speedup_256_best.pth.tar'
    onnx = a.assets / 'timm_resnet50.onnx'
    cfg = ROOT / 'configs/supernet/timm_resnet50.toml'
    sources = {str(f.relative_to(ROOT)): sha(f) for folder in ['autotailor', 'onnx2torch', 'exp_scripts']
               for f in sorted((ROOT/folder).rglob('*.py'))}
    provenance = {'checkpoint_sha256': sha(checkpoint), 'onnx_sha256': sha(onnx),
        'config_sha256': sha(cfg), 'source_sha256': hashlib.sha256(canonical(sources).encode()).hexdigest(),
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'seed': seed, 'smoke': a.smoke, 'batch_size': a.batch_size,
        'calibration': 'first two sorted training JPEGs per class; Resize(256) bicubic, center crop',
        'validation': 'official validation set; Resize(256) bicubic, center crop at candidate resolution',
        'scope': '16 preselected candidates; sampled frontier, not exhaustive search',
        'tf32': False}
    t = load_tailor(onnx, cfg)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = saved['state_dict']
    if set(state) != set(globvar.super_weights):
        raise ValueError({'checkpoint_only': sorted(set(state)-set(globvar.super_weights)),
                          'graph_only': sorted(set(globvar.super_weights)-set(state))})
    count = changed = 0
    with torch.no_grad():
        for node, weights in globvar.super_weights.items():
            assert set(weights) == set(state[node]), node
            for key, value in weights.items():
                src = state[node][key].to(value.device)
                assert value.shape == src.shape and value.dtype == src.dtype, (node, key)
                changed += int(not torch.equal(value, src))
                value.copy_(src)  # Preserve all graph references to shared tensors.
                assert torch.equal(value, src)
                count += 1
    assert changed > 0, 'Checkpoint did not change ONNX initialization'
    provenance.update(checkpoint_epoch=saved['epoch'], shared_nodes=len(state),
                      restored_tensors=count, tensors_changed_from_onnx=changed)
    del saved, state
    # Four scale presets plus twelve seeded independent architecture samples.
    codes = []
    for r, w, d, e in [(128,.65,-2,.15), (160,.8,-1,.2), (192,1.,0,.25), (224,1.,0,.25)]:
        c = copy.deepcopy(t.supercode); c['resolution'] = r
        c['width_mult'] = [w]*len(c['width_mult'])
        c['reduce_depth'] = [d]*len(c['reduce_depth'])
        c['BottleneckResidualBlock']['expand_ratio'] = [[e]*len(x) for x in c['BottleneckResidualBlock']['expand_ratio']]
        # Tailor's transform honors skipped stages; encode them explicitly too.
        for name in ['width_mult', 'reduce_depth']:
            for idx in t.stage_vars.get(name+'_skipped', []):
                c[name][idx] = 1. if name == 'width_mult' else 0
        if canonical(c) not in [canonical(x) for x in codes]: codes.append(c)
    while len(codes) < 16:
        c = t.sample_subnet()
        if canonical(c) not in [canonical(x) for x in codes]: codes.append(c)
    candidates = [{'id': hashlib.sha256(canonical(c).encode()).hexdigest()[:16], 'code': c} for c in codes]
    write_json(a.output/'candidates.json', candidates)
    if a.smoke: candidates = [candidates[3]]
    synsets = read_synsets(a.dataset/'LOC_synset_mapping.txt')
    train = a.dataset/'ILSVRC/Data/CLS-LOC/train'
    assert sorted(x.name for x in train.iterdir() if x.is_dir()) == synsets
    records, audit = read_validation_records(a.dataset/'ILSVRC/Data/CLS-LOC/val', a.dataset/'LOC_val_solution.csv', synsets)
    calibration = []
    for target, name in enumerate(synsets):
        files = sorted((train/name).glob('*.JPEG'))[:2]
        assert len(files) == 2, name
        calibration.extend((f, target) for f in files)
    if a.smoke: calibration, records = calibration[:64], records[:100]
    provenance.update(validation_label_audit=audit, calibration_images=len(calibration), validation_images=len(records),
        labels_sha256=sha(a.dataset/'LOC_val_solution.csv'), synsets_sha256=sha(a.dataset/'LOC_synset_mapping.txt'))
    write_json(a.output/'calibration-manifest.json', [{'path': str(f.relative_to(a.dataset)), 'target': y} for f,y in calibration])
    target = a.output/'accuracy.json'
    result = {'provenance': provenance, 'candidates': candidates, 'results': []}
    if target.exists():
        result = json.loads(target.read_text())
        assert result['provenance'] == provenance and result['candidates'] == candidates, 'Resume fingerprint mismatch'
        for item in result['results']:
            assert item['image_count'] == len(records)
            assert sha(a.output/item['torchscript']) == item['torchscript_sha256']
    else: write_json(target, result)
    done = {r['id'] for r in result['results']}
    for candidate in candidates:
        if candidate['id'] in done: continue
        start = time.monotonic(); code = candidate['code']; r = int(code['resolution'])
        t.transform(code)
        model = copy.deepcopy(t.tir.build()).cuda().eval()
        flops, params = t.tir.count_flops_params()
        transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(r), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        def loader(rows): return DataLoader(FlatImageNetValidation(rows, transform), batch_size=a.batch_size,
            shuffle=False, num_workers=a.workers, pin_memory=True, drop_last=False)
        n = set_running_statistics(model, loader(calibration), torch.device('cuda:0'))
        assert n == len(calibration)
        metrics = evaluate(model, loader(records), torch.device('cuda:0'))
        assert metrics['image_count'] == len(records)
        model = model.cpu().eval()
        inputs = torch.randn(1, 3, r, r)
        with torch.no_grad():
            traced = torch.jit.trace(model, inputs)
            torch.testing.assert_close(traced(inputs), model(inputs))
        export = a.output/(candidate['id']+'.pt'); traced.save(str(export))
        # Reference output for conversion equivalence on deterministic ones input.
        with torch.no_grad(): reference = model(torch.ones(1, 3, r, r)).numpy()
        np.save(a.output/(candidate['id']+'.reference.npy'), reference)
        item = dict(candidate, **metrics, flops=float(flops), parameters=int(params),
            calibration_images=n, torchscript=export.name, torchscript_sha256=sha(export),
            elapsed_seconds=time.monotonic()-start)
        result['results'].append(item); write_json(target, result)
        print('COMPLETED '+json.dumps(item), flush=True)
        del model, traced; torch.cuda.empty_cache()
    print('ALL_CANDIDATES_COMPLETE', flush=True)

if __name__ == '__main__': main()
