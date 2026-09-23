#!/usr/bin/env python3
"""Convert checkpoint-backed exports, gate phone outputs, and time both target phones."""
import argparse, concurrent.futures, hashlib, json, os, re, shlex, statistics, subprocess, time
from pathlib import Path
import numpy as np
from ppadb.client import Client


PROFILE = False

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, data):
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(p)


def measure(serial, item, root, model, checker, converter):
    identifier = item['id']; dest = root/'latency-fp32'/serial/identifier; dest.mkdir(parents=True, exist_ok=True)
    weights = {s: sha(str(model)+s) for s in ['.param', '.bin']}
    identity = {'id': identifier, 'serial': serial, 'torchscript_sha256': item['torchscript_sha256'],
        'ncnn_sha256': weights, 'benchmark_sha256': sha(checker), 'converter_sha256': sha(converter),
        'precision': 'NCNN FP32; FP16 packed/storage/arithmetic disabled; PNNX fp16=0',
        'threads': 1, 'powersave': 2, 'warmup': 8, 'runs': 100, 'repetitions': 3,
        'cooldown_seconds_per_run': 2 if PROFILE else 10, 'runtime': 'NCNN 20240102, Android arm64 CPU'}
    result_path = dest/'result.json'
    if result_path.exists():
        old = json.loads(result_path.read_text()); assert old['identity'] == identity
        return old
    d = Client(host='127.0.0.1', port=int(os.getenv('ADB_PORT', '5037'))).device(serial)
    if d is None: raise RuntimeError('Phone unavailable: '+serial)
    remote = '/data/local/tmp/autotailor-ae/compound-'+identifier
    d.shell('mkdir -p '+remote)
    for src, name in [(checker,'bench'), (Path(str(model)+'.param'),'model.param'), (Path(str(model)+'.bin'),'model.bin')]:
        d.push(str(src), remote+'/'+name)
    d.shell('chmod 700 '+remote+'/bench')
    res = item['code']['resolution']
    args = ['./bench', '100', '1', '2', '-1', '0' if PROFILE else '1', 'model_name=model', f'shape=[{res},{res},3]']
    values, equivalence = [], None
    for rep in range(3):
        if PROFILE: time.sleep(2)
        temp = d.shell('dumpsys battery')
        # Logits are printed during warmup, outside the timed loop.
        cmd = 'cd '+remote+' && AE_CHECK_LOGITS=1 '+shlex.join(args)+' 2>&1; echo AE_EXIT:$?'
        started = time.time(); log = d.shell(cmd)
        (dest/f'run-{rep}.log').write_text(log)
        (dest/f'battery-{rep}.txt').write_text(temp)
        if 'AE_EXIT:0' not in log or re.search(r'AE_ERROR|failed|Segmentation fault', log, re.I):
            raise RuntimeError('Phone benchmark failed: '+str(dest))
        match = re.search(r'^AE_LOGITS (.+)$', log, re.M)
        if not match: raise RuntimeError('Missing logits: '+str(dest))
        actual = np.fromstring(match[1], sep=' ')
        expected = np.load(root/(identifier+'.reference.npy')).reshape(-1)
        assert actual.shape == expected.shape == (1000,) and np.isfinite(actual).all()
        relative_l2 = float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-12))
        equivalence = {'input': 'ones(1,3,resolution,resolution)', 'relative_l2': relative_l2,
            'max_absolute_error': float(np.max(np.abs(actual-expected))),
            'same_top1': bool(actual.argmax() == expected.argmax()),
            'scope': 'Single synthetic input conversion smoke check; not full phone accuracy validation'}
        write(dest/f'equivalence-{rep}.json', equivalence)
        assert relative_l2 < 1e-4 and equivalence['same_top1'], equivalence
        match = re.search(r'time_avg\s+([0-9.eE+-]+)', log)
        assert match
        latency = float(match[1]); assert np.isfinite(latency) and latency > 0
        values.append({'latency_ms': latency, 'start_unix': started, 'end_unix': time.time()})
    result = {'identity': identity, 'equivalence': equivalence, 'repetitions': values,
        'latency_ms': statistics.median(x['latency_ms'] for x in values),
        'latency_aggregation': 'median of three means, each over 100 inferences',
        'model': d.shell('getprop ro.product.model').strip(),
        'soc': d.shell('getprop ro.soc.model').strip(),
        'android': d.shell('getprop ro.build.version.release').strip()}
    write(result_path, result); print('PHONE_COMPLETE '+serial+' '+identifier, flush=True)
    return result


def main():
    global PROFILE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--tools', type=Path, default=Path('/results/compound-frontier/tools'))
    p.add_argument('--serial', required=True)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--partial', action='store_true', help='Skip incomplete model transfers; caller must check final count')
    a = p.parse_args(); root = a.root.resolve(); PROFILE = a.profile
    serials = [a.serial]
    result = json.loads((root/'accuracy.json').read_text())
    converter = a.tools/'pnnx-20240410-linux/pnnx'; checker = a.tools/'benchncnn_check_fp32'
    for item in result['results']:
        pt = root/item['torchscript']
        if not pt.exists() or not (root/(item['id']+'.reference.npy')).exists(): continue
        if a.partial and sha(pt) != item['torchscript_sha256']: continue
        assert sha(pt) == item['torchscript_sha256']
        prefix = root/(item['id'].replace('-', '_')+'.ncnn')
        stamp = root/(item['id']+'.conversion.json')
        identity = {'torchscript_sha256': sha(pt), 'converter_sha256': sha(converter), 'fp16': 0}
        if stamp.exists():
            previous = json.loads(stamp.read_text()); assert previous['identity'] == identity
            assert previous['outputs'] == {s: sha(str(prefix)+s) for s in ['.param', '.bin']}
        else:
            r = item['code']['resolution']
            cmd = [str(converter), str(pt), f'inputshape=[1,3,{r},{r}]', 'fp16=0']
            with (root/(item['id']+'.conversion.log')).open('w') as log:
                subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            write(stamp, {'identity': identity, 'command': cmd,
                          'outputs': {s: sha(str(prefix)+s) for s in ['.param', '.bin']}})
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(measure, serial, item, root, prefix, checker, converter)
                       for serial in serials]
            for f in futures: f.result()
    print('AVAILABLE_CANDIDATES_COMPLETE', flush=True)

if __name__ == '__main__': main()
