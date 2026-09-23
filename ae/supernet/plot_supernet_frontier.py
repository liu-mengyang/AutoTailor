#!/usr/bin/env python3
"""Join checkpoint accuracy to phone results by architecture and export hash."""
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def frontier(rows, cost):
    best = -1; vertices = []
    for r in sorted(rows, key=lambda r: (r[cost], -r['top1_percent'])):
        if r['top1_percent'] > best:
            best = r['top1_percent']; vertices.append(r)
    brute = {(r[cost],r['top1_percent']) for r in rows if not any(
        q[cost] <= r[cost] and q['top1_percent'] >= r['top1_percent'] and
        (q[cost] < r[cost] or q['top1_percent'] > r['top1_percent']) for q in rows)}
    assert brute == {(r[cost],r['top1_percent']) for r in vertices}
    return vertices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('accuracy',type=Path); p.add_argument('latency_root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); data=json.loads(a.accuracy.read_text())
    assert not data['provenance']['smoke']
    assert len(data['results']) == len(data['candidates']) == 16
    assert all(r['image_count']==50000 and r['calibration_images']==2000 for r in data['results'])
    a.output.mkdir(parents=True,exist_ok=True)
    serials=sorted(p.name for p in a.latency_root.iterdir() if p.is_dir())
    if len(serials) != 2:
        p.error('Expected measurement directories for exactly two phones.')
    rows=[]
    for serial in serials:
        for r in data['results']:
            path=a.latency_root/serial/r['id']/'result.json'; phone=json.loads(path.read_text())
            identity=phone['identity']
            assert identity['id']==r['id'] and identity['serial']==serial
            assert identity['torchscript_sha256']==r['torchscript_sha256']
            assert phone['equivalence']['same_top1'] and phone['equivalence']['relative_l2'] < 1e-4
            for rep in range(3):
                check=json.loads((path.parent/f'equivalence-{rep}.json').read_text())
                assert check['same_top1'] and check['relative_l2'] < 1e-4
            times=[x['latency_ms'] for x in phone['repetitions']]; assert len(times)==3
            rows.append({'serial':serial,'id':r['id'],'resolution':r['code']['resolution'],
                'top1_percent':r['top1_percent'],'top5_percent':r['top5_percent'],'flops':r['flops'],
                'latency_ms':phone['latency_ms'],'latency_min_ms':min(times),'latency_max_ms':max(times)})
    fig, axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
    for ax, serial in zip(axes[:2],serials):
        subset=[r for r in rows if r['serial']==serial]; pareto=frontier(subset,'latency_ms')
        for r in subset:
            ax.errorbar(r['latency_ms'],r['top1_percent'],
                xerr=[[r['latency_ms']-r['latency_min_ms']],[r['latency_max_ms']-r['latency_ms']]],
                fmt='o',color='#2376ab',alpha=.7,capsize=2)
        ax.scatter([r['latency_ms'] for r in pareto],[r['top1_percent'] for r in pareto],
            s=95,facecolors='none',edgecolors='black',label=f'Frontier ({len(pareto)} points)',zorder=5)
        ax.set(title=f'Phone {serial}',xlabel='CPU latency (ms; median of 3 means)'); ax.legend(fontsize=8)
        for r in subset: r['is_latency_frontier']=r['id'] in {p['id'] for p in pareto}
    unique=rows[:16]; pareto=frontier(unique,'flops'); ax=axes[2]
    ax.scatter([r['flops']/1e9 for r in unique],[r['top1_percent'] for r in unique],color='#2376ab')
    ax.scatter([r['flops']/1e9 for r in pareto],[r['top1_percent'] for r in pareto],s=95,
        facecolors='none',edgecolors='black',label=f'Frontier ({len(pareto)} points)')
    ax.set(title='Compute',xlabel='Reported compute (GFLOPs)'); ax.legend(fontsize=8)
    axes[0].set_ylabel('ImageNet validation top-1 accuracy (%)')
    for ax in axes: ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('ResNet50 SuperNet · 16 sampled architectures',fontsize=13)
    fig.text(.5,.018,'Accuracy: GPU FP32, 50,000 images  •  Phones: NCNN FP32, 1 thread, 3 × 100 runs  •  Error bars: range of run means',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    for ext in ['png','pdf']: fig.savefig(a.output/f'checkpoint-pareto.{ext}',dpi=200)
    with (a.output/'measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (a.output/'summary.json').write_text(json.dumps({'protocol':{
        k:v for k,v in data['provenance'].items() if not k.endswith('_sha256') and k != 'source_hashes'},
        'frontiers':{s:[r for r in rows if r['serial']==s and r['is_latency_frontier']] for s in serials}},indent=2)+'\n')

if __name__=='__main__': main()
