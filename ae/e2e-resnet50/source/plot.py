#!/usr/bin/env python3
"""Plot only completed official-validation rows with matching phone model hashes."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--serial',default='3B15AT002BW00000');a=p.parse_args()
root=a.root; rows=[]
for folder,name,file in [('autotailor','AutoTailor — AE preview','accuracy.json'),('adaptivenet','AdaptiveNet — sampled preview','validation.json'),('nestdnn','NestDNN* — recovered 40-epoch runs','accuracy.json')]:
    data=json.loads((root/folder/file).read_text())
    for item in data['results']:
        assert item['image_count']==50000
        assert abs(item['top1_percent']-100*item['top1_correct']/50000)<1e-10
        lat=json.loads((root/folder/'latency-fp32'/a.serial/item['id']/'result.json').read_text())
        assert lat['identity']['torchscript_sha256']==item['torchscript_sha256']
        assert lat['identity']['serial']==a.serial and lat['equivalence']['relative_l2']<1e-4 and lat['equivalence']['same_top1']
        assert lat['identity']['cooldown_seconds_per_run']==10
        values=[r['latency_ms'] for r in lat['repetitions']];assert len(values)==3
        rows.append({'method':folder,'label':name,'id':item['id'],'latency_ms':lat['latency_ms'],'min_ms':min(values),'max_ms':max(values),'top1_percent':item['top1_percent'],'top5_percent':item['top5_percent'],'image_count':50000,'torchscript_sha256':item['torchscript_sha256']})
assert set(r['method'] for r in rows)=={'autotailor','adaptivenet','nestdnn'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(9,5.6))
colors={'autotailor':'#1864ab','adaptivenet':'#e67700','nestdnn':'#2b8a3e'}
markers={'autotailor':'o','adaptivenet':'s','nestdnn':'^'}
for method in colors:
    data=sorted((r for r in rows if r['method']==method),key=lambda r:r['latency_ms']);front=[];best=-1
    for r in data:
        if r['top1_percent']>best:front.append(r);best=r['top1_percent']
    ax.scatter([r['latency_ms'] for r in data],[r['top1_percent'] for r in data],color=colors[method],alpha=.3,s=30,marker=markers[method])
    x=[r['latency_ms'] for r in front];y=[r['top1_percent'] for r in front]
    ax.errorbar(x,y,xerr=[[r['latency_ms']-r['min_ms'] for r in front],[r['max_ms']-r['latency_ms'] for r in front]],fmt=markers[method]+'-',color=colors[method],label=data[0]['label'],capsize=3,lw=2,ms=6)
ax.set_xlabel('Measured phone inference latency (ms)');ax.set_ylabel('ImageNet validation top-1 accuracy (%)')
ax.set_title('ResNet50 end-to-end AE preview',loc='left',fontweight='bold',pad=28)
ax.text(0,1.025,'PLQ110 / SM8750 · NCNN · 1 CPU thread · FP32',transform=ax.transAxes,color='#555',fontsize=10)
ax.grid(alpha=.18);ax.legend(loc='lower right',fontsize=9,frameon=True)
fig.text(.1,.035,'50,000 validation images on zlTarget-GPU; latency on the attached replacement phone.\nPer-method sampled frontiers; horizontal bars span three timing repetitions. Similar setup, not exact paper reproduction.',fontsize=8.5,color='#555')
fig.subplots_adjust(bottom=.2,top=.84,left=.1,right=.97)
for suffix in ['png','pdf']:fig.savefig(root/('e2e-resnet50-preview.'+suffix),dpi=220)
with (root/'figure-data.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(root/'figure-data.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
