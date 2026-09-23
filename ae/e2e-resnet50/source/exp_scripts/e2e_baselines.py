#!/usr/bin/env python3
"""Recovered baseline weights: exact ImageNet evaluation and NCNN-ready exports."""
import argparse,copy,hashlib,json,math,random,sys
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from baseline_helpers import *
OUT=Path('/results/e2e-resnet50'); ASSETS=Path('/assets/e2e-resnet50'); DATA=Path('/datasets/imagenet')
MEAN=(.485,.456,.406); STD=(.229,.224,.225)
def protocol():
    import torchvision
    return {'torch':str(torch.__version__),'torchvision':torchvision.__version__,'gpu':torch.cuda.get_device_name(0),'tf32':False,'batch_size':128,'preprocessing':'Resize(256) bilinear, CenterCrop(224), ImageNet normalization','bn':'as stored in released/fine-tuned checkpoint','labels_sha256':sha(DATA/'LOC_val_solution.csv'),'synsets_sha256':sha(DATA/'LOC_synset_mapping.txt')}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def transform(): return transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
def loader(ds): return DataLoader(ds,batch_size=128,num_workers=8,pin_memory=True,drop_last=False)
def valset():
    syn=read_synsets(DATA/'LOC_synset_mapping.txt'); assert syn==sorted(p.name for p in (DATA/'ILSVRC/Data/CLS-LOC/train').iterdir() if p.is_dir()); rows,audit=read_validation_records(DATA/'ILSVRC/Data/CLS-LOC/val',DATA/'LOC_val_solution.csv',syn)
    return FlatImageNetValidation(rows,transform())
def export(model,item,dest):
    model=model.cpu().eval(); x=torch.randn(1,3,224,224)
    with torch.no_grad():
        if hasattr(model,'multiblocks'):
            parts=[model.conv1,model.bn1,model.relu,model.maxpool]; idx=0
            while idx<len(model.subnet):
                choice=model.subnet[idx]; parts.append(model.multiblocks[idx][choice]); idx+=choice+1
            compact=torch.nn.Sequential(*parts,model.global_pool,model.fc).eval()
            torch.testing.assert_close(compact(x),model(x)); model=compact
        traced=model if isinstance(model,torch.jit.ScriptModule) else torch.jit.trace(model,x)
        torch.testing.assert_close(traced(x),model(x)); reference=model(torch.ones_like(x)).numpy()
    p=dest/(item['id']+'.pt'); traced.save(str(p)); np.save(dest/(item['id']+'.reference.npy'),reference)
    return dict(item,torchscript=p.name,torchscript_sha256=sha(p))

def nest():
    dest=OUT/'nestdnn';dest.mkdir(exist_ok=True);target=dest/'accuracy.json'
    result=json.loads(target.read_text()) if target.exists() else {'scope':'Recovered NestDNN* 40-epoch training checkpoints; as-trained BN; 50k validation; no retraining','results':[]}
    result['protocol']=protocol()
    files=sorted((ASSETS/'nestdnn').glob('*/*_checkpoint.pt')); assert len(files)==7
    ds=valset()
    for pt in files:
        item={'id':'nest-'+pt.parent.name,'method':'NestDNN* (recovered, 40 epochs)','code':{'resolution':224},'source_sha256':sha(pt)}
        if any(x['id']==item['id'] for x in result['results']):continue
        model=torch.jit.load(str(pt),map_location='cpu').eval()
        best=next(pt.parent.glob('*_best.pth.tar')); state=torch.load(best,map_location='cpu',weights_only=True)['state_dict']
        actual=model.state_dict(); assert state.keys()==actual.keys(), (state.keys()-actual.keys(),actual.keys()-state.keys())
        assert all(torch.equal(state[k],actual[k]) for k in state),'TorchScript and best checkpoint tensors differ'
        item['best_checkpoint_sha256']=sha(best); item['best_state_matches_export']=True
        model=model.cuda(); metrics=evaluate(model,loader(ds),torch.device('cuda:0')); assert metrics['image_count']==50000
        result['results'].append(dict(export(model,item,dest),**metrics));write_json(target,result);print('COMPLETE',item['id'],metrics,flush=True)
        del model;torch.cuda.empty_cache()

def adaptive(final=False,serial='3B15AT002BW00000'):
    sys.path.insert(0,'/opt/baselines/adaptivenet/ondevice')
    from mytimm.models import create_model
    base=create_model('resnet50',pretrained=False).eval();base.get_skip_blocks();base.eval()
    checkpoint=ASSETS/'adaptivenet/resnet1epoch59acc69.pth';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    print(base.load_state_dict(state,strict=True),flush=True)
    dest=OUT/'adaptivenet';dest.mkdir(exist_ok=True);target=dest/('validation.json' if final else 'training-accuracy.json')
    result=json.loads(target.read_text()) if target.exists() else {'checkpoint_sha256':sha(checkpoint),'scope':'AE sampled AdaptiveNet preview; 16 seeded subnets; as-trained BN','results':[]}
    assert result['checkpoint_sha256']==sha(checkpoint)
    result['protocol']=protocol()
    random.seed(20260923);np.random.seed(20260923);codes=[base.generate_main_subnet()]
    while len(codes)<16:
        c=[int(v) for v in base.generate_random_subnet()]
        if c not in codes:codes.append(c)
    items=[{'id':'ad-'+hashlib.sha256(json.dumps(c).encode()).hexdigest()[:16],'method':'AdaptiveNet (AE sampled preview)','subnet':c,'code':{'resolution':224}} for c in codes]
    if final:
        train=json.loads((dest/'training-accuracy.json').read_text()); choices=[]
        for i in train['results']:
            lat=json.loads((dest/'latency-fp32'/serial/i['id']/'result.json').read_text());i['selection_latency_ms']=lat['latency_ms']
        for budget in [10,20,35,50,75,110]:
            feasible=[i for i in train['results'] if i['selection_latency_ms']<=budget]
            if feasible: choices.append(dict(max(feasible,key=lambda i:i['top1_percent']),budget_ms=budget))
        assert choices
        selection={'scope':'Selected using 8k train-image accuracy and target-phone latency, before official validation','selected':choices,'serial':serial}
        selection_path=dest/'selection.json'
        if selection_path.exists():assert json.loads(selection_path.read_text())==selection
        else:write_json(selection_path,selection)
        chosen={i['id'] for i in choices};items=[i for i in items if i['id'] in chosen];ds=valset()
        result['selection_sha256']=sha(selection_path)
    else:
        ds=datasets.ImageFolder(str(OUT/'train10k/val'),transform()); assert len(ds)==8000
        assert all(Path(p).resolve().is_relative_to(DATA/'ILSVRC/Data/CLS-LOC/train') for p,y in ds.samples)
    for item in items:
        if any(x['id']==item['id'] for x in result['results']):continue
        model=copy.deepcopy(base);model.apply_subnet(item['subnet']);model=model.cuda().eval()
        metrics=evaluate(model,loader(ds),torch.device('cuda:0'));assert metrics['image_count']==(50000 if final else 8000)
        if final:
            # Reuse the exact previously profiled export, verify evaluation matches it.
            pt=dest/(item['id']+'.pt');jit=torch.jit.load(str(pt)).cuda().eval()
            with torch.no_grad():torch.testing.assert_close(jit(torch.ones(1,3,224,224,device='cuda')),model(torch.ones(1,3,224,224,device='cuda')))
            saved=dict(item,torchscript=pt.name,torchscript_sha256=sha(pt));del jit
        else:saved=export(model,item,dest)
        result['results'].append(dict(saved,**metrics));write_json(target,result)
        if not final:write_json(dest/'accuracy.json',{'scope':'TRAINING IMAGE SCORES ONLY; awaiting selected official validation','results':result['results']})
        print('COMPLETE',item['id'],metrics,flush=True);del model;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['nest','adaptive','adaptive-final']);p.add_argument('--serial',default='3B15AT002BW00000');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20260923);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.action=='nest':nest()
    else:adaptive(a.action=='adaptive-final',a.serial)
