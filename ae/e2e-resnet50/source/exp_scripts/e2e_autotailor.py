#!/usr/bin/env python3
"""AE preview: sensitivity-guided search with a phone-fitted latency surrogate."""
import argparse, copy, hashlib, json, math, os, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); os.environ['AUTOTAILOR_HOME']=str(ROOT)
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from imagenet_helpers import load_tailor, read_synsets, read_validation_records, FlatImageNetValidation, evaluate, write_json, MEAN, STD
from autotailor.tailor.acc_predictors.sensitivity_checkpoint import load_shared_checkpoint
from autotailor.tailor.acc_predictors.sensitivity_collection import calibrate_batch_norm
from autotailor.tailor.acc_predictors.sensitivity_provenance import require_training_sensitivity
from autotailor.tailor.acc_predictors.sensitivity_estimator import SensitivityEstimator
import autotailor.tir.globvar as globvar
OUT=Path('/results/e2e-resnet50'); ASSETS=Path('/assets/e2e-resnet50/autotailor')
SEED=20260923

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]
def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
def init():
    seed(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    cache=OUT/'pool-cache'; cache.mkdir(parents=True,exist_ok=True); os.chdir(cache)
    t=load_tailor(ASSETS/'timm_resnet50.a1_in1k.onnx',ROOT/'configs/supernet/timm_resnet50.toml')
    info=load_shared_checkpoint(ASSETS/'resnet50_fix_best.pth.tar',globvar.super_weights)
    t.tir.bind_weight(); print(info,flush=True); return t

def export(model,item,dest):
    dest.mkdir(parents=True,exist_ok=True); model=model.cpu().eval(); r=item['code']['resolution']
    x=torch.randn(1,3,r,r)
    with torch.no_grad():
        traced=torch.jit.trace(model,x); torch.testing.assert_close(traced(x),model(x))
        ref=model(torch.ones_like(x)).numpy()
    p=dest/(item['id']+'.pt'); traced.save(str(p)); np.save(dest/(item['id']+'.reference.npy'),ref)
    return dict(item,torchscript=p.name,torchscript_sha256=sha(p))

def pool(t):
    codes=[copy.deepcopy(t.supercode)]
    for r,w,d,e in [(128,.65,-2,.15),(160,.8,-1,.2),(192,1.,0,.25)]:
        c=copy.deepcopy(t.supercode); c['resolution']=r
        for k,v in [('width_mult',w),('reduce_depth',d)]:
            skip={i%len(c[k]) for i in t.stage_vars.get(k+'_skipped',[])}
            c[k]=[c[k][i] if i in skip else v for i in range(len(c[k]))]
        c['BottleneckResidualBlock']['expand_ratio']=[[e]*len(x) for x in c['BottleneckResidualBlock']['expand_ratio']]
        codes.append(c)
    while len(codes)<128:
        c=t.sample_subnet()
        if c not in codes: codes.append(c)
    entries=[]
    # Stratified random profile split; fixed before looking at accuracy or latency.
    indices=list(range(4,128)); random.Random(SEED).shuffle(indices)
    training=[0,1,2,3]+indices[:28]; heldout=indices[28:36]
    for i,c in enumerate(codes):
        t.transform(c); f,p=t.tir.count_flops_params()
        entries.append({'id':'at-'+ident(c),'method':'AutoTailor (AE preview)','code':c,'flops':float(f),'parameters':int(p),
                        'profile_split':'train' if i in training else 'test' if i in heldout else 'search'})
    manifest={'checkpoint_sha256':sha(ASSETS/'resnet50_fix_best.pth.tar'),'onnx_sha256':sha(ASSETS/'timm_resnet50.a1_in1k.onnx'),'seed':SEED,'candidates':entries}
    path=OUT/'pool.json'
    if path.exists(): assert json.loads(path.read_text())==manifest
    else: write_json(path,manifest)
    dest=OUT/'profiles'; dest.mkdir(exist_ok=True)
    target=dest/'accuracy.json'
    done=json.loads(target.read_text()) if target.exists() else {'results':[],'scope':'Latency profiling exports only; no accuracy labels'}
    for item in entries:
        if item['profile_split']=='search' or any(v['id']==item['id'] for v in done['results']): continue
        t.transform(item['code']); model=copy.deepcopy(t.tir.build()).cpu().eval()
        done['results'].append(export(model,item,dest)); write_json(target,done); print('EXPORT',item['id'],flush=True)

def features(item):
    x=np.log([item['flops'],item['parameters'],item['code']['resolution']])
    return x

def select(t,serial):
    s=json.loads((OUT/'sensitivity.json').read_text()); require_training_sensitivity(s)
    estimator=SensitivityEstimator(t,OUT/'sensitivity.json')
    pooldata=json.loads((OUT/'pool.json').read_text()); entries=pooldata['candidates']
    rows=[]
    for item in entries:
        if item['profile_split']=='search': continue
        p=OUT/'profiles'/'latency-fp32'/serial/item['id']/'result.json'
        latency=json.loads(p.read_text()); assert latency['identity']['serial']==serial
        rows.append((item,latency['latency_ms']))
    train=[(i,y) for i,y in rows if i['profile_split']=='train']; test=[(i,y) for i,y in rows if i['profile_split']=='test']
    x=np.array([features(i) for i,y in train]); mean=x.mean(0); scale=x.std(0); assert (scale>0).all()
    def design(x):
        z=(np.asarray(x)-mean)/scale
        return np.concatenate([np.ones((len(z),1)),z,np.array([z[:,i]*z[:,j] for i in range(3) for j in range(i,3)]).T],axis=1)
    X=design(x); y=np.log([y for i,y in train]); penalty=np.eye(X.shape[1])*.1; penalty[0,0]=0
    coef=np.linalg.solve(X.T@X+penalty,X.T@y)
    def predict(items): return np.exp(design([features(i) for i in items])@coef)
    measured=np.array([y for i,y in test]); predicted=predict([i for i,y in test])
    audit=[{'id':i['id'],'measured_ms':float(y),'predicted_ms':float(p)} for (i,y),p in zip(test,predicted)]
    predictor={'serial':serial,'type':'AE preview: quadratic ridge regression on log FLOPs, parameters, resolution; log latency target','mean':mean.tolist(),'scale':scale.tolist(),'coefficients':coef.tolist(),'ridge':.1,'train_count':len(train),'test_count':len(test),'test_mape_percent':float(np.mean(np.abs(predicted/measured-1))*100),'test_mae_ms':float(np.mean(np.abs(predicted-measured))),'test_max_relative_error_percent':float(np.max(np.abs(predicted/measured-1))*100),'heldout':audit,'training':[{'id':i['id'],'latency_ms':y} for i,y in train],'pool_sha256':sha(OUT/'pool.json')}
    write_json(OUT/'latency-predictor.json',predictor)
    for i,lat in zip(entries,predict(entries)):
        i['predicted_latency_ms']=float(lat); i['predicted_accuracy']=float(estimator.predict_accuracy(i['code']))
    selected=[]
    for budget in [10,20,35,50,75,110]:
        feasible=[i for i in entries if i['predicted_latency_ms']<=budget]
        if feasible:
            winner=max(feasible,key=lambda i:i['predicted_accuracy']); selected.append(dict(winner,budget_ms=budget))
    assert selected
    write_json(OUT/'selection.json',{'scope':'AE preview, 128 fixed seeded candidates, sensitivity accuracy and phone-fitted latency; selection completed before official validation','serial':serial,'sensitivity_sha256':sha(OUT/'sensitivity.json'),'predictor_sha256':sha(OUT/'latency-predictor.json'),'selected':selected,'predictions':entries})
    print('SELECTED',[(i['id'],i['budget_ms']) for i in selected],flush=True)

def evaluate_selected(t):
    selected=json.loads((OUT/'selection.json').read_text())
    dest=OUT/'autotailor'; dest.mkdir(exist_ok=True); target=dest/'accuracy.json'
    result=json.loads(target.read_text()) if target.exists() else {'selection_sha256':sha(OUT/'selection.json'),'results':[]}
    assert result['selection_sha256']==sha(OUT/'selection.json')
    dataset=Path('/datasets/imagenet'); synsets=read_synsets(dataset/'LOC_synset_mapping.txt')
    records,audit=read_validation_records(dataset/'ILSVRC/Data/CLS-LOC/val',dataset/'LOC_val_solution.csv',synsets)
    result['validation_audit']=audit
    for item in selected['selected']:
        if any(v['id']==item['id'] for v in result['results']): continue
        seed(); t.transform(item['code']); model=copy.deepcopy(t.tir.build()).cuda().eval(); r=item['code']['resolution']
        cal=datasets.ImageFolder(str(OUT/'train10k/train'),transforms.Compose([transforms.RandomResizedCrop(r),transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=32/255,saturation=.5),transforms.ToTensor(),transforms.Normalize(MEAN,STD)]))
        n=calibrate_batch_norm(model,DataLoader(cal,batch_size=128,num_workers=8,shuffle=False,generator=torch.Generator().manual_seed(SEED)),'cuda:0'); assert n==2000
        transform=transforms.Compose([transforms.Resize(math.ceil(r/.875)),transforms.CenterCrop(r),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
        loader=DataLoader(FlatImageNetValidation(records,transform),batch_size=128,num_workers=8,pin_memory=True,drop_last=False)
        metrics=evaluate(model,loader,torch.device('cuda:0')); assert metrics['image_count']==50000
        exported=export(model,item,dest); result['results'].append(dict(exported,**metrics,calibration_images=n)); write_json(target,result); print('COMPLETE',item['id'],metrics,flush=True)
        del model; torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pool','select','evaluate']);p.add_argument('--serial',default='3B15AT002BW00000');a=p.parse_args(); t=init()
    if a.action=='pool': pool(t)
    elif a.action=='select': select(t,a.serial)
    else: evaluate_selected(t)
