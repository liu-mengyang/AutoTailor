#!/usr/bin/env python3
"""Validate actual container dependencies and target/data access; never claims reproduction."""
import argparse, json, os, platform, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__); p.add_argument('role',choices=['phone','gpu']); a=p.parse_args()
import torch, torchvision, onnx, onnxruntime, onnx_graphsurgeon
from autotailor.tir.tailor_ir import TailorIR
report={'role':a.role,'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'onnx':onnx.__version__}
if a.role=='phone':
    # pure-python-adb talks to the existing server without replacing/restarting it.
    from ppadb.client import Client
    c=Client(host='127.0.0.1',port=int(os.getenv('ADB_PORT','5037')))
    serial=os.environ['ANDROID_SERIAL']; d=c.device(serial)
    if d is None: raise RuntimeError('Selected phone not connected: '+serial)
    report['device']={k:d.shell('getprop '+k).strip() for k in ['ro.product.model','ro.soc.model','ro.build.version.release','ro.build.fingerprint']}
    report['serial']=serial
else:
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable: verify NVIDIA Container Toolkit and --gpus')
    report['gpu']=torch.cuda.get_device_name(0)
    report['cuda_compute_check']=float((torch.ones(4, device='cuda') * 2).sum().item())
    root=Path('/datasets/imagenet')
    for f in ['LOC_val_solution.csv','LOC_synset_mapping.txt']:
        with (root/f).open() as stream: stream.readline()
    from PIL import Image
    val=root/'ILSVRC/Data/CLS-LOC/val'
    files=sorted(val.glob('*.JPEG'))
    if len(files)!=50000: raise RuntimeError(f'Expected 50,000 validation images, found {len(files)}')
    with Image.open(files[0]) as im: im.load()
    train=root/'ILSVRC/Data/CLS-LOC/train'
    classes=[x for x in train.iterdir() if x.is_dir()]
    if len(classes)!=1000: raise RuntimeError('Expected 1,000 training classes')
    report.update(validation_images=len(files),training_classes=len(classes))
report['status']='passed'; print(json.dumps(report,indent=2))
out=Path('/results'); out.mkdir(exist_ok=True)
(out/('preflight-'+a.role+'.json')).write_text(json.dumps(report,indent=2)+'\n')
