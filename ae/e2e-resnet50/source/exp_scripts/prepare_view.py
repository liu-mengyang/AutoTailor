from pathlib import Path
import json, random
root=Path('/datasets/imagenet/ILSVRC/Data/CLS-LOC/train')
view=Path('/results/e2e-resnet50/train10k')
rows=[]
for folder in sorted(root.iterdir()):
    if not folder.is_dir(): continue
    files=sorted(folder.glob('*.JPEG')); random.Random(20260923+len(rows)).shuffle(files)
    assert len(files)>=10
    for i,f in enumerate(files[:10]):
        split='train' if i<2 else 'val'
        p=view/split/folder.name/f.name; p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists(): p.symlink_to(f)
        assert p.resolve()==f.resolve()
        rows.append({'split':split,'source':str(f)})
assert len(rows)==10000
(view/'manifest.json').write_text(json.dumps(rows,indent=2))
(view/'evaluator.json').write_text(json.dumps({'valdir':str(view)}))
