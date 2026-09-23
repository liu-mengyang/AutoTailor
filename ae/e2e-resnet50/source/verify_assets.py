#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
root=Path('/home/lmy/AutoTailor-AE-results/e2e-resnet50')
rows=json.loads((root/'ASSETS.json').read_text())
for r in rows:
 p=Path(r['destination_path']);assert p.stat().st_size==r['bytes'],str(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 assert h.hexdigest()==r['sha256'],str(p)
 print('VERIFIED',p,flush=True)
(root/'asset-verification.json').write_text(json.dumps({'status':'complete','file_count':len(rows),'manifest_sha256':hashlib.sha256((root/'ASSETS.json').read_bytes()).hexdigest()},indent=2)+'\n')
