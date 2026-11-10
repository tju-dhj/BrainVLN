"""Immutable local provenance for uncommitted experiment sources."""
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def record_run(out, args):
    root=Path(__file__).resolve().parents[2]
    out=Path(out).resolve()
    if not out.is_relative_to(root/'storage'): raise ValueError('Provenance outside project storage')
    source=out/'source'; source.mkdir(exist_ok=False)
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True)
    hashes={}
    paths=[]
    for folder in ['scripts','spikingnav','tests','configs','docs']:
        paths.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.sh','.yaml','.yml','.json','.md'])
    paths.extend(root/name for name in ['README.md','AGENTS.md','pyproject.toml'] if (root/name).exists())
    for p in paths:
        rel=p.relative_to(root); dest=source/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest); hashes[str(rel)]=sha256(dest)
    (out/'worktree.patch').write_text(git('diff','--binary','HEAD'))
    (out/'git_status.txt').write_text(git('status','--short'))
    manifest={'utc':datetime.now(timezone.utc).isoformat(),'branch':git('branch','--show-current').strip(),
              'head':git('rev-parse','HEAD').strip(),'source_sha256':hashes,'source_status':'uncommitted working tree snapshot',
              'argv':sys.argv,'args':vars(args),'python':sys.executable,'python_version':platform.python_version(),
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'inputs':{}}
    for key in ['data','validation_data','episode_list','checkpoint','resume','init_checkpoint','exclude_episodes']:
        path=Path(getattr(args,key,''))
        if str(path)!='.' and path.is_file(): manifest['inputs'][key]={'path':str(path.resolve()),'sha256':sha256(path)}
    import torch, numpy
    manifest['environment']={'torch':torch.__version__,'numpy':numpy.__version__,'cuda':torch.version.cuda,
                             'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (out/'provenance.json').write_text(json.dumps(manifest,indent=2))
