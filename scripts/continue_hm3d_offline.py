#!/usr/bin/env python3
"""Continue a completed budgeted collection with finite IL training and fixed validation."""
import argparse, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True); p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--collection-root',default=''); p.add_argument('--tag',default='continuation')
    p.add_argument('--class-balance',choices=['none','balanced'],default='none')
    p.add_argument('--evaluation-list',required=True); args=p.parse_args()
    root=Path(__file__).resolve().parents[1]; run=Path(args.run).resolve()
    if not run.is_relative_to(root/'storage'): p.error('run outside project storage')
    if args.epochs<1: p.error('epochs must be positive')
    if not args.tag.replace('_','').isalnum(): p.error('invalid tag')
    collection=Path(args.collection_root).resolve() if args.collection_root else run/'scale_200k'
    state_path=run/(args.tag+'_status.json')
    if state_path.exists(): raise FileExistsError(state_path)
    state={'pid':os.getpid(),'status':'waiting_for_collection','epochs':args.epochs,'started_unix':time.time(),'args':vars(args)}
    def save():
        tmp=state_path.with_suffix('.partial'); tmp.write_text(json.dumps(state,indent=2)); tmp.replace(state_path)
    def execute(cmd,log_name):
        state['command']=cmd; save()
        with (run/log_name).open('x') as log:
            rc=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc: raise RuntimeError(f'{log_name}: exit {rc}')
    save()
    try:
        while True:
            status=json.loads((collection/'status.json').read_text())
            if status['status']=='complete': break
            if status['status']!='running': raise RuntimeError(f"Collection did not finish its budget: {status['status']}")
            if time.time()-state['started_unix']>8*3600: raise TimeoutError('Collection wait exceeded 8 hours')
            os.kill(int(status['pid']),0)
            if Path('/proc',str(status['pid']),'stat').read_text().split()[2]=='Z': raise RuntimeError('Collection supervisor exited without final status')
            time.sleep(30)
        files=json.loads((run/'train_files.json').read_text())['files']
        for directory in status.get('data_directories',[str(collection/'coverage'),str(collection/'expansion')]):
            for line in (Path(directory)/'episodes.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row.get('audit',{}).get('ok'): files.append(row['path'])
        seen=set(); windows=0; frames=0; categories=set(); scenes=set()
        for path in files:
            with np.load(path,allow_pickle=False) as d:
                meta=json.loads(str(d['metadata'])); key=(meta['scene_id'],str(meta['episode_id']))
                if key in seen: raise ValueError(f'Duplicate training episode {key}')
                if meta['content_scene'] in status['heldout_scenes']: raise ValueError('Held-out scene leaked')
                if meta['split']!='train' or meta['success']!=1: raise ValueError('Invalid training split/status')
                seen.add(key); n=len(d['gt_actions']); frames+=n; windows+=math.ceil(n/16)
                categories.add(meta['category']); scenes.add(meta['content_scene'])
        if len(scenes)!=status['scene_count'] or len(categories)!=6: raise ValueError('Scene/category coverage incomplete')
        if shutil.disk_usage(run).free<12e9: raise RuntimeError('Less than 12 GB for checkpoints/reserve')
        manifest=run/'scaled_train_files.json'
        with manifest.open('x') as f: json.dump({'files':files,'unique_frames':frames,'scenes':len(scenes),'categories':sorted(categories)},f,indent=2)
        updates_per_epoch=math.ceil(windows/32); updates=args.epochs*updates_per_epoch
        state.update(status='training',unique_frames=frames,episodes=len(files),updates=updates,updates_per_epoch=updates_per_epoch); save()
        execute(['bash','scripts/hm3d_experiment.sh','train',
                 f'data={manifest}',f'validation_data={run/"validation_files.json"}',f'output={run/"scaled_baseline"}',
                 f'epochs={args.epochs}','batch_size=32','window=16','context=8','calibration_batches=128',
                 'save_every=100','log_every=100',f'class_balance={args.class_balance}'],'scaled_baseline.log')
        summary=json.loads((run/'scaled_baseline/summary.json').read_text())
        if not summary['reload_logits_equal']: raise ValueError('Checkpoint reload check failed')
        state.update(status='evaluating',checkpoint=summary['checkpoint']); save()
        execute(['bash','scripts/hm3d_experiment.sh','eval','--output',str(run/'scaled_eval'),'--split','val',
                 '--checkpoint',summary['checkpoint'],'--episode-list',args.evaluation_list],'scaled_eval.log')
        state.update(status='complete',evaluation=json.loads((run/'scaled_eval/summary.json').read_text()))
    except Exception as exc:
        state.update(status='stopped',error=str(exc)); raise
    finally:
        state['elapsed_seconds']=time.time()-state['started_unix']; save()

if __name__=='__main__': main()
