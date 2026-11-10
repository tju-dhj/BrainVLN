#!/usr/bin/env python3
"""Budgeted two-pass collection: scene coverage first, then additional unique episodes."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from spikingnav.habitat_objectnav.provenance import record_run
from spikingnav.habitat_objectnav.make_env import DEFAULT_EPISODE_ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True); p.add_argument('--target-steps',type=int,default=200000)
    p.add_argument('--prior-collections',nargs='*',default=[])
    p.add_argument('--exclude-manifests',nargs='*',default=[])
    p.add_argument('--heldout-scenes',default='3XYAD64HpDr'); p.add_argument('--reserve-gb',type=float,default=18)
    p.add_argument('--seed',type=int,default=54321); args=p.parse_args()
    root=Path(__file__).resolve().parents[1]; out=Path(args.output).resolve()
    if not out.is_relative_to(root/'storage'): p.error('output outside project storage')
    out.mkdir(parents=True,exist_ok=False); record_run(out,args)
    scenes=[f.name[:-8] for f in sorted((Path(DEFAULT_EPISODE_ROOT)/'train'/'content').glob('*.json.gz')) if f.name[:-8] not in args.heldout_scenes.split(',')]
    excluded=[]; retained_steps=0; retained_keys=set()
    for directory in args.prior_collections:
        for line in (Path(directory)/'episodes.jsonl').read_text().splitlines():
            row=json.loads(line); key=(row['content_scene'],str(row['episode_id']))
            excluded.append({'content_scene':key[0],'episode_id':key[1]})
            if row.get('audit',{}).get('ok') and key not in retained_keys:
                retained_steps+=row['steps']; retained_keys.add(key)
    for path in args.exclude_manifests:
        excluded.extend({'content_scene':r['content_scene'],'episode_id':r['episode_id']} for r in map(json.loads,Path(path).read_text().splitlines()))
    state={'pid':os.getpid(),'started_unix':time.time(),'status':'running','target_steps':args.target_steps,
           'heldout_scenes':args.heldout_scenes.split(','),'scene_count':len(scenes),'completed_passes':[], 'effective_steps':retained_steps,'retained_steps':retained_steps,
           'data_directories':args.prior_collections+[str(out/'coverage'),str(out/'expansion')]}
    def save():
        tmp=out/'status.partial'; tmp.write_text(json.dumps(state,indent=2)); tmp.replace(out/'status.json')
    try:
        for name,quota in [('coverage',1),('expansion',12)]:
            remaining=args.target_steps-state['effective_steps']
            if remaining<=0: break
            exclude_file=out/(name+'_exclude.json'); exclude_file.write_text(json.dumps(excluded))
            cmd=['bash','scripts/hm3d_experiment.sh','collect','--output',str(out/name),'--split','train',
                 '--scenes',','.join(scenes),'--per-category',str(quota),'--seed',str(args.seed),
                 '--target-steps',str(remaining),'--reserve-gb',str(args.reserve_gb),'--exclude-episodes',str(exclude_file)]
            state.update(active_pass=name,command=cmd); save()
            with (out/(name+'.log')).open('x') as log:
                rc=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT).returncode
            summary_path=out/name/'summary.json'; summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
            state['completed_passes'].append({'name':name,'returncode':rc,'summary':summary})
            state['effective_steps']+=summary.get('effective_steps',0)
            if rc: raise RuntimeError(f'{name} exited {rc}; inspect {name}.log; existing files retained')
            excluded.extend({'content_scene':r['content_scene'],'episode_id':r['episode_id']} for r in map(json.loads,(out/name/'episodes.jsonl').read_text().splitlines()))
        state['status']='complete' if state['effective_steps']>=args.target_steps else 'quota_exhausted'
    except Exception as exc:
        state.update(status='stopped',error=str(exc)); raise
    finally:
        state['elapsed_seconds']=time.time()-state['started_unix']; save()

if __name__=='__main__': main()
