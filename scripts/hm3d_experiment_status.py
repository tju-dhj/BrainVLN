#!/usr/bin/env python3
"""Read live experiment progress without importing Habitat or using the GPU."""
import argparse, json, shutil, time
from pathlib import Path


def rows(path):
    if not path.exists(): return []
    result=[]
    for line in path.read_text().splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: pass  # a live writer may be flushing the last line
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',default=''); args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    run=Path(args.run) if args.run else Path((root/'storage/current_hm3d_sft_run.txt').read_text().strip())
    result={'run':str(run),'free_gb':round(shutil.disk_usage(run).free/1e9,2)}
    for name in ['overfit_bn_consistent','overfit_bn_resume','baseline','baseline_eval','baseline_eval_unique','baseline_eval_aligned','baseline_balanced','baseline_balanced_eval','scaled_baseline','scaled_eval']:
        directory=run/name; summary=directory/'summary.json'
        if (directory/'INVALID.json').exists(): result[name]={'status':'invalid','reason':json.loads((directory/'INVALID.json').read_text())}
        elif summary.exists(): result[name]={'status':'finished','summary':json.loads(summary.read_text())}
        elif directory.exists():
            metrics=rows(directory/'metrics.jsonl'); episodes=rows(directory/'episodes.jsonl')
            result[name]={'status':'no_final_summary','latest_update':metrics[-1] if metrics else None,'evaluated_episodes':len(episodes)}
    scale=Path((run/'active_collection.txt').read_text().strip()) if (run/'active_collection.txt').exists() else run/'scale_200k'
    state=json.loads((scale/'status.json').read_text()) if (scale/'status.json').exists() else {}
    allrows=[]
    for directory in state.get('data_directories',[str(scale/'coverage'),str(scale/'expansion')]): allrows.extend(rows(Path(directory)/'episodes.jsonl'))
    successful=[r for r in allrows if r.get('audit',{}).get('ok')]
    elapsed=time.time()-state.get('started_unix',time.time()); steps=sum(r['steps'] for r in successful)
    rate=(steps-state.get('retained_steps',0))/max(elapsed,1); stored=sum(r['audit']['bytes'] for r in successful)
    result['scale']={'directory':str(scale),'retained_steps':state.get('retained_steps',0),'status':state.get('status'),'active_pass':state.get('active_pass'),
                     'attempts':len(allrows),'saved_episodes':len(successful),'effective_steps':steps,
                     'scenes_attempted':len({r['content_scene'] for r in allrows}),
                     'scenes_saved':len({r['content_scene'] for r in successful}),
                     'categories_saved':sorted({r['category'] for r in successful}),'stored_gb':stored/1e9,
                     'elapsed_seconds':elapsed,'steps_per_second_wall':rate,
                     'estimated_remaining_minutes':max(0,state.get('target_steps',0)-steps)/rate/60 if rate>0 and state.get('status')=='running' else None}
    pointer=run/'active_continuation.txt'
    if pointer.exists():
        status=Path(pointer.read_text().strip())
        result['continuation']=json.loads(status.read_text()) if status.exists() else {'status':'not_started'}
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
