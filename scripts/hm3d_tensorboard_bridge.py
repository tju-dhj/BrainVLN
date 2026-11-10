#!/usr/bin/env python3
"""Mirror immutable JSON results and append-only training logs into TensorBoard.

This observer never imports the policy or modifies training/checkpoints. Step on
training plots is optimizer update; collection plots use attempted episodes.
JSON logs do not contain per-update timestamps, so wall time is export time.
"""
import argparse
import json
import math
import time
from pathlib import Path
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

TRAIN_RUNS=['overfit_bn_consistent','overfit_bn_resume','baseline','baseline_balanced','scaled_baseline']
EVAL_RUNS={'baseline':'baseline_eval_aligned','baseline_balanced':'baseline_balanced_eval','scaled_baseline':'scaled_eval'}
ACTIONS=['stop','forward','left','right','look_up','look_down']


class Tail:
    def __init__(self): self.offsets={}
    def read(self,path):
        if not path.exists(): return []
        with path.open('rb') as stream:
            stream.seek(self.offsets.get(path,0)); data=stream.read()
        end=data.rfind(b'\n')+1
        if not end: return []
        self.offsets[path]=self.offsets.get(path,0)+end
        return [json.loads(line) for line in data[:end].splitlines() if line]


def read_json(path):
    if not path.exists(): return None
    try: return json.loads(path.read_text())
    except json.JSONDecodeError: return None  # a summary may still be being written


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True); p.add_argument('--output',required=True)
    p.add_argument('--train-runs',nargs='+',default=TRAIN_RUNS)
    p.add_argument('--eval-map',default='',help='JSON mapping training run name to student eval directory')
    p.add_argument('--interval',type=float,default=5); p.add_argument('--once',action='store_true')
    args=p.parse_args(); root=Path(__file__).resolve().parents[1]
    run=Path(args.run).resolve(); out=Path(args.output).resolve()
    if not run.is_relative_to(root/'storage') or not out.is_relative_to(root/'storage'): p.error('paths must be under project storage')
    if args.interval<1: p.error('interval must be at least one second')
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps({**vars(args),'training_x_axis':'optimizer updates',
        'collection_x_axis':'attempted episodes','wall_time':'export time, not original experiment timestamp',
        'evaluation':'final completed student evaluations only; no fabricated intermediate SR'},indent=2))
    eval_runs=json.loads(args.eval_map) if args.eval_map else EVAL_RUNS
    writers={}; tail=Tail(); written=set(); first={}; last={}; rows_seen={}
    attempted=0; saved=0; frames=0; byte_count=0
    def scalar(name,tag,value,step):
        if value is None or not math.isfinite(float(value)): return
        if name not in writers: writers[name]=EventFileWriter(str(out/name),flush_secs=5)
        writers[name].add_event(Event(wall_time=time.time(),step=int(step),
            summary=Summary(value=[Summary.Value(tag=tag,simple_value=float(value))])))
    def scores(name,prefix,metrics,step):
        if not metrics: return
        for key in ['loss','accuracy','valid_steps']:
            if key in metrics: scalar(name,prefix+'/'+key,metrics[key],step)
        for action,value in zip(ACTIONS,metrics.get('per_action_recall',[])):
            scalar(name,prefix+'/recall_'+action,value,step)
    try:
        while True:
            for name in args.train_runs:
                directory=run/name; protocol=read_json(directory/'protocol.json') or {}
                for row in tail.read(directory/'metrics.jsonl'):
                    update=row['update']; first.setdefault(name,update-1); last[name]=update
                    rows_seen[name]=rows_seen.get(name,0)+1
                    if protocol.get('transition_c',1.0) not in (1,1.0): loss_tag='train/bc_loss_transition'
                    elif protocol.get('class_balance','none')!='none': loss_tag='train/bc_loss_weighted'
                    else: loss_tag='train/bc_loss'
                    scalar(name,loss_tag,row['loss'],update)
                    scalar(name,'train/bc_loss_unweighted',row.get('bc_loss_unweighted'),update)
                    for key in ['accuracy','gradient_norm','update_seconds','peak_cuda_mib','valid_frames']:
                        scalar(name,'train/'+key,row[key],update)
                    scalar(name,'train/processed_supervised_frames',row['effective_steps'],update)
                for row in tail.read(directory/'health.jsonl'):
                    for key,value in row.items():
                        if key!='update' and isinstance(value,(int,float)):
                            scalar(name,'health/'+key,value,row['update'])
                initial=read_json(directory/'initial_metrics.json')
                if initial and name in first and (name,'initial') not in written:
                    scores(name,'fixed_diagnostic',initial,first[name]); written.add((name,'initial'))
                summary=read_json(directory/'summary.json')
                if summary and (name,'final') not in written:
                    scores(name,'fixed_diagnostic',summary.get('final'),summary['updates'])
                    scores(name,'heldout_actions',summary.get('validation'),summary['updates'])
                    written.add((name,'final'))
                if name in eval_runs:
                    evaluation=read_json(run/eval_runs[name]/'summary.json')
                    if summary and evaluation and (name,'student_eval') not in written:
                        if evaluation['mode']!='student' or evaluation['attempts']!=evaluation['unique_episodes']:
                            raise ValueError('Evaluation is not a unique autonomous student evaluation')
                        for key in ['sr','spl','attempts','mean_episode_steps']:
                            scalar(name,'student_val/'+key,evaluation[key],summary['updates'])
                        written.add((name,'student_eval'))
            pointer=run/'active_collection.txt'
            collection=Path(pointer.read_text().strip()) if pointer.exists() else run/'scale_200k'
            status=read_json(collection/'status.json') or {}
            directories=status.get('data_directories',[str(collection/'coverage'),str(collection/'expansion')])
            for directory in directories:
                for row in tail.read(Path(directory)/'episodes.jsonl'):
                    attempted+=1
                    if row.get('audit',{}).get('ok'):
                        saved+=1; frames+=row['steps']; byte_count+=row['audit']['bytes']
                    for key,value in [('accepted_frames',frames),('saved_episodes',saved),('stored_gb',byte_count/1e9),('accepted_episode_fraction',saved/attempted)]:
                        scalar('collection','collection/'+key,value,attempted)
            for writer in writers.values(): writer.flush()
            progress={'updated_unix':time.time(),'training_updates_exported':rows_seen,
                      'last_optimizer_update':last,'collection_accepted_frames':frames,'collection_saved_episodes':saved}
            tmp=out/'bridge_status.partial'; tmp.write_text(json.dumps(progress,indent=2)); tmp.replace(out/'bridge_status.json')
            if args.once: break
            time.sleep(args.interval)
    finally:
        for writer in writers.values(): writer.close()

if __name__=='__main__': main()
