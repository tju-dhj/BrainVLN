#!/usr/bin/env python3
"""Collect success-checked geodesic experts, or evaluate a policy on fixed episodes."""
from __future__ import annotations
import argparse, json, os, random, shutil, time
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
from spikingnav.habitat_objectnav.make_env import build_habitat_config, load_dataset, DEFAULT_EPISODE_ROOT, HM3D_CATEGORIES
from spikingnav.habitat_objectnav.expert import query_shortest_path_expert, _action_names
from spikingnav.habitat_objectnav.trajectory import write_trajectory,observation_rgb,TrajectoryValidationError
from spikingnav.habitat_objectnav.episode_selection import validate_episode_keys,resolve_selected_episodes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True); p.add_argument('--split',default='train',choices=['train','val'])
    p.add_argument('--scenes',default=''); p.add_argument('--per-category',type=int,default=2)
    p.add_argument('--seed',type=int,default=12345); p.add_argument('--max-steps',type=int,default=500)
    p.add_argument('--attempt-multiplier',type=int,default=8); p.add_argument('--reserve-gb',type=float,default=10)
    p.add_argument('--target-steps',type=int,default=0); p.add_argument('--checkpoint',default='')
    p.add_argument('--exclude-episodes',default='',help='JSON list of content_scene/episode_id keys already collected')
    p.add_argument('--episode-list',default='',help='JSON list of scene/episode keys; evaluation is policy-independent')
    args=p.parse_args(); out=Path(args.output).resolve()
    root=Path(__file__).resolve().parents[1]
    if not out.is_relative_to(root/'storage'): p.error('output must be under project storage')
    out.mkdir(parents=True,exist_ok=False)
    from spikingnav.habitat_objectnav.provenance import record_run
    record_run(out,args)
    (out/'protocol.json').write_text(json.dumps({**vars(args),'policy_rgb_preprocessing':'PIL bilinear 224x224, JPEG quality 85 roundtrip, ImageNet normalization; identical to stored training images'},indent=2))
    import torch
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    scenes=args.scenes.split(',') if args.scenes else [f.name[:-8] for f in sorted((Path(DEFAULT_EPISODE_ROOT)/args.split/'content').glob('*.json.gz'))]
    excluded={(r['content_scene'],str(r['episode_id'])) for r in json.loads(Path(args.exclude_episodes).read_text())} if args.exclude_episodes else set()
    selected_keys=json.loads(Path(args.episode_list).read_text()) if args.episode_list else None
    if selected_keys is not None:
        validate_episode_keys(selected_keys)
        scenes=sorted({r['content_scene'] for r in selected_keys})
    model=None; device=torch.device('cuda:0')
    if args.checkpoint:
        from spikingnav.models.actor_critic import SpikingNavModel
        from spikingnav.config import DEFAULT_CONFIG
        from spikingnav.habitat_objectnav.il_variants import make_model
        payload=torch.load(args.checkpoint,map_location=device,weights_only=False)
        model=make_model(payload.get('model_options'),pretrained_ann=False).to(device)
        model.load_state_dict(payload['model']); model.eval()
    from habitat.core.env import Env
    from habitat.config import read_write
    from omegaconf import OmegaConf
    start=time.monotonic(); total_steps=0; successes=0; attempts=0; total_bytes=0; rows=[]
    manifests=(out/'episodes.jsonl').open('x')
    try:
        for content_scene in scenes:
            cfg=build_habitat_config(split=args.split,scenes=[content_scene],max_episode_steps=args.max_steps,seed=args.seed)
            ds=load_dataset(cfg)
            mapping=ds.category_to_task_category_id
            if mapping!={k:i for i,k in enumerate(HM3D_CATEGORIES)}: raise ValueError(f'category mismatch {mapping}')
            grouped=defaultdict(list)
            for e in ds.episodes:
                if (content_scene,str(e.episode_id)) not in excluded: grouped[e.object_category].append(e)
            rng=random.Random(args.seed)
            for v in grouped.values(): rng.shuffle(v)
            candidates=[]
            # Round-robin target categories within each scene, with fixed attempt caps.
            cap=args.per_category*(1 if model is not None else args.attempt_multiplier)
            for i in range(cap):
                for c in HM3D_CATEGORIES:
                    if i<len(grouped[c]): candidates.append(grouped[c][i])
            if selected_keys is not None:
                candidates=resolve_selected_episodes(ds.episodes,selected_keys,content_scene)
            counts=defaultdict(int)
            with read_write(cfg):
                cfg.habitat.simulator.scene=ds.episodes[0].scene_id
                cfg.habitat.environment.iterator_options.shuffle=False
            if not (out/'resolved_config.yaml').exists():
                (out/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg))
            # Avoid loading another scene after reset: one scene per environment.
            from spikingnav.habitat_objectnav.provenance import sha256
            annotation=Path(DEFAULT_EPISODE_ROOT)/args.split/'content'/(content_scene+'.json.gz')
            scene=Path(ds.episodes[0].scene_id)
            if not scene.is_absolute(): scene=root/scene
            versions={'annotation':str(annotation),'annotation_sha256':sha256(annotation),
                      'scene':str(scene.resolve()),'scene_sha256':sha256(scene)}
            navmesh=scene.with_suffix('.navmesh')
            if navmesh.exists(): versions['navmesh_sha256']=sha256(navmesh)
            with (out/'input_data.jsonl').open('a') as f: f.write(json.dumps(versions)+'\n')
            env=Env(config=cfg.habitat,dataset=ds)
            try:
                names=_action_names(env.task)
                if names!=['stop','move_forward','turn_left','turn_right','look_up','look_down']: raise ValueError(names)
                sensors=cfg.habitat.simulator.agents.main_agent.sim_sensors
                depth_cfg=OmegaConf.to_container(sensors['depth_sensor'],resolve=True)
                if not depth_cfg['normalize_depth']: raise ValueError('Normalized depth required')
                for ep in candidates:
                    if selected_keys is None and counts[ep.object_category]>=args.per_category: continue
                    if shutil.disk_usage(out).free<args.reserve_gb*1e9: raise RuntimeError('Disk reserve reached; existing episodes preserved')
                    if args.target_steps and total_steps>=args.target_steps: break
                    # A one-element iterator makes reset choose exactly this episode.
                    env.episode_iterator=iter([ep]); obs=env.reset()
                    assert str(env.current_episode.episode_id)==str(ep.episode_id)
                    attempts+=1; goal=int(np.asarray(obs['objectgoal']).item()); follower=None
                    if HM3D_CATEGORIES[goal]!=ep.object_category: raise ValueError('Goal sensor/category mismatch')
                    record={k:[] for k in ['rgb','depth','actions','gt_actions','positions','rotations','distance_to_goal']}
                    memory=model.initial_memory(1,device) if model is not None else None
                    reason='timeout'; t0=time.monotonic()
                    for step in range(args.max_steps):
                        if model is None:
                            action,valid,follower=query_shortest_path_expert(SimpleNamespace(habitat_env=env,config=cfg.habitat),follower)
                            if not valid: reason='invalid_expert'; break
                        else:
                            from spikingnav.habitat_objectnav.trainer import _prepare_rgb
                            rgb=torch.from_numpy(observation_rgb(np.asarray(obs['rgb'])[...,:3])[None]).to(device)
                            with torch.no_grad():
                                logits,_,memory=model(_prepare_rgb(rgb,224),torch.tensor([goal],device=device),memory)
                            action=int(logits[-1].argmax(-1).item())
                        state=env.sim.get_agent_state(); rotation=state.rotation
                        record['actions'].append(action); record['gt_actions'].append(action if model is None else -1)
                        record['positions'].append(np.asarray(state.position).copy())
                        record['rotations'].append([rotation.x,rotation.y,rotation.z,rotation.w])
                        record['distance_to_goal'].append(float(env.get_metrics()['distance_to_goal']))
                        if model is None:
                            record['rgb'].append(np.asarray(obs['rgb'])[...,:3].copy()); record['depth'].append(np.asarray(obs['depth']).copy())
                        obs=env.step(action)
                        if env.episode_over:
                            reason='stop' if action==names.index('stop') else 'timeout'; break
                    metrics=env.get_metrics(); success=float(metrics.get('success',0)); spl=float(metrics.get('spl',0))
                    row={'content_scene':content_scene,'scene_id':ep.scene_id,'episode_id':str(ep.episode_id),
                         'category':ep.object_category,'goal':goal,'success':success,'spl':spl,'steps':len(record['actions']),
                         'reason':reason,'seconds':time.monotonic()-t0,'actions':np.bincount(record['actions'],minlength=6).tolist(),
                         'distance_to_goal':float(metrics['distance_to_goal']) if np.isfinite(metrics['distance_to_goal']) else None}
                    if model is None and success==1 and reason=='stop':
                        state=env.sim.get_agent_state()
                        record.update(terminal_position=np.asarray(state.position),terminal_rgb=np.asarray(obs['rgb'])[...,:3],terminal_depth=np.asarray(obs['depth']))
                        path=out/'data'/args.split/content_scene/(str(ep.episode_id)+'.npz')
                        meta={**row,'action_names':names,'seed':args.seed,'split':args.split,'expert':'geodesic',
                              'depth_sensor':depth_cfg,'success_distance':float(cfg.habitat.task.measurements.success.success_distance)}
                        try:
                            _,audit=write_trajectory(path,record,meta)
                        except TrajectoryValidationError as error:
                            row.update(reason='trajectory_rejected',accepted_for_training=False,rejected_path=str(error.path),audit=error.audit)
                        else:
                            row.update(path=str(path),audit=audit,accepted_for_training=True)
                            total_bytes+=audit['bytes']; total_steps+=audit['steps']; counts[ep.object_category]+=1; successes+=1
                    elif model is not None:
                        total_steps+=len(record['actions']); counts[ep.object_category]+=1; successes+=success
                    rows.append(row); manifests.write(json.dumps(row,allow_nan=False)+'\n'); manifests.flush()
                    print(json.dumps({k:row[k] for k in ['content_scene','episode_id','category','steps','success','spl','reason']}),flush=True)
            finally: env.close()
            if args.target_steps and total_steps>=args.target_steps: break
    finally:
        manifests.close()
        elapsed=time.monotonic()-start
        summary={'mode':'student' if model is not None else 'expert','attempts':attempts,'successes':successes,
                 'sr':successes/attempts if attempts else 0,'environment_sr':sum(r['success'] for r in rows)/attempts if attempts else 0,'spl':sum(r['spl'] for r in rows)/attempts if attempts else 0,
                 'effective_steps':total_steps,'stored_bytes':total_bytes,'seconds':elapsed,
                 'effective_steps_per_second':total_steps/elapsed,'bytes_per_effective_step':total_bytes/total_steps if total_steps else 0,
                 'mean_episode_steps':sum(r['steps'] for r in rows)/attempts if attempts else 0,
                 'unique_episodes':len({(r['scene_id'],r['episode_id']) for r in rows})}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print('SUMMARY '+json.dumps(summary),flush=True)

if __name__=='__main__': main()
