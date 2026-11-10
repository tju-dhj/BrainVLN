#!/usr/bin/env python3
"""Fix a policy-independent evaluation set using Habitat canonical episode IDs.

Raw HM3D content files may repeat episode_id=0. Habitat renumbers episodes
when loading each content file; selection must use that same loader.
"""
import argparse, json, random
from pathlib import Path
from spikingnav.habitat_objectnav.make_env import build_habitat_config,load_dataset,DEFAULT_EPISODE_ROOT,HM3D_CATEGORIES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True); p.add_argument('--scenes',default='')
    p.add_argument('--count',type=int,default=200); p.add_argument('--seed',type=int,default=23456)
    args=p.parse_args(); root=Path(__file__).resolve().parents[1]; out=Path(args.output).resolve()
    if not out.is_relative_to(root/'storage'): p.error('output outside project storage')
    scenes=args.scenes.split(',') if args.scenes else [f.name[:-8] for f in sorted((Path(DEFAULT_EPISODE_ROOT)/'val/content').glob('*.json.gz'))]
    groups=[]
    for scene in scenes:
        ds=load_dataset(build_habitat_config(split='val',scenes=[scene],seed=args.seed))
        ids=[str(e.episode_id) for e in ds.episodes]
        if len(ids)!=len(set(ids)): raise ValueError('Loader returned duplicate episode IDs')
        for category in HM3D_CATEGORIES:
            eps=[e for e in ds.episodes if e.object_category==category]
            random.Random(args.seed).shuffle(eps)
            if eps: groups.append((scene,eps))
    chosen=[]
    for i in range(max(map(lambda g:len(g[1]),groups))):
        for scene,eps in groups:
            if i>=len(eps): continue
            ep=eps[i]; chosen.append({'content_scene':scene,'episode_id':str(ep.episode_id),'category':ep.object_category,
                                     'start_position':list(ep.start_position),'start_rotation':list(ep.start_rotation)})
            if len(chosen)==args.count: break
        if len(chosen)==args.count: break
    if len(chosen)!=args.count: raise ValueError('Insufficient episodes')
    if len({(r['content_scene'],r['episode_id']) for r in chosen})!=len(chosen): raise ValueError('Duplicate selection')
    with out.open('x') as f: json.dump(chosen,f,indent=2)
    print(json.dumps({'output':str(out),'episodes':len(chosen),'scenes':len(scenes),'seed':args.seed}))

if __name__=='__main__': main()
