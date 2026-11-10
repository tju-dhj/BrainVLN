"""Validate fixed evaluation identity before entering the simulator."""
import numpy as np


def validate_episode_keys(rows):
    keys=[(r['content_scene'],str(r['episode_id'])) for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate evaluation episode keys; raw HM3D IDs are not canonical')


def resolve_selected_episodes(episodes,rows,content_scene):
    validate_episode_keys(rows)
    by_id={str(e.episode_id):e for e in episodes}
    if len(by_id)!=len(episodes): raise ValueError('Duplicate loaded episode IDs')
    selected=[]
    for row in rows:
        if row['content_scene']!=content_scene: continue
        ep=by_id[str(row['episode_id'])]
        if 'category' in row and ep.object_category!=row['category']: raise ValueError('Evaluation category mismatch')
        for field in ['start_position','start_rotation']:
            if field in row and not np.allclose(getattr(ep,field),row[field],rtol=0,atol=1e-7):
                raise ValueError('Evaluation start state mismatch')
        selected.append(ep)
    return selected
