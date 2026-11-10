"""Versioned, pickle-free expert trajectories for reproducible offline IL."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

SCHEMA = 2
ACTIONS = ('stop', 'move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down')


class TrajectoryValidationError(ValueError):
    def __init__(self,audit,path):
        self.audit=audit; self.path=Path(path)
        super().__init__(str(audit))


def pack_rgb(frames):
    blobs = []
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((224, 224), Image.Resampling.BILINEAR).save(buf, format='JPEG', quality=85)
        blobs.append(np.frombuffer(buf.getvalue(), dtype=np.uint8))
    offsets = np.concatenate(([0], np.cumsum([len(b) for b in blobs]))).astype(np.int64)
    return np.concatenate(blobs), offsets


def decode_rgb(data, indices):
    blob, offsets = data['rgb_bytes'], data['rgb_offsets']
    return np.stack([np.asarray(Image.open(io.BytesIO(blob[offsets[i]:offsets[i+1]].tobytes())).convert('RGB')) for i in indices])


def observation_rgb(frame):
    """Apply exactly the stored RGB transform to a live observation."""
    blob,offsets=pack_rgb([frame])
    return decode_rgb({'rgb_bytes':blob,'rgb_offsets':offsets},[0])[0]


def write_trajectory(path, record, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    n = len(record['actions'])
    if not n or any(len(record[k]) != n for k in ['rgb', 'depth', 'gt_actions', 'positions', 'rotations', 'distance_to_goal']):
        raise ValueError('Observation/action sequence lengths differ')
    depth = np.stack(record['depth']).astype(np.float32)
    if not np.isfinite(depth).all() or depth.min() < 0 or depth.max() > 1.00001:
        raise ValueError('Expected finite normalized depth in [0,1]')
    rgb, offsets = pack_rgb(record['rgb'])
    depth = np.stack([np.asarray(Image.fromarray(d.squeeze()).resize((224,224), Image.Resampling.NEAREST)) for d in depth])
    meta = {**metadata, 'schema':SCHEMA, 'steps':n, 'rgb_encoding':'JPEG quality=85, 224x224',
            'depth_encoding':'uint16 normalized [0,1], nearest resize 224x224',
            'alignment':'obs_t and pose_t before executed action_t; terminal observation stored separately'}
    payload = dict(rgb_bytes=rgb, rgb_offsets=offsets, depth_u16=np.rint(depth*65535).astype(np.uint16),
                   actions=np.asarray(record['actions'], dtype=np.int16), gt_actions=np.asarray(record['gt_actions'], dtype=np.int16),
                   valid=np.ones(n,dtype=np.bool_), positions=np.asarray(record['positions'],dtype=np.float32),
                   rotations=np.asarray(record['rotations'],dtype=np.float32),
                   distance_to_goal=np.asarray(record['distance_to_goal'],dtype=np.float32),
                   goal=np.int64(metadata['goal']), metadata=np.asarray(json.dumps(meta,ensure_ascii=False)),
                   terminal_position=np.asarray(record['terminal_position'],dtype=np.float32))
    terminal_rgb, terminal_offsets = pack_rgb([record['terminal_rgb']])
    payload.update(terminal_rgb_bytes=terminal_rgb, terminal_rgb_offsets=terminal_offsets)
    d=np.asarray(record['terminal_depth']).squeeze()
    if not np.isfinite(d).all() or d.min()<0 or d.max()>1.00001:
        raise ValueError('Invalid terminal depth')
    payload['terminal_depth_u16']=np.rint(np.asarray(Image.fromarray(d.astype(np.float32)).resize((224,224),Image.Resampling.NEAREST))*65535).astype(np.uint16)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial')
    with tmp.open('xb') as f:
        np.savez_compressed(f,**payload)
    audit = audit_trajectory(tmp)
    if not audit['ok']:
        rejected=path.with_suffix('.rejected')
        if rejected.exists(): raise FileExistsError(rejected)
        os.replace(tmp,rejected)
        raise TrajectoryValidationError(audit,rejected)
    os.replace(tmp,path)
    return meta, audit


def audit_trajectory(path):
    with np.load(path,allow_pickle=False) as d:
        meta=json.loads(str(d['metadata'])); n=meta['steps']
        labels=d['gt_actions']; actions=d['actions']; names=meta['action_names']
        checks={'lengths':all(len(d[k])==n for k in ['depth_u16','actions','gt_actions','valid','positions','rotations','distance_to_goal']),
                'rgb_offsets':len(d['rgb_offsets'])==n+1 and int(d['rgb_offsets'][-1])==len(d['rgb_bytes']) and bool(np.all(np.diff(d['rgb_offsets'])>0)),
                'finite_poses':bool(np.isfinite(d['positions']).all() and np.isfinite(d['rotations']).all()),
                'label_range':bool(((labels>=0)&(labels<len(names))).all()),
                'expert_executed':bool(np.array_equal(actions,labels)),
                'stop_saved':bool(actions[-1]==names.index('stop')),
                'success':bool(meta['success']==1.0),
                'depth':d['depth_u16'].shape==(n,224,224),
                'finite_distance':bool(np.isfinite(d['distance_to_goal']).all())}
        # Decode all frames at collection time so broken encodings fail before training.
        frames=decode_rgb(d,range(n)); checks['rgb_shape']=frames.shape==(n,224,224,3)
        # Rotations and STOP must not translate the agent; blocked forward steps are allowed.
        next_positions=np.concatenate([d['positions'][1:],d['terminal_position'][None]],axis=0)
        moved=np.linalg.norm(next_positions-d['positions'],axis=1)
        nonforward=actions!=names.index('move_forward')
        checks['pose_action_alignment']=bool(np.all(moved[nonforward]<1e-4))
        return {'ok':all(checks.values()),'checks':checks,'steps':n,'bytes':Path(path).stat().st_size,
                'action_counts':np.bincount(actions,minlength=len(names)).tolist(),
                'depth_normalized_range':[float(d['depth_u16'].min()/65535),float(d['depth_u16'].max()/65535)],
                'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
