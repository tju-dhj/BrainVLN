#!/usr/bin/env python3
"""Inspect stored IL inputs and SPN saturation without updating a checkpoint.

Batch-statistics and feature-rescaling probes are counterfactual diagnostics,
not validated policies or autonomous navigation evaluations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from spikingnav.habitat_objectnav.il_trainer import imitation_loss
from spikingnav.habitat_objectnav.offline_experiment import collate
from spikingnav.habitat_objectnav.provenance import record_run, sha256
from spikingnav.habitat_objectnav.trainer import _prepare_rgb
from spikingnav.habitat_objectnav.trajectory import audit_trajectory, decode_rgb
from spikingnav.models.actor_critic import SpikingNavModel


def sample_batch(manifest, count, frames, seed):
    files = json.loads(manifest.read_text())['files']
    chosen = np.random.default_rng(seed).choice(len(files), min(count, len(files)), replace=False)
    items, audit = [], []
    for i in chosen:
        path = Path(files[int(i)])
        check = audit_trajectory(path)
        if not check['ok']:
            raise ValueError(f'Trajectory audit failed: {path}')
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d['metadata']))
            n = len(d['actions']); start = max(0, n - frames)
            rgb = decode_rgb(d, range(start, n))
            actions = d['gt_actions'][start:n].astype(np.int64)
            mask = torch.ones(len(actions), 1); mask[0] = 0
            items.append(dict(rgb=torch.from_numpy(rgb), goal=torch.full((len(actions),), int(d['goal'])),
                              action=torch.from_numpy(actions), mask=mask, valid=torch.ones(len(actions))))
            audit.append(dict(path=str(path), sha256=check['sha256'], steps=n, selected_start=start,
                              goal=int(d['goal']), category=meta['category'], checks=check['checks'],
                              rgb_shape=list(rgb.shape), rgb_range=[int(rgb.min()), int(rgb.max())],
                              action_counts=np.bincount(actions, minlength=6).tolist(),
                              final_distance=float(d['distance_to_goal'][-1]), depth_range=check['depth_normalized_range']))
    return collate(items), audit


def tensor_stats(t):
    t = t.detach().float()
    return dict(min=float(t.min()), max=float(t.max()), abs_mean=float(t.abs().mean()),
                std=float(t.std()), finite=bool(torch.isfinite(t).all()))


def policy_probe(model, features, batch, scale=1.0):
    r = (features.detach() * scale).requires_grad_()
    valid = batch['valid'].to(r.device); labels = batch['action'].to(r.device)
    u = model.initial_memory(r.shape[1], r.device); logits = []; states = []; pre_values = []
    raw_values = []; feature_grads = None
    for i in range(len(r)):
        u = torch.where(batch['mask'][i].to(r.device) > 0, u, torch.zeros_like(u))
        # Exact forward equations from the unchanged SPN; record before reset/clamp.
        pre = model.cfg.leak * u + model.spn.w_r(r[i]) + model.spn.w_h(u)
        raw = pre * (pre < model.cfg.threshold)
        log, _, next_u = model.spn.step(r[i], u)
        logits.append(log); states.append(next_u); pre_values.append(pre.detach()); raw_values.append(raw.detach())
        u = next_u
    logits = torch.stack(logits)
    loss = imitation_loss(logits, labels, valid)
    alternative = (F.cross_entropy(logits.reshape(-1, 6), labels.reshape(-1), reduction='none').view_as(valid) * valid).sum() / valid.sum()
    model.zero_grad(set_to_none=True); loss.backward()
    groups = {}
    for name in ['w_r', 'w_h', 'actor']:
        parameters = list(getattr(model.spn, name).parameters())
        groups[name] = sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None) ** .5
    active = valid.bool(); pre = torch.stack(pre_values)[active]; raw = torch.stack(raw_values)[active]
    state = torch.stack(states).detach()[active]; predictions = logits.detach().argmax(-1)[active]
    truth = labels[active]
    return dict(feature_scale=scale, valid_frames=int(valid.sum()), loss=float(loss.detach()),
                original_vs_new_loss_abs_diff=float((loss-alternative).detach().abs()),
                accuracy=float((predictions == truth).float().mean()),
                truth_counts=torch.bincount(truth, minlength=6).tolist(),
                prediction_counts=torch.bincount(predictions, minlength=6).tolist(),
                mean_probabilities=logits.detach().softmax(-1)[active].mean(0).tolist(),
                features=tensor_stats(r.detach()[active]), pre_reset_membrane=tensor_stats(pre),
                spike_fraction=float((pre >= model.cfg.threshold).float().mean()),
                lower_clamp_fraction=float((raw < -10*model.cfg.threshold).float().mean()),
                zero_state_fraction=float((state == 0).float().mean()),
                state_unique_values=int(state.unique().numel()),
                feature_gradient_norm=float(r.grad.norm()), feature_gradient_nonzero_fraction=float((r.grad[active] != 0).float().mean()),
                parameter_gradient_norms=groups)



@torch.no_grad()
def sensitivity_probe(model, x, goals):
    snapshots = {}
    handles = []
    def capture(name):
        def hook(module, inputs, output):
            # Final spiking substep for this layer, overwritten on each call.
            snapshots[name] = output.detach().clone()
        return hook
    for name, module in model.sse.named_modules():
        if name.endswith(('sn1', 'sn2')):
            handles.append(module.register_forward_hook(capture(name)))
    z = model.sse.encode_visual(x)
    for handle in handles: handle.remove()
    layer_variation = {name: {'std_across_images': float(v.std(dim=0).mean()),
                             'nonzero_fraction': float((v != 0).float().mean())}
                       for name, v in snapshots.items()}
    f = model.encode(x, goals)
    flipped = model.encode(x.flip(0), goals)
    changed_goal = model.encode(x, (goals + 1) % 6)
    return dict(raw_rgb_std_across_images=float(x.std(dim=0).mean()),
                compressed_visual_std_across_images=float(z.std(dim=0).mean()),
                fusion_max_change_rgb_swap=float((f-flipped).abs().max()),
                fusion_max_change_goal_swap=float((f-changed_goal).abs().max()),
                layer_variation=layer_variation)



def replay_inputs(model, paths, out):
    """Replay stored expert actions and compare observations before each action."""
    from habitat.core.env import Env
    from habitat.config import read_write
    from PIL import Image
    from spikingnav.habitat_objectnav.make_env import build_habitat_config, load_dataset, HM3D_CATEGORIES
    from spikingnav.habitat_objectnav.expert import _action_names
    from spikingnav.habitat_objectnav.trajectory import observation_rgb
    device = next(model.parameters()).device
    records = []
    for filename in paths:
        with np.load(filename, allow_pickle=False) as d:
            meta=json.loads(str(d['metadata'])); actions=d['actions'].astype(int)
            stored=decode_rgb(d,range(len(actions))); depth=d['depth_u16']; poses=d['positions']
            cfg=build_habitat_config(split=meta['split'],scenes=[meta['content_scene']],seed=meta['seed'],max_episode_steps=500)
            dataset=load_dataset(cfg)
            selected=[e for e in dataset.episodes if str(e.episode_id)==str(meta['episode_id'])]
            if len(selected)!=1: raise ValueError('Replay episode ID is not unique')
            ep=selected[0]
            with read_write(cfg):
                cfg.habitat.simulator.scene=ep.scene_id
                cfg.habitat.environment.iterator_options.shuffle=False
            env=Env(config=cfg.habitat,dataset=dataset)
            frames=[]; memory=model.initial_memory(1,device)
            try:
                env.episode_iterator=iter([ep]); obs=env.reset(); names=_action_names(env.task)
                assert names==meta['action_names']
                assert dataset.category_to_task_category_id=={v:k for k,v in enumerate(HM3D_CATEGORIES)}
                for i,action in enumerate(actions):
                    live=observation_rgb(np.asarray(obs['rgb'])[...,:3])
                    live_depth=np.rint(np.asarray(Image.fromarray(np.asarray(obs['depth']).squeeze()).resize((224,224),Image.Resampling.NEAREST))*65535).astype(np.uint16)
                    goal=int(np.asarray(obs['objectgoal']).item())
                    assert goal==int(d['goal'])
                    a=_prepare_rgb(torch.from_numpy(stored[i:i+1]).to(device),224)
                    b=_prepare_rgb(torch.from_numpy(live[None]).to(device),224)
                    raw=_prepare_rgb(torch.from_numpy(np.asarray(obs['rgb'])[None,...,:3].copy()).to(device),224)
                    with torch.no_grad():
                        la,_,next_memory=model(a,torch.tensor([goal],device=device),memory)
                        lb,_,_=model(b,torch.tensor([goal],device=device),memory)
                        lc,_,_=model(raw,torch.tensor([goal],device=device),memory)
                    memory=next_memory
                    frames.append(dict(frame=i,expert_action=int(action),student_stored=int(la[-1].argmax()),
                        student_live=int(lb[-1].argmax()),student_original_eval_preprocess=int(lc[-1].argmax()),
                        rgb_equal=bool(np.array_equal(live,stored[i])),rgb_max_abs=int(np.abs(live.astype(int)-stored[i].astype(int)).max()),
                        normalized_rgb_max_abs=float((a-b).abs().max()),logits_max_abs=float((la-lb).abs().max()),
                        depth_max_abs=int(np.abs(live_depth.astype(int)-depth[i].astype(int)).max()),
                        position_max_abs=float(np.abs(np.asarray(env.sim.get_agent_state().position)-poses[i]).max())))
                    obs=env.step(int(action))
                    if env.episode_over:
                        if i!=len(actions)-1: raise ValueError('Replay ended early')
                        break
                record=dict(path=filename,sha256=sha256(filename),action_names=names,steps=len(frames),
                            success=float(env.get_metrics()['success']),spl=float(env.get_metrics()['spl']),
                            final_distance=float(env.get_metrics()['distance_to_goal']),frames=frames)
                records.append(record)
                (out/'replay.json').write_text(json.dumps(records,indent=2))
                print('REPLAY',json.dumps({k:v for k,v in record.items() if k!='frames'}),flush=True)
            finally: env.close()
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True); p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=4); p.add_argument('--frames',type=int,default=24)
    p.add_argument('--replay',action='store_true',help='Replay one sampled train and one heldout expert episode')
    p.add_argument('--seed',type=int,default=718); p.add_argument('--device',default='cpu')
    args=p.parse_args(); root=Path(__file__).resolve().parents[1]
    run=Path(args.run).resolve(); out=Path(args.output).resolve()
    if not out.is_relative_to(root/'storage'): p.error('output must be within project storage')
    out.mkdir(parents=True,exist_ok=False); record_run(out,args)
    torch.set_num_threads(4); torch.manual_seed(args.seed); device=torch.device(args.device)
    batches={}; audits={}
    for name,manifest in [('train','scaled_train_files.json'),('heldout','validation_files.json')]:
        batches[name],audits[name]=sample_batch(run/manifest,args.samples,args.frames,args.seed)
    (out/'input_audit.json').write_text(json.dumps(audits,indent=2))
    model=SpikingNavModel(6,6,pretrained_ann=False).to(device)
    checkpoints=sorted((run/'scaled_baseline').glob('ckpt_steps_*.pt'))
    if not checkpoints: raise FileNotFoundError('No completed checkpoints')
    results=[]; versions={str(ck):sha256(ck) for ck in checkpoints}
    for checkpoint in checkpoints:
        payload=torch.load(checkpoint,map_location=device,weights_only=False)
        modes=['frozen_bn','batch_bn_counterfactual','batch_bn_no_recurrence_counterfactual'] if checkpoint==checkpoints[-1] else ['frozen_bn']
        for mode in modes:
            for split,batch in batches.items():
                model.load_state_dict(payload['model']); model.eval()
                if mode.startswith('batch_bn'):
                    for module in model.modules():
                        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.train()
                if mode=='batch_bn_no_recurrence_counterfactual':
                    with torch.no_grad(): model.spn.w_h.weight.zero_()
                t,n=batch['action'].shape
                x=_prepare_rgb(batch['rgb'].reshape(t*n,224,224,3).to(device),224).view(t,n,3,224,224)
                goal=batch['goal'].to(device)
                with torch.no_grad():
                    features=torch.stack([model.encode(x[i],goal[i]) for i in range(t)])
                sensitivity=sensitivity_probe(model,x[0],goal[0])
                weights={'recurrent_spectral_norm':float(torch.linalg.matrix_norm(model.spn.w_h.weight.detach(),ord=2)),
                         'sensory_spectral_norm':float(torch.linalg.matrix_norm(model.spn.w_r.weight.detach(),ord=2))}
                scales=[1.0,.01] if mode=='frozen_bn' and checkpoint==checkpoints[-1] else [1.0]
                for scale in scales:
                    result={'checkpoint':str(checkpoint),'updates':payload['updates'],'split':split,'mode':mode,
                            'input_rgb':tensor_stats(x),'sensitivity':sensitivity,'weights':weights,**policy_probe(model,features,batch,scale)}
                    results.append(result)
                    print(json.dumps(result),flush=True)
                    (out/'report.json').write_text(json.dumps(dict(args=vars(args),checkpoint_sha256=versions,results=results,
                       scope='8 sampled trajectory suffixes; counterfactuals are not validated policies; no checkpoint updated'),indent=2))
    if args.replay:
        model.load_state_dict(torch.load(checkpoints[-1],map_location=device,weights_only=False)['model']); model.eval()
        replay_inputs(model,[audits[name][0]['path'] for name in ['train','heldout']],out)
    print('OUTPUT',out,flush=True)


if __name__=='__main__': main()
