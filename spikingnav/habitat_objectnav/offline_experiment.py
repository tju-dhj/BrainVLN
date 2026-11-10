"""Measured, resumable offline BC on schema-v2 expert trajectories."""
from __future__ import annotations
import hashlib,json,queue,random,subprocess,sys,threading,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from spikingnav.config import DEFAULT_CONFIG
from spikingnav.habitat_objectnav.il_variants import make_model, project_recurrence, health
from spikingnav.models.actor_critic import SpikingNavModel
from spikingnav.habitat_objectnav.ckpt_keeper import checkpoint_steps, prune_student_checkpoints
from spikingnav.habitat_objectnav.trajectory import decode_rgb
from spikingnav.habitat_objectnav.trainer import _prepare_rgb


def trajectory_files(root):
    root=Path(root)
    files=[Path(p) for p in json.loads(root.read_text())['files']] if root.is_file() else sorted(root.rglob('*.npz'))
    if not files: raise ValueError(f'No trajectories under {root}')
    return files


class Windows(Dataset):
    """Deterministic coverage of every frame; optional preceding context.

    The recurrent state is reset at each context prefix. This is truncated
    sequence training, not exact full-episode state reconstruction.
    """
    def __init__(self,files,window=16,context=8,transition_c=3.0):
        self.files=list(files); self.items=[]; self.valid_steps=0; self.hashes={}
        self.window=window; self.context=context; self.transition_c=float(transition_c)
        self.action_counts=np.zeros(6,dtype=np.int64)
        if window<1 or context<0: raise ValueError('Invalid window/context')
        if not np.isfinite(self.transition_c) or self.transition_c<=0: raise ValueError('Invalid transition weight')
        for path in self.files:
            with np.load(path,allow_pickle=False) as d:
                meta=json.loads(str(d['metadata']))
                if meta['schema']!=2 or meta['success']!=1: raise ValueError(f'Not a validated successful trajectory: {path}')
                n=len(d['gt_actions']); self.valid_steps+=n
                if not np.all(d['valid']): raise ValueError('Invalid labels')
                self.action_counts+=np.bincount(d['gt_actions'],minlength=6)
            self.hashes[str(path)]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            self.items.extend((path,start,min(n,start+window)) for start in range(0,n,window))
    def __len__(self): return len(self.items)
    def __getitem__(self,index):
        path,start,end=self.items[index]; begin=max(0,start-self.context)
        with np.load(path,allow_pickle=False) as d:
            rgb=decode_rgb(d,range(begin,end)); actions=d['gt_actions'][begin:end].astype(np.int64)
            goal=int(d['goal']); previous=None if begin==0 else int(d['gt_actions'][begin-1])
        valid=np.ones(end-begin,dtype=np.float32); valid[:start-begin]=0
        mask=np.ones((end-begin,1),dtype=np.float32); mask[0]=0
        # w=c only when the expert action changes. Episode start has no predecessor, so w=1.
        # The predecessor may sit before this window; context frames stay available for the comparison.
        weights=np.ones(end-begin,dtype=np.float32); prior=previous
        for i,action in enumerate(actions):
            if prior is not None and int(action)!=prior: weights[i]=self.transition_c
            prior=int(action)
        return {'rgb':torch.from_numpy(rgb),'action':torch.from_numpy(actions),
                'goal':torch.full((end-begin,),goal,dtype=torch.long),'valid':torch.from_numpy(valid),
                'mask':torch.from_numpy(mask),'bc_weight':torch.from_numpy(weights)}


def collate(items):
    t=max(len(i['action']) for i in items); n=len(items)
    batch={'rgb':torch.zeros(t,n,224,224,3,dtype=torch.uint8),'action':torch.zeros(t,n,dtype=torch.long),
           'goal':torch.zeros(t,n,dtype=torch.long),'valid':torch.zeros(t,n),'mask':torch.zeros(t,n,1),
           'bc_weight':torch.zeros(t,n)}
    for i,item in enumerate(items):
        for key in batch: batch[key][:len(item['action']),i]=item[key]
    return batch


def forward_batch(model,batch,device,class_weights=None,transition_weights=False):
    t,n=batch['action'].shape
    rgb=_prepare_rgb(batch['rgb'].reshape(t*n,224,224,3).to(device),224).view(t,n,3,224,224)
    goal=batch['goal'].to(device); mask=batch['mask'].to(device)
    memory=model.initial_memory(n,device); logits=[]
    for j in range(t):
        pred,_,memory=model(rgb[j],goal[j],memory,mask[j].unsqueeze(0)); logits.append(pred[-1])
    logits=torch.stack(logits)
    labels=batch['action'].to(device); valid=batch['valid'].to(device)
    # Paper BC loss divides by the supervised step count T. A changed expert action keeps weight c
    # in the numerator, so its gradient is c times a repeated action and is not renormalized away.
    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,6),labels.reshape(-1),weight=class_weights,reduction='none').view(t,n)
    if transition_weights:
        loss=loss*batch['bc_weight'].to(device)
    denominator=valid.sum() if class_weights is None else (class_weights[labels]*valid).sum()
    loss=(loss*valid).sum()/denominator.clamp_min(1e-8)
    return loss,logits,labels,valid


@torch.no_grad()
def score(model,loader,device,max_batches=0):
    old=model.training; model.eval(); total=0; loss_sum=0.; confusion=torch.zeros(6,6,dtype=torch.int64)
    for i,batch in enumerate(loader):
        if max_batches and i>=max_batches: break
        loss,logits,labels,valid=forward_batch(model,batch,device)
        count=int(valid.sum()); total+=count; loss_sum+=float(loss)*count
        a=labels[valid.bool()].cpu(); b=logits.argmax(-1)[valid.bool()].cpu()
        confusion+=torch.bincount(a*6+b,minlength=36).reshape(6,6)
    model.train(old)
    return {'loss':loss_sum/max(total,1),'accuracy':float(confusion.diag().sum())/max(total,1),
            'valid_steps':total,'confusion':confusion.tolist(),
            'per_action_recall':[(float(confusion[i,i]/confusion[i].sum()) if confusion[i].sum() else None) for i in range(6)]}


def run_student_eval(repo, checkpoint, episode_list, dest):
    """Evaluate one checkpoint on the GPU. The caller keeps its own model on GPU."""
    dest=Path(dest); summary_path=dest/'summary.json'
    if summary_path.exists(): return json.loads(summary_path.read_text())
    cmd=[sys.executable,'-B',str(Path(repo)/'scripts'/'collect_hm3d_expert.py'),'--output',str(dest),
         '--split','val','--checkpoint',str(checkpoint),'--episode-list',str(episode_list)]
    completed=subprocess.run(cmd,cwd=str(repo))
    if completed.returncode!=0 or not summary_path.exists():
        raise RuntimeError(f'Student eval failed: {checkpoint}')
    return json.loads(summary_path.read_text())


def train(args):
    out=Path(args.output).resolve(); root=Path(__file__).resolve().parents[2]
    episode_list=str(getattr(args,'student_eval_list','') or '')
    keep_sr=int(getattr(args,'keep_sr_checkpoints',2))
    if episode_list:
        episode_list=str(Path(episode_list).resolve())
        if not Path(episode_list).is_file(): raise FileNotFoundError(f'Missing student eval list: {episode_list}')
        if keep_sr<1: raise ValueError('keep_sr_checkpoints must be positive')
    if not out.is_relative_to(root/'storage'): raise ValueError('output outside project storage')
    out.mkdir(parents=True,exist_ok=False)
    from spikingnav.habitat_objectnav.provenance import record_run
    record_run(out,args)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('GPU required for this experiment')
    ds=Windows(trajectory_files(args.data),args.window,args.context,args.transition_c)
    epochs=int(getattr(args,'epochs',0) or 0)
    if epochs<0: raise ValueError('epochs must be non-negative')
    updates_per_epoch=(len(ds)+args.batch_size-1)//args.batch_size
    if epochs>0: args.updates=epochs*updates_per_epoch
    print('EPOCH '+json.dumps({'windows':len(ds),'supervised_steps_per_epoch':int(ds.valid_steps),'updates_per_epoch':updates_per_epoch,'epochs':epochs,'updates':args.updates,'batch_size':args.batch_size}),flush=True)
    class_weights=None
    if args.class_balance in ('balanced','sqrt_balanced'):
        counts=ds.action_counts; present=counts>0; weights=np.zeros(6,dtype=np.float32)
        weights[present]=counts.sum()/(int(present.sum())*counts[present])
        if args.class_balance=='sqrt_balanced': weights[present]=np.minimum(np.sqrt(weights[present]),5.0)
        class_weights=torch.tensor(weights,device=device)
    generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=not args.fixed,num_workers=args.workers,
                      drop_last=False,collate_fn=collate,generator=generator)
    eval_loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate)
    validation=None
    if args.validation_data:
        vd=Windows(trajectory_files(args.validation_data),args.window,args.context,args.transition_c)
        def ids(files):
            found=set()
            for p in files:
                with np.load(p,allow_pickle=False) as d:
                    m=json.loads(str(d['metadata'])); found.add((m['scene_id'],m['episode_id']))
            return found
        if ids(ds.files)&ids(vd.files): raise ValueError('Train/validation episode overlap')
        validation=DataLoader(vd,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate)
    if args.fixed:
        # Include endpoints (STOP supervision), not only the first forward steps.
        indices=np.linspace(0,len(ds)-1,min(args.batch_size,len(ds)),dtype=int).tolist()
        fixed_batch=collate([ds[i] for i in indices]); eval_loader=[fixed_batch]
    else: fixed_batch=None
    model_options={'normalization':args.normalization,'spn_stability':args.spn_stability}
    model=make_model(model_options,pretrained_ann=not(args.resume or args.init_checkpoint or args.no_pretrained)).to(device)
    project_recurrence(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    update=0; seen=0; reference_loss=None
    def training_mode():
        model.train()
        # Batch statistics depend on episode minibatches and spiking internal time.
        # Keep the same normalization rule during training and autonomous inference.
        for module in model.modules():
            if args.normalization=='frozen_bn' and isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.eval()
    if args.resume:
        payload=torch.load(args.resume,map_location=device,weights_only=False)
        if payload.get('model_options',{'normalization':'frozen_bn','spn_stability':'none'})!=model_options: raise ValueError('Resume model options differ')
        if payload['dataset_sha256']!=ds.hashes: raise ValueError('Resume dataset differs')
        if payload['args'].get('class_balance','none')!=args.class_balance: raise ValueError('Resume class balancing differs')
        if payload['args'].get('transition_c',1.0)!=args.transition_c: raise ValueError('Resume transition weight differs')
        for key in ['window','context','batch_size','fixed','seed','calibration_batches']:
            if payload['args'][key]!=getattr(args,key): raise ValueError(f'Resume protocol differs: {key}')
        model.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer'])
        update=int(payload['updates']); seen=int(payload['steps'])
        reference_loss=payload.get('overfit_reference_loss')
        if args.fixed and reference_loss is None:
            reference_loss=json.loads((Path(args.resume).parent/'initial_metrics.json').read_text())['loss']
        torch.set_rng_state(payload['torch_rng'].cpu()); np.random.set_state(payload['numpy_rng']); random.setstate(payload['python_rng'])
        torch.cuda.set_rng_state_all([s.cpu() for s in payload['cuda_rng']])
        # Resume resets the epoch iterator; recorded explicitly, not bitwise replay.
    elif args.init_checkpoint:
        payload=torch.load(args.init_checkpoint,map_location=device,weights_only=False)
        if payload.get('model_options',{'normalization':'frozen_bn','spn_stability':'none'})!=model_options: raise ValueError('Initialization model options differ')
        model.load_state_dict(payload['model'])
        project_recurrence(model)
    elif args.normalization!='group_norm':
        # Calibrate on training observations only; never touch held-out labels or images.
        model.train()
        calibration_loader=[fixed_batch] if args.fixed else DataLoader(ds,batch_size=args.batch_size,shuffle=True,
            num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(args.seed+100))
        with torch.no_grad():
            for i,batch in enumerate(calibration_loader):
                if i>=args.calibration_batches: break
                forward_batch(model,batch,device)
    training_mode()
    config={**vars(args),'device':str(device),'model':asdict(DEFAULT_CONFIG),'unique_frames':ds.valid_steps,'updates_per_epoch':updates_per_epoch,'supervised_steps_per_epoch':int(ds.valid_steps),
            'windows':len(ds),'calibration_sampling':'fixed batch for overfit; otherwise shuffled train windows, seed+100','action_counts':ds.action_counts.tolist(),'class_weights':class_weights.tolist() if class_weights is not None else None,'transition_c':args.transition_c,'bc_loss':'sum(w_t * nll_t) / T; w_t=c when the expert action changes, else 1; evaluation stays unweighted cross-entropy','student_eval':f'GPU subprocess at each saved checkpoint; training model stays on GPU; keep {keep_sr} highest SR' if episode_list else 'off','normalization':args.normalization,'model_options':model_options,'dataset_sha256':ds.hashes,'resume_sampling':'new deterministic iterator; no mid-epoch cursor restore'}
    (out/'protocol.json').write_text(json.dumps(config,indent=2,default=str))
    initial=score(model,eval_loader,device,max_batches=0 if args.fixed else args.score_batches)
    if reference_loss is None: reference_loss=initial['loss']
    (out/'initial_metrics.json').write_text(json.dumps(initial,indent=2)); print('INITIAL '+json.dumps(initial),flush=True)
    health_indices=np.linspace(0,len(ds)-1,min(args.batch_size,len(ds)),dtype=int).tolist()
    health_batch=collate([ds[i] for i in health_indices])
    def write_health(step):
        result={'update':step,**health(model,health_batch,device)}
        with (out/'health.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
        print('HEALTH '+json.dumps(result),flush=True)
    if args.health_every: write_health(update)
    training_mode(); start=time.monotonic(); initial_seen=seen; start_update=update; total_train_seconds=0.; last_path=None
    logs=(out/'metrics.jsonl').open('x'); iterator=iter(loader)
    def save_checkpoint():
        path=out/f'ckpt_steps_{seen}.pt'
        payload={'model':model.state_dict(),'model_options':model_options,'optimizer':optimizer.state_dict(),'steps':seen,'updates':update,
                 'dataset_sha256':ds.hashes,'args':vars(args),'algo':'offline_il','overfit_reference_loss':reference_loss,
                 'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                 'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        if path.exists(): return path
        tmp=path.with_suffix('.partial'); torch.save(payload,tmp); tmp.replace(path)
        return path
    jobs=queue.Queue(); submitted=set(); protected=set(); eval_lock=threading.Lock(); worker=None; kept_sr=[]
    def prune_saved(protected_names):
        kept,deleted=prune_student_checkpoints(out,keep_sr,protect=protected_names)
        if episode_list: print('KEEP_SR '+json.dumps({'kept':[item['ckpt'] for item in kept],'deleted':deleted}),flush=True)
        return kept,deleted
    def eval_one(path):
        dest=out/'student_eval'/path.stem
        try:
            summary=run_student_eval(root,path,episode_list,dest)
        except Exception as exc:
            dest.mkdir(parents=True,exist_ok=True)
            (dest/'failed.json').write_text(json.dumps({'checkpoint':path.name,'error':str(exc)}))
            print('STUDENT_EVAL_FAILED '+json.dumps({'checkpoint':path.name,'error':str(exc)}),flush=True)
            return
        row={'checkpoint':path.name,'steps':checkpoint_steps(path.name),'sr':summary['sr'],'spl':summary['spl'],'episodes':summary['attempts']}
        with eval_lock:
            with (out/'student_eval.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            prune_saved(set(protected))
        print('STUDENT_EVAL '+json.dumps(row),flush=True)
    def eval_loop():
        while True:
            path=jobs.get()
            try:
                if path is None: return
                eval_one(path)
            finally: jobs.task_done()
    def submit_eval(path):
        if not episode_list or path is None: return
        with eval_lock:
            protected.clear(); protected.add(path.name)
        if path.name in submitted: return
        submitted.add(path.name); jobs.put(path)
    if episode_list:
        worker=threading.Thread(target=eval_loop,name='student-eval',daemon=True); worker.start()
    try:
        while update<args.updates:
            training_mode()
            if fixed_batch is None:
                try: batch=next(iterator)
                except StopIteration: iterator=iter(loader); batch=next(iterator)
            else: batch=fixed_batch
            torch.cuda.synchronize(); t0=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss,logits,labels,valid=forward_batch(model,batch,device,class_weights,transition_weights=True)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            plain=torch.nn.functional.cross_entropy(logits.detach().reshape(-1,6),labels.reshape(-1),reduction='none').view_as(valid)
            plain=(plain*valid).sum()/valid.sum().clamp_min(1e-8)
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True); optimizer.step()
            project_recurrence(model)
            torch.cuda.synchronize(); dt=time.monotonic()-t0; total_train_seconds+=dt
            count=int(valid.sum()); seen+=count; update+=1
            row={'update':update,'effective_steps':seen,'loss':float(loss.detach()),'bc_loss_unweighted':float(plain),
                 'accuracy':float(((logits.argmax(-1)==labels)*valid).sum()/valid.sum()),
                 'gradient_norm':float(norm),'update_seconds':dt,'valid_frames':count,
                 'peak_cuda_mib':torch.cuda.max_memory_allocated()/2**20}
            logs.write(json.dumps(row)+'\n'); logs.flush()
            if update==start_update+1 or update%args.log_every==0: print('TRAIN '+json.dumps(row),flush=True)
            if args.health_every and update%args.health_every==0: write_health(update)
            if update%args.save_every==0:
                last_path=save_checkpoint(); print('SAVED '+str(last_path),flush=True); submit_eval(last_path)
    finally:
        logs.close()
        if sys.exc_info()[0] is not None and worker is not None and worker.is_alive():
            jobs.put(None); jobs.join(); worker.join()
    last_path=save_checkpoint(); submit_eval(last_path)
    if worker is not None:
        jobs.put(None); jobs.join(); worker.join(); worker=None
    final=score(model,eval_loader,device,max_batches=0 if args.fixed else args.score_batches)
    val=score(model,validation,device) if validation is not None else None
    # Reload into a distinct model and compare predictions on the same observations.
    restored=make_model(model_options,pretrained_ann=False).to(device)
    payload=torch.load(last_path,map_location=device,weights_only=False); restored.load_state_dict(payload['model']); restored.eval(); model.eval()
    check_batch=fixed_batch if fixed_batch is not None else next(iter(eval_loader))
    with torch.no_grad():
        _,a,_,_=forward_batch(model,check_batch,device); _,b,_,_=forward_batch(restored,check_batch,device)
    equal=bool(torch.equal(a,b))
    if not equal: raise AssertionError('Checkpoint reload changed predictions')
    if episode_list:
        with eval_lock: protected.clear()
        kept_sr,_=prune_saved(set())
        if not last_path.exists() and kept_sr: last_path=out/kept_sr[0]['ckpt']
    summary={'initial':initial,'final':final,'validation':val,'updates':update,'effective_steps':seen,
             'kept_sr_checkpoints':[item['ckpt'] for item in kept_sr],
             'unique_frames':ds.valid_steps,'training_seconds':total_train_seconds,'elapsed_seconds':time.monotonic()-start,
             'effective_training_fps':(seen-initial_seen)/total_train_seconds if total_train_seconds else None,
             'checkpoint':str(last_path),'reload_logits_equal':equal,
             'fixed_overfit_pass':(final['loss']<reference_loss*.6 and final['accuracy']>=.8 and min(v for v in final['per_action_recall'] if v is not None)>=.5) if args.fixed else None}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print('SUMMARY '+json.dumps(summary),flush=True)
    if args.fixed and not summary['fixed_overfit_pass']: raise RuntimeError('Overfit gate not reached')
