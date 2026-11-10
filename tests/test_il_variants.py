"""Protect checkpoint compatibility and opt-in numerical constraints."""
import torch
from spikingnav.models.actor_critic import SpikingNavModel
from spikingnav.habitat_objectnav.il_variants import make_model, project_recurrence


def test_default_state_and_outputs_match_original():
    torch.set_num_threads(4)
    torch.manual_seed(81)
    original=SpikingNavModel(6,6,pretrained_ann=False).eval()
    restored=make_model().eval(); restored.load_state_dict(original.state_dict(),strict=True)
    x=torch.randn(2,3,224,224); goals=torch.tensor([0,3])
    with torch.no_grad():
        a=original(x,goals)[0]; b=restored(x,goals)[0]
    assert torch.equal(a,b)


def test_group_norm_train_eval_and_checkpoint_consistency():
    torch.set_num_threads(4)
    options={'normalization':'group_norm','spn_stability':'bounded'}
    model=make_model(options); project_recurrence(model)
    assert not any(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules())
    x=torch.randn(2,3,224,224); goals=torch.tensor([0,3])
    with torch.no_grad():
        model.train(); a=model(x,goals)[0]
        model.eval(); b=model(x,goals)[0]
        restored=make_model(options).eval(); restored.load_state_dict(model.state_dict())
        c=restored(x,goals)[0]
    assert torch.equal(a,b) and torch.equal(b,c)


def test_projection_bounds_recurrence_and_gradient_is_finite():
    torch.set_num_threads(4)
    model=make_model({'normalization':'group_norm','spn_stability':'bounded'})
    with torch.no_grad(): model.spn.w_h.weight.mul_(100)
    project_recurrence(model)
    assert float(torch.linalg.matrix_norm(model.spn.w_h.weight.detach(),ord=2))<=0.40001
    r=torch.randn(8,2,model.sse.out_dim,requires_grad=True)*1000
    r.retain_grad()
    logits,_,state=model.spn(r,model.initial_memory(2,torch.device('cpu')))
    logits.square().mean().backward()
    assert torch.isfinite(state).all() and torch.isfinite(r.grad).all()
    assert float(r.grad.abs().sum())>0


def test_health_probe_does_not_mutate_batchnorm_buffers_or_modes():
    from spikingnav.habitat_objectnav.il_variants import health
    torch.set_num_threads(4)
    model=make_model({'normalization':'adaptive_bn'}).train()
    buffers={k:v.clone() for k,v in model.named_buffers()}
    modes={id(m):m.training for m in model.modules()}
    batch={'rgb':torch.randint(0,256,(2,2,224,224,3),dtype=torch.uint8),
           'goal':torch.tensor([[0,1],[0,1]]),'action':torch.ones(2,2,dtype=torch.long),
           'valid':torch.ones(2,2),'mask':torch.tensor([[[0.],[0.]],[[1.],[1.]]])}
    result=health(model,batch,torch.device('cpu'))
    assert 'train_eval_feature_max_diff' in result
    assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
    assert all(m.training==modes[id(m)] for m in model.modules())
