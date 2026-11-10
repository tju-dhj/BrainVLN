"""Explicit, checkpoint-recorded IL ablations; historical defaults are unchanged."""
import torch
from torch import nn
from spikingnav.models.actor_critic import SpikingNavModel

DEFAULT_OPTIONS = {'normalization': 'frozen_bn', 'spn_stability': 'none'}


def replace_bn(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            replacement = nn.GroupNorm(8, child.num_features, eps=child.eps, affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            replace_bn(child)


def make_model(options=None, pretrained_ann=False):
    options = {**DEFAULT_OPTIONS, **(options or {})}
    if options['normalization'] not in ('frozen_bn', 'adaptive_bn', 'group_norm'):
        raise ValueError('Unknown normalization')
    if options['spn_stability'] not in ('none', 'bounded'):
        raise ValueError('Unknown SPN stability option')
    model = SpikingNavModel(action_dim=6, num_categories=6, pretrained_ann=pretrained_ann)
    if options['normalization'] == 'group_norm':
        replace_bn(model.sse)
    if options['spn_stability'] == 'bounded':
        model.spn.sensory_norm = nn.LayerNorm(model.cfg.hidden_size, elementwise_affine=False)
        model.spn.sensory_scale = 0.5
        model.spn.recurrent_spectral_cap = 0.4
    return model


@torch.no_grad()
def project_recurrence(model):
    cap = getattr(model.spn, 'recurrent_spectral_cap', None)
    if cap is None:
        return None
    w = model.spn.w_h.weight
    norm = torch.linalg.matrix_norm(w, ord=2)
    if norm > cap:
        w.mul_(cap / norm)
    return float(norm)


@torch.no_grad()
def health(model, batch, device):
    """Fixed training observations only; never update BN statistics."""
    from spikingnav.habitat_objectnav.trainer import _prepare_rgb
    old = {module: module.training for module in model.modules()}
    model.eval()
    t, n = batch['action'].shape
    rgb = _prepare_rgb(batch['rgb'].reshape(t*n, 224, 224, 3).to(device), 224).view(t,n,3,224,224)
    goals = batch['goal'].to(device)
    x = rgb[0]
    z = model.sse.encode_visual(x)
    f = model.encode(x, goals[0])
    changed = model.encode(x.flip(0), goals[0])
    u = model.initial_memory(n, device); states=[]; predictions=[]
    for i in range(t):
        logits, _, u = model(rgb[i], goals[i], u, batch['mask'][i].to(device).unsqueeze(0))
        active = batch['valid'][i].to(device).bool()
        states.append(u[active]); predictions.append(logits[-1].argmax(-1)[active])
    state = torch.cat(states); prediction = torch.cat(predictions)
    result = {'compressed_visual_std': float(z.std(dim=0, unbiased=False).mean()),
              'rgb_swap_feature_max_change': float((f-changed).abs().max()),
              'fusion_abs_max': float(f.abs().max()), 'state_abs_max': float(state.abs().max()),
              'state_lower_clamp_fraction': float((state <= -10*model.cfg.threshold).float().mean()),
              'state_zero_fraction': float((state == 0).float().mean()),
              'recurrent_spectral_norm': float(torch.linalg.matrix_norm(model.spn.w_h.weight, ord=2)),
              'prediction_counts': torch.bincount(prediction,minlength=6).tolist()}
    for module, mode in old.items(): module.training = mode
    # Measure the current training mode without retaining any BN-statistics updates.
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    train_z = model.sse.encode_visual(x)
    train_f = model.encode(x, goals[0])
    train_changed = model.encode(x.flip(0), goals[0])
    result.update(train_compressed_visual_std=float(train_z.std(dim=0, unbiased=False).mean()),
                  train_rgb_swap_feature_max_change=float((train_f-train_changed).abs().max()),
                  train_eval_feature_max_diff=float((train_f-f).abs().max()))
    for name, value in model.named_buffers(): value.copy_(buffers[name])
    return result
