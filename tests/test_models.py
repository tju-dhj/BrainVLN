import torch

from spikingnav.config import DEFAULT_CONFIG
from spikingnav.models.ann_nav import ANNNavModel
from spikingnav.models.actor_critic import SpikingNavModel
from spikingnav.models.spn import SpikingPolicyNetwork
from spikingnav.models.sse import SpikingSensingEncoder
from spikingnav.nn.lif import LIF, reset_net
from spikingnav.utils.flops import count_parameters


def test_lif_reset_and_surrogate():
    neuron = LIF(leak=0.5, threshold=1.0)
    x = torch.ones(4, 8, requires_grad=True)
    s1 = neuron.forward(x)
    assert s1.shape == x.shape
    s1.sum().backward()
    assert x.grad is not None
    reset_net(neuron)
    assert neuron.membrane is None


def test_sse_shapes():
    sse = SpikingSensingEncoder(num_categories=12, pretrained_ann=False)
    rgb = torch.randn(3, 3, 224, 224)
    goal = torch.tensor([0, 4, 11])
    r = sse(rgb, goal)
    assert r.shape == (3, sse.out_dim)
    assert sse.out_dim == DEFAULT_CONFIG.feature_hw ** 2 * DEFAULT_CONFIG.visual_channels


def test_spn_episode_reset():
    spn = SpikingPolicyNetwork(input_dim=16, action_dim=4)
    r = torch.randn(5, 2, 16)
    u = torch.zeros(2, DEFAULT_CONFIG.hidden_size)
    masks = torch.ones(5, 2, 1)
    masks[3] = 0
    logits, values, u_t = spn(r, u, masks)
    assert logits.shape == (5, 2, 4)
    assert values.shape == (5, 2, 1)
    assert u_t.shape == (2, DEFAULT_CONFIG.hidden_size)

    long_r = torch.randn(128, 2, 16)
    long_masks = torch.ones(128, 2, 1)
    logits, values, u_t = spn(long_r, u, long_masks)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(u_t).all()
    assert u_t.abs().max() <= 10.0 * DEFAULT_CONFIG.threshold + 1e-4


def test_spikingnav_forward_backward():
    model = SpikingNavModel(action_dim=6, num_categories=12, pretrained_ann=False)
    rgb = torch.randn(2, 2, 224, 224, 3)
    goal = torch.randint(0, 12, (2, 2))
    logits, values, memory = model(rgb, goal)
    assert logits.shape == (2, 2, 6)
    assert values.shape == (2, 2, 1)
    (logits.mean() + values.mean()).backward()
    assert memory.shape == (2, DEFAULT_CONFIG.hidden_size)


def test_pointnav_goal_encoder():
    model = SpikingNavModel(action_dim=4, num_categories=None, pretrained_ann=False)
    rgb = torch.randn(1, 224, 224, 3)
    goal = torch.tensor([[0.4, -0.2]])
    logits, values, _ = model(rgb, goal)
    assert logits.shape[-1] == 4
    assert values.shape[-1] == 1


def test_annnav_forward():
    model = ANNNavModel(
        action_dim=6, num_categories=12, freeze_backbone=False, pretrained=False
    )
    rgb = torch.randn(1, 2, 224, 224, 3)
    goal = torch.randint(0, 12, (1, 2))
    logits, values, h = model(rgb, goal)
    assert logits.shape == (1, 2, 6)
    assert values.shape == (1, 2, 1)
    assert h.shape[0] == 1


def test_parameter_scale():
    snn = SpikingNavModel(action_dim=6, num_categories=12, pretrained_ann=False)
    ann = ANNNavModel(
        action_dim=6, num_categories=12, freeze_backbone=False, pretrained=False
    )
    snn_m = count_parameters(snn) / 1e6
    ann_m = count_parameters(ann) / 1e6
    # Paper: 12.1M vs 14.0M. Allow a modest implementation gap.
    assert 10.0 < snn_m < 14.5
    assert 11.0 < ann_m < 16.0
    assert snn_m < ann_m + 1.0
