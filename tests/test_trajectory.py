import numpy as np
import pytest
from spikingnav.habitat_objectnav.trajectory import write_trajectory, audit_trajectory, ACTIONS
from spikingnav.habitat_objectnav.offline_experiment import Windows,collate


def example(tmp_path):
    path=tmp_path/'episode.npz'
    record={'rgb':[np.full((12,12,3),i*50,dtype=np.uint8) for i in range(3)],
            'depth':[np.full((12,12,1),i*.25,dtype=np.float32) for i in range(3)],
            'actions':[1,2,0],'gt_actions':[1,2,0], 'positions':[[0,0,0],[0,0,.25],[0,0,.25]],
            'rotations':[[0,0,0,1]]*3,'distance_to_goal':[.25,0,0], 'terminal_position':[0,0,.25],
            'terminal_rgb':np.ones((12,12,3),dtype=np.uint8), 'terminal_depth':np.zeros((12,12,1))}
    meta={'goal':2,'success':1.,'action_names':list(ACTIONS),'scene_id':'train-scene','episode_id':'1'}
    return path,record,meta


def test_roundtrip_alignment_and_effective_count(tmp_path):
    path,r,m=example(tmp_path); write_trajectory(path,r,m)
    assert audit_trajectory(path)['ok']
    ds=Windows([path],window=2,context=1)
    assert ds.valid_steps==3
    batch=collate([ds[0],ds[1]])
    assert batch['valid'].sum().item()==3  # excludes both context and padding
    assert batch['action'][:,1].tolist()==[2,0]
    assert batch['rgb'][0,1].float().mean()>40
    with pytest.raises(FileExistsError): write_trajectory(path,r,m)


def test_bad_depth_is_rejected(tmp_path):
    path,r,m=example(tmp_path); r['depth'][0][:]=2.
    with pytest.raises(ValueError,match='normalized depth'): write_trajectory(path,r,m)
    assert not path.exists()


def test_transition_weight_uses_the_true_previous_action(tmp_path):
    path,r,m=example(tmp_path); write_trajectory(path,r,m)
    ds=Windows([path],window=2,context=0,transition_c=3)
    assert ds[0]['bc_weight'].tolist()==[1.,3.]          # episode start, then 1 -> 2
    assert ds[1]['action'].tolist()==[0] and ds[1]['bc_weight'].tolist()==[3.]  # predecessor is outside the window
    held=Windows([path],window=2,context=1,transition_c=3)[1]
    assert held['valid'].tolist()==[0.,1.] and held['bc_weight'].tolist()==[3.,3.]


def test_transition_loss_divides_by_step_count_and_ignores_padding():
    import torch
    from spikingnav.habitat_objectnav.offline_experiment import forward_batch
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.logits=torch.nn.Parameter(torch.tensor([.2,-.4,.7,-.1,.3,-.5]))
        def initial_memory(self,n,device): return torch.zeros(n,1,device=device)
        def forward(self,rgb,goal,memory,masks):
            return self.logits.view(1,1,6).expand(1,len(goal),6),None,memory
    model=Policy(); device=torch.device('cpu')
    batch={'rgb':torch.zeros(3,1,224,224,3,dtype=torch.uint8),'action':torch.tensor([[1],[1],[2]]),
           'goal':torch.zeros(3,1,dtype=torch.long),'mask':torch.ones(3,1,1),
           'valid':torch.tensor([[1.],[1.],[0.]]),'bc_weight':torch.tensor([[1.],[3.],[3.]])}
    loss,logits,labels,valid=forward_batch(model,batch,device,transition_weights=True)
    nll=torch.nn.functional.cross_entropy(logits.detach().reshape(-1,6),labels.reshape(-1),reduction='none')
    assert torch.allclose(loss,(nll[0]+3*nll[1])/2)
    loss.backward()
    assert model.logits.grad.abs().sum()>0
    model.zero_grad(set_to_none=True)
    plain,_,_,_=forward_batch(model,batch,device,transition_weights=False)
    assert torch.allclose(plain,(nll[0]+nll[1])/2)


def test_weighted_loss_ignores_padding_in_gradient():
    import torch
    from spikingnav.habitat_objectnav.offline_experiment import forward_batch
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.logits=torch.nn.Parameter(torch.tensor([.1,.4,-.3,0.,0.,0.]))
        def initial_memory(self,n,device): return torch.zeros(n,1,device=device)
        def forward(self,rgb,goal,memory,masks):
            return self.logits.view(1,1,6).expand(1,len(goal),6),None,memory
    model=Policy(); device=torch.device('cpu')
    batch={'rgb':torch.zeros(2,1,224,224,3,dtype=torch.uint8),'action':torch.tensor([[1],[0]]),
           'goal':torch.zeros(2,1,dtype=torch.long),'mask':torch.ones(2,1,1),'valid':torch.tensor([[1.],[0.]])}
    weights=torch.tensor([20.,2.,1.,1.,0.,0.])
    loss,_,_,_=forward_batch(model,batch,device,weights); loss.backward(); actual=model.logits.grad.clone()
    model.logits.grad=None
    expected=torch.nn.functional.cross_entropy(model.logits[None],torch.tensor([1])); expected.backward()
    assert torch.allclose(loss,expected)
    assert torch.allclose(actual,model.logits.grad)


def test_live_policy_rgb_matches_saved_training_observation(tmp_path):
    from spikingnav.habitat_objectnav.trajectory import observation_rgb,decode_rgb
    path,r,m=example(tmp_path)
    r['rgb'][0]=np.random.default_rng(42).integers(0,256,(480,640,3),dtype=np.uint8)
    write_trajectory(path,r,m)
    with np.load(path,allow_pickle=False) as data:
        assert np.array_equal(observation_rgb(r['rgb'][0]),decode_rgb(data,[0])[0])


def test_invalid_alignment_never_publishes_a_training_file(tmp_path):
    from spikingnav.habitat_objectnav.trajectory import TrajectoryValidationError
    path,r,m=example(tmp_path); r['positions'][2]=[0,.005,.25]; r['terminal_position']=[0,.005,.25]
    with pytest.raises(TrajectoryValidationError) as exc: write_trajectory(path,r,m)
    assert not path.exists()
    assert exc.value.path.suffix=='.rejected' and exc.value.path.exists()
    assert not exc.value.audit['checks']['pose_action_alignment']
    assert list(tmp_path.glob('*.npz'))==[]
