import torch
import pytest
from highway_env.planner.diffusion.projection import JointTrajectoryProjector, load_jtp_config
from highway_env.planner.diffusion.projection.grouped_solver import (build_priority_groups, protect_direction, accept_nonlinear, solve_groups)
from highway_env.planner.diffusion.projection.linearization import K2StaticCache, shared_linearization, jacobian
from highway_env.planner.diffusion.common.execution import VehicleExecution
from tests.test_joint_trajectory_projection import _context, _road, StraightLineExecution


def k2():
    torch.manual_seed(42)
    metadata={'model_type':'K2','condition_hidden':16,'model_kwargs':{'history_steps':5,'latent_dim':8,'hidden_dim':16},
              'normalization':{'state_mean':[14.,.1,.01,0.,0.],'state_std':[3.,.5,.1,.1,.2],'control_mean':[.1,.01],'control_std':[2.,.05]},
              'condition_normalization':{'condition_mean':[5000.,.8],'condition_std':[10000.,.1]}}
    model=VehicleExecution({'model':'koopman_k2'},dt=.1,koopman_metadata=metadata).double()
    with torch.no_grad():
        model.koopman.condition_net[-1].weight.normal_(std=.005)
        model.koopman.decode_correction[-1].weight.normal_(std=.03)
    return model


def test_structured_k2_matches_full_autograd_at_two_working_points():
    cfg=load_jtp_config(overrides={'execution':{'horizon_steps':4,'control_steps':4}})
    proj=JointTrajectoryProjector(k2(),cfg)
    u=torch.randn(2,4,2,dtype=torch.double)*.02
    ctx=proj._to_context(_context(2), n=2,device=u.device,dtype=u.dtype)
    ctx['koopman_pose0_si'][:,2]=torch.tensor([.3,-.7],dtype=u.dtype)
    ctx['koopman_road_mu'][1]=.55
    ctx['koopman_payload_kg'][1]=9000.
    cache=K2StaticCache(proj.rollout_fn,ctx,proj.control_interpolation)
    for value in (u,u+.015):
        js=cache.output_jacobians(value)
        for key,j in js.items():
            ref=jacobian(lambda x:proj._rollout(x.reshape_as(value),ctx)[key],value.flatten())
            torch.testing.assert_close(j,ref,rtol=1e-7,atol=1e-9)


def test_all_constraint_jacobians_match_frozen_association_autograd():
    cfg=load_jtp_config(overrides={'execution':{'horizon_steps':4,'control_steps':4}})
    proj=JointTrajectoryProjector(k2(),cfg)
    u=torch.randn(2,4,2,dtype=torch.double)*.02
    ctx=proj._to_context(_context(2), n=2,device=u.device,dtype=u.dtype)
    roads=proj._road_list(_road(),2,device=u.device,dtype=u.dtype)
    actor={'pose':torch.tensor([[10.+t,2.,.1] for t in range(5)],dtype=u.dtype),'length_m':4.5,'width_m':1.8}
    actors=(actor,)
    current=proj._evaluate(u,ctx,roads,external_actors=actors)
    selected={key:torch.arange(min(3,len(val)),device=u.device) for key,val in current.groups.items()}
    cache=K2StaticCache(proj.rollout_fn,ctx,proj.control_interpolation)
    rows,metric,method=shared_linearization(proj,u,ctx,roads,current,selected,actors,cache)
    for key,j in rows.items():
        ref=jacobian(lambda x:proj._constraint_groups(x.reshape_as(u),ctx,roads,current.associations,actors)[1][key][selected[key]],u.flatten())
        torch.testing.assert_close(j,ref,rtol=1e-7,atol=1e-9)
    assert method=='structured_k2'
    assert int(torch.linalg.cholesky_ex(metric).info)==0


def test_metric_protection_and_safe_failure():
    cfg=load_jtp_config()['projection']
    M=torch.tensor([[2.,.4],[.4,1.]],dtype=torch.double)
    H=torch.tensor([[1.,0.],[.5,1.]],dtype=torch.double)
    d=torch.tensor([1.,1.],dtype=torch.double)
    step,_,_,ok=protect_direction(d,H,torch.linalg.cholesky(M),cfg)
    assert ok
    assert float((H@step).max())<=cfg['protection_tolerance']
    # Prior row improvement is allowed, not constrained to its nullspace.
    candidate=torch.tensor([-1.,-.5],dtype=torch.double)
    step,_,_,ok=protect_direction(candidate,H,torch.linalg.cholesky(M),cfg)
    assert ok
    torch.testing.assert_close(step,candidate)


def test_group_solver_protects_preceding_rows_and_uses_one_factorization():
    cfg=load_jtp_config()['projection']; cfg['max_step_norm']=10.
    u=torch.zeros(1,1,2,dtype=torch.double)
    groups={'a':torch.tensor([.2],dtype=u.dtype),'b':torch.tensor([.4],dtype=u.dtype)}
    selected={key:torch.tensor([0]) for key in groups}
    rows={'a':torch.tensor([[1.,0.]],dtype=u.dtype),'b':torch.tensor([[-1.,1.]],dtype=u.dtype)}
    delta,diag=solve_groups(u,groups,selected,rows,torch.eye(2,dtype=u.dtype),[['a'],['b']],cfg,torch.ones(2,dtype=u.dtype),10.,1.)
    assert diag['metric_factorizations']==1 and len(diag['groups'])==2
    assert all(g['protection_peak']<=cfg['protection_tolerance'] for g in diag['groups'])
    assert (rows['a']@delta.flatten()).item()<0
    assert (groups['b']+rows['b']@delta.flatten()).item()<.4


def test_priority_and_acceptance_and_explicit_qp_reservation():
    cfg=load_jtp_config(); pc=cfg['projection']
    groups={key:torch.tensor([.1]) for key in pc['priority_weights']}
    groups['road']=torch.tensor([3.])
    order,scores=build_priority_groups(groups,pc)
    assert order[0][0]=='road'
    assert sum(map(len,order))==len(groups)
    assert accept_nonlinear([1.,2.,0.],[.8,2.,0.],pc)
    assert not accept_nonlinear([1.,2.,0.],[.8,2.1,0.],pc)
    assert not accept_nonlinear([1.,2.,0.],[.8,2.,.1],pc)
    assert not accept_nonlinear([1.,2.,0.],[1.,2.,0.],pc)
    pc['solver_method']='constrained_qp'
    projector=JointTrajectoryProjector(StraightLineExecution(),cfg)
    with pytest.raises(NotImplementedError,match='Stage 2'):
        projector.project(torch.zeros(1,40,2),_context(),_road())


def test_real_k2_projector_reuses_cache_and_returns_consistent_rollout():
    cfg=load_jtp_config(overrides={'execution':{'horizon_steps':4,'control_steps':4},'projection':{'max_iterations':3}})
    proj=JointTrajectoryProjector(k2(),cfg)
    u=torch.zeros(2,4,2,dtype=torch.double);u[:,:,0]=2.
    ctx=proj._to_context(_context(2),n=2,device=u.device,dtype=u.dtype)
    result=proj.project(u,ctx,_road())
    assert result.diagnostics['static_cache_builds']==1
    assert result.diagnostics['linearization_count']>=1
    assert all(step['linearization_method']=='structured_k2' for step in result.diagnostics['projection'])
    expected=proj._rollout(result.controls,ctx)
    torch.testing.assert_close(result.trajectories,expected['trajectory'])
    torch.testing.assert_close(result.states,expected['states'])
    for item in result.diagnostics['projection']:
        if item['accepted'] and not result.projector_constraints_satisfied:
            assert accept_nonlinear(item['priority_group_maxima_before'],item['priority_group_maxima_after'],cfg['projection'])
