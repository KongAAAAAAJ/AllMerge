import json, tempfile, sys, types, importlib.machinery
from pathlib import Path
import numpy as np, torch
try:
    import gymnasium
except ImportError:
    gymnasium=types.ModuleType('gymnasium'); gymnasium.__spec__=importlib.machinery.ModuleSpec('gymnasium',loader=None)
    envs=types.ModuleType('gymnasium.envs'); envs.__spec__=importlib.machinery.ModuleSpec('gymnasium.envs',loader=None)
    reg=types.ModuleType('gymnasium.envs.registration'); reg.__spec__=importlib.machinery.ModuleSpec('gymnasium.envs.registration',loader=None)
    reg.register=lambda *args,**kwargs:None; envs.registration=reg; gymnasium.envs=envs
    sys.modules['gymnasium']=gymnasium; sys.modules['gymnasium.envs']=envs; sys.modules['gymnasium.envs.registration']=reg
from highway_env.vehicle.koopman.models.conditional_model import ConditionalKoopman
from highway_env.planner.diffusion.training.control import train

root=Path('/tmp/trace_safety_train_smoke'); import shutil; shutil.rmtree(root,ignore_errors=True)
(root/'episodes').mkdir(parents=True); (root/'reports').mkdir()
for ep in range(2):
    n=9
    arrays={
      'ego_state':np.zeros((n,14),np.float32),
      'agent_states':np.zeros((n,16,11),np.float32),
      'agent_valid_mask':np.zeros((n,16),bool),
      'map_polylines':np.zeros((n,8,32,10),np.float32),
      'map_valid_mask':np.zeros((n,8),bool),
      'target_point':np.tile([[80.,0.]],(n,1)).astype(np.float32),
      'target_lane_polyline':np.zeros((n,32,10),np.float32),
      'road_lane_geometry':np.zeros((n,3,40,9),np.float32),
      'road_lane_point_mask':np.ones((n,3,40),bool),
      'road_lane_mask':np.ones((n,3),bool),
      'road_current_lane_slot':np.ones(n,np.int16),
      'road_target_lane_slot':np.ones(n,np.int16),
      'road_lane_connections':np.ones((n,3,3),bool),
      'road_lane_meta':np.zeros((n,3,7),np.float32),
      'expert_trajectory_xy':np.zeros((n,8,2),np.float32),
      'future_trajectory_dense':np.zeros((n,40,2),np.float32),
      'koopman_state_hist_si':np.zeros((n,5,5),np.float32),
      'koopman_control_hist_si':np.zeros((n,4,2),np.float32),
      'koopman_pose0_si':np.zeros((n,3),np.float32),
      'koopman_payload_kg':np.full(n,3500.,np.float32),
      'koopman_road_mu':np.full(n,.6,np.float32),
      'episode_index':np.full(n,ep,np.int32),
      'frame_index':np.repeat(np.arange(3),3).astype(np.int32),
      'ego_index':np.tile(np.arange(3),3).astype(np.int32),
      'seed':np.full(n,ep,np.int32),
    }
    for i,y in enumerate((-3.7,0.,3.7)):
      arrays['road_lane_geometry'][:,i,:,0]=np.linspace(-20,160,40)
      arrays['road_lane_geometry'][:,i,:,1]=y
      arrays['road_lane_geometry'][:,i,:,7]=3.7
      arrays['map_polylines'][:,i,:,0]=np.linspace(-20,160,32)
      arrays['map_polylines'][:,i,:,1]=y
      arrays['map_polylines'][:,i,:,4]=3.7
      arrays['map_polylines'][:,i,:,6+(1 if i==0 else 0)]=1.
      arrays['map_valid_mask'][:,i]=True
    arrays['target_lane_polyline'][:,:,0]=np.linspace(-20,160,32)
    arrays['target_lane_polyline'][:,:,1]=0.
    arrays['target_lane_polyline'][:,:,4]=3.7
    arrays['target_lane_polyline'][:,:,6]=1.
    arrays['ego_state'][:,0]=20.
    arrays['koopman_state_hist_si'][:,:,0]=20.
    for row in range(n):
      ego=int(arrays['ego_index'][row]); arrays['koopman_pose0_si'][row]=[ego*15.,0.,0.]
    arrays['future_trajectory_dense'][:,:,0]=np.arange(1,41,dtype=np.float32)[None,:]*2.
    arrays['expert_trajectory_xy'][:,:,0]=np.arange(1,9,dtype=np.float32)[None,:]*10.
    name=f'episode_{ep:06d}.npz'; np.savez(root/'episodes'/name,**arrays)
manifest={'schema_version':'trace_expert_raw_k2_v2','complete':True,'trajectory':{'dt_s':.1},
 'episode_files':[{'name':f'episode_{ep:06d}.npz','episode_index':ep,'samples':9} for ep in range(2)],
 'split_ratios':[.5,.5,0.],'split_seed':0}
(root/'reports/manifest.json').write_text(json.dumps(manifest))
# tiny K2 checkpoint
norm=dict(state_mean=[0.]*5,state_std=[1.]*5,control_mean=[0.]*2,control_std=[1.]*2)
cond=dict(condition_mean=[3500.,.6],condition_std=[1000.,.1])
kw=dict(history_steps=5,latent_dim=8,hidden_dim=16)
k2=ConditionalKoopman('K2',condition_hidden=12,**norm,**cond,**kw)
k2path=root/'k2.pt'; torch.save(dict(model_type='K2',model_kwargs=kw,normalization=norm,condition_normalization=cond,condition_hidden=12,model_state=k2.state_dict()),k2path)
# fixed anchor bank
anchors=np.zeros((10,40,2),np.float32); anchors[...,0]=0.0
meta={'implementation':'A0_FIXED_CONTROL_KMEANS_EXEMPLARS_V4','num_anchors':10,'source_split':'train','cluster_space':'flattened_40_step_physical_control_sequence','prototype_type':'nearest_expert_to_cluster_centroid','medoid_selection_used':False,'trajectory_to_control_matching_used':False,'mode_valid_mask_used':False,'safety_scoring_used':False,'adaptive_selection_used':False}
scale=np.array([2.5,.08],np.float32)
bank=root/'bank.npz'; np.savez_compressed(bank,anchors_controls=anchors,cluster_centers_controls=anchors,control_scale=scale,metadata_json=np.asarray(json.dumps(meta,sort_keys=True,separators=(',',':'))))
small={'num_modes':10,'max_agents':16,'max_map_polylines':8,'map_points':32,'d_model':32,'d_ffn':64,'num_heads':4,'num_scene_layers':0,'num_denoiser_layers':1,'dropout':0.0}
ctrl={'horizon_steps':40,'dt':.1,'wheelbase_m':5.5,'max_accel_mps2':2.5,'max_steer_rad':.08,'num_train_timesteps':1000,'inference_steps':2,'inference_start_timestep':8,'inference_noise_scale':1.0}
losses={'mode_selection':'goal_endpoint','trajectory':{'weight':1.},'background_collision':{'weight':1.},'intertruck_collision':{'weight':1.},'ttc':{'weight':1.},'road_boundary':{'weight':1.},'safety':{'vehicle_length_m':8.,'vehicle_width_m':2.6,'collision_margin_m':.2,'road_margin_m':.15,'ttc_threshold_s':1.,'softplus_temperature_m':.2,'collision_distance_scale_m':1.,'ttc_distance_scale_m':5.,'road_distance_scale_m':1.},'ltr':{'weight':0.,'limit':.6},'fuel':{'weight':0.},'progress':{'weight':0.,'minimum_progress_m':60.}}
cfg={'output_space':'control','model':small,'control':ctrl,'execution':{'model':'koopman_k2','checkpoint':str(k2path)},'anchors':{'strategy':'fixed','num_selected':10,'bank_path':str(bank)},'losses':losses,'training':{'batch_size':3,'num_workers':0,'max_epochs':1,'lr':1e-4,'weight_decay':1e-4,'grad_clip':1.,'patience':0,'seed':0}}
print('running training smoke')
out=train(cfg,root=root,output_dir=root/'out',max_epochs=1,max_steps=1,device='cpu')
print('TRAIN_OK',out)
print((root/'out'/'loss.csv').read_text())
