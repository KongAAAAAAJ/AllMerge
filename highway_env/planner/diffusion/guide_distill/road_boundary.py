"""Probe-only signed distance/reward to outermost *union* of road lanes.

Production W4 road score is intentionally unmodified. This geometric teacher
uses lane geometry from the current env, builds a shapely union of lane ribbons,
and uses a differentiable PyTorch nearest-segment distance for guidance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch


def _polygon_components(geom):
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return [p for g in geom.geoms for p in _polygon_components(g)]
    return []


def build_outer_road_union(road, *, sampling_m=0.4):
    """Construct true drivable union, removing internal lane seams/endcaps."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from highway_env.planner.diffusion.trajectory_mode_reward.geometry import _iter_lanes
    polys=[]
    for lane in _iter_lanes(road):
        L=float(lane.length)
        if not np.isfinite(L) or L<=0:
            continue
        samples=np.linspace(0,L,max(2,int(math.ceil(L/sampling_m))+1))
        try:
            left=[np.asarray(lane.position(float(s),0.5*float(lane.width_at(float(s)))),dtype=float) for s in samples]
            right=[np.asarray(lane.position(float(s),-0.5*float(lane.width_at(float(s)))),dtype=float) for s in samples[::-1]]
            poly=Polygon(np.vstack([left,right]))
            if not poly.is_valid:poly=poly.buffer(0)
            if not poly.is_empty:polys.append(poly)
        except (ValueError,AttributeError,TypeError):
            continue
    if not polys:
        raise RuntimeError('No valid lane ribbons to construct outermost road boundary')
    union=unary_union(polys)
    if not union.is_valid:union=union.buffer(0)
    # Internal holes are not outermost road boundaries under the user's definition.
    pieces=[Polygon(p.exterior) for p in _polygon_components(union)]
    union=unary_union(pieces)
    if union.is_empty:raise RuntimeError('Road union is empty')
    return union


@dataclass
class RoadBoundaryField:
    geometry: object
    edges: np.ndarray
    center_index: object
    sample_refs: np.ndarray
    inside_scale_m:float=2.0
    inside_bonus:float=1.0
    outside_slope:float=0.5
    max_sampling_m:float=0.4

    @classmethod
    def from_road(cls,road,*,sampling_m=0.4,inside_scale_m=2.,inside_bonus=1.,outside_slope=None):
        from scipy.spatial import cKDTree
        geo=build_outer_road_union(road,sampling_m=sampling_m)
        # Subdivide every actual outer-border polyline edge so KD search finds
        # nearby boundary *segments*, not internal lane seams.
        segs=[]
        for poly in _polygon_components(geo):
            ring=np.asarray(poly.exterior.coords,dtype=np.float64)
            for p,q in zip(ring[:-1],ring[1:]):
                n=max(1,int(math.ceil(np.linalg.norm(q-p)/sampling_m)))
                xs=np.linspace(0.,1.,n+1)
                for a,b in zip(xs[:-1],xs[1:]):
                    segs.append([p+(q-p)*a,p+(q-p)*b])
        edges=np.asarray(segs,dtype=np.float64)
        if edges.ndim!=3 or len(edges)==0 or not np.isfinite(edges).all():
            raise RuntimeError('Invalid road boundary edges')
        mids=edges.mean(axis=1)
        return cls(geometry=geo,edges=edges,center_index=cKDTree(mids),
                   sample_refs=mids,inside_scale_m=float(inside_scale_m),inside_bonus=float(inside_bonus),
                   outside_slope=float(inside_bonus/inside_scale_m if outside_slope is None else outside_slope),
                   max_sampling_m=float(sampling_m))

    def _inside_numpy(self,points):
        try:
            from shapely import contains_xy
            return np.asarray(contains_xy(self.geometry,points[:,0],points[:,1]),dtype=bool)
        except ImportError:
            from shapely.geometry import Point
            from shapely.prepared import prep
            shape=prep(self.geometry)
            return np.asarray([shape.covers(Point(p)) for p in points],dtype=bool)

    def signed_distance(self,world_points:torch.Tensor,*,neighbors=6):
        """Signed margin (inside positive) with gradients through world points.

        The nearest-segment index and inside test are recomputed from detached
        geometry each forward pass. At a boundary corner the distance is
        piecewise differentiable, which is sufficient for backtracked guidance.
        """
        if world_points.shape[-1]!=2 or not torch.isfinite(world_points).all():
            raise ValueError('Expected finite world XY points')
        orig_shape=world_points.shape[:-1]
        pts=world_points.reshape(-1,2)
        query=pts.detach().cpu().numpy()
        k=min(len(self.edges),max(1,int(neighbors)))
        _,idx=self.center_index.query(query,k=k)
        idx=np.asarray(idx).reshape(len(query),k)
        edge=torch.as_tensor(self.edges[idx],device=pts.device,dtype=pts.dtype)
        a=edge[:,:,0,:];b=edge[:,:,1,:]
        ab=b-a
        u=((pts[:,None,:]-a)*ab).sum(-1)/(ab.square().sum(-1).clamp_min(1e-12))
        nearest=a+u.clamp(0.,1.)[:,:,None]*ab
        dist=torch.linalg.vector_norm(pts[:,None,:]-nearest,dim=-1).amin(dim=1)
        inside=torch.as_tensor(self._inside_numpy(query),device=pts.device,dtype=torch.bool)
        margin=torch.where(inside,dist,-dist)
        return margin.reshape(orig_shape)

    def reward_of_margin(self,margin:torch.Tensor):
        """C0 continuous signed road reward. More interior margin -> more reward;
        outside: the nearer to the boundary, the smaller the penalty.
        """
        inside=self.inside_bonus*(-torch.expm1(-margin.clamp_min(0.)/self.inside_scale_m))
        outside=self.outside_slope*margin.clamp_max(0.)
        return inside+outside

    def dense_world_footprint_margins(self,xy:torch.Tensor,pose,config,*,tracking_aware=True):
        """Match W4 0.1s LINEAR interpolation, local->world, tracking OBB."""
        from highway_env.planner.diffusion.trajectory_mode_reward.geometry import tracking_aware_dimensions
        if xy.ndim!=3 or tuple(xy.shape[1:])!=(8,2):
            raise ValueError('Expected [N,8,2] metric XY control points')
        device=xy.device;dtype=xy.dtype
        dt=float(config.trajectory_dt_s);dt_dense=float(config.interpolation_dt_s)
        source=np.arange(9,dtype=np.float64)*dt
        target=np.arange(0.,source[-1]+.5*dt_dense,dt_dense)
        ii=np.clip(np.searchsorted(source,target,side='right')-1,0,7)
        ii=torch.as_tensor(ii,device=device,dtype=torch.long)
        alpha=torch.as_tensor((target-source[ii.detach().cpu().numpy()])/(source[ii.detach().cpu().numpy()+1]-source[ii.detach().cpu().numpy()]),device=device,dtype=dtype)
        knots=torch.cat([torch.zeros((len(xy),1,2),device=device,dtype=dtype),xy],dim=1)
        dense=knots[:,ii,:]+alpha[None,:,None]*(knots[:,ii+1,:]-knots[:,ii,:])
        delta=torch.diff(dense,dim=1,prepend=torch.zeros_like(dense[:,:1,:]))
        valid=(torch.linalg.vector_norm(delta,dim=-1)>1e-6)
        yaw_raw=torch.atan2(delta[:,:,1],delta[:,:,0])
        ids=torch.arange(len(target),device=device)[None,:].expand(len(xy),-1)
        last=torch.cummax(torch.where(valid,ids,torch.zeros_like(ids)),dim=1).values
        yaw=torch.gather(yaw_raw,1,last)
        active=torch.cummax(valid.to(torch.int64),dim=1).values.bool()
        yaw=torch.where(active,yaw,torch.zeros_like(yaw))
        pose=torch.as_tensor(pose,device=device,dtype=dtype).reshape(3)
        c=torch.cos(pose[2]);s=torch.sin(pose[2])
        world_x=pose[0]+c*dense[...,0]-s*dense[...,1]
        world_y=pose[1]+s*dense[...,0]+c*dense[...,1]
        world_h=yaw+pose[2]
        dims=tracking_aware_dimensions(config) if tracking_aware else (config.vehicle_length_m,config.vehicle_width_m)
        length,width=map(float,dims)
        corners=torch.as_tensor([[length/2,width/2],[length/2,-width/2],[-length/2,width/2],[-length/2,-width/2]],device=device,dtype=dtype)
        ca=torch.cos(world_h)[...,None];sa=torch.sin(world_h)[...,None]
        cx=world_x[...,None]+ca*corners[:,0]-sa*corners[:,1]
        cy=world_y[...,None]+sa*corners[:,0]+ca*corners[:,1]
        footprint=torch.stack([cx,cy],dim=-1)
        margins=self.signed_distance(footprint).amin(dim=-1)
        return margins,torch.stack([world_x,world_y,world_h],dim=-1)

    def components(self,xy,pose,config):
        margin,world=self.dense_world_footprint_margins(xy,pose,config)
        # Exclude t=0: initial ego pose cannot be changed by planning.
        m=margin[:,1:]
        score=self.reward_of_margin(m)
        reward=0.6*score.amin(dim=1)+0.4*score.mean(dim=1)
        return {'road':reward,'road_min_margin_m':m.amin(dim=1),
                'road_outside_fraction':(m<0).to(xy.dtype).mean(dim=1),
                'road_time_margin':m,'road_world_poses':world}
