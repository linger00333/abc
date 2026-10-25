"""Image-only inference with fixed calibration, plane reconstruction and train augmentations."""
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from rsrd_reo import MonoREO as Baseline, prediction as classify, height_loss, ROOT
from geometry import axis_rotation, warp_image_a2, project_points

def design_grid():
    x=-1+(torch.arange(64).float()+.5)*.03
    z=7.08-(torch.arange(164).float()+.5)*.03
    zz,xx=torch.meshgrid(z,x,indexing='ij')
    return torch.stack([torch.ones_like(xx),xx+.04,zz-4.62],-1)

def uniform_lower_image_grid(nz,nx,j):
    """Calibration-free initial layout in normalized image coords: BEV x -> image u,
    BEV z (far->near rows) -> image v, both spread over the full frame. The only prior is
    the generic ordering 'farther is higher in the image'; J points jittered on a 5x3
    stencil so they do not collapse. No K/R involved."""
    u=torch.linspace(-.9,.9,nx);v=torch.linspace(-.9,.9,nz)
    vv,uu=torch.meshgrid(v,u,indexing='ij')
    base=torch.stack([uu,vv],-1).reshape(nz*nx,1,2)
    du=torch.tensor([-1.,0.,1.]).repeat(5)*.02;dv=torch.linspace(-.04,.04,5).repeat_interleave(3)
    return base+torch.stack([du,dv],-1)[None]

class GeometrySampler(nn.Module):
    def __init__(self,cfg):
        super().__init__();d=cfg['dim']; nz,nx=cfg['attn_grid'];j=15
        fixed=cfg['geometry_fixed']; k=torch.tensor(fixed['K'],dtype=torch.float64);r=torch.tensor(fixed['r_vert2cam'],dtype=torch.float64)
        self.register_buffer('K_fixed',k.float());self.register_buffer('R_fixed',r.float())
        x=-1+(torch.arange(nx,dtype=torch.float64)+.5)*1.92/nx
        z=7.08-(torch.arange(nz,dtype=torch.float64)+.5)*4.92/nz
        zz,xx=torch.meshgrid(z,x,indexing='ij'); hs=torch.tensor([-20,-10,0,10,20],dtype=torch.float64)
        xyz=torch.stack([xx[...,None].expand(-1,-1,5),1.1-hs[None,None,:].expand(nz,nx,-1)/100,zz[...,None].expand(-1,-1,5)],-1)
        self.mode=cfg.get('reference_mode','fixed')
        if self.mode=='fixed_wrong':
            # Deliberately miscalibrated prior: rotation error applied to the fixed mounting.
            e=cfg['wrong_rotation_deg'];delta=torch.from_numpy(axis_rotation('roll',e['roll'])@axis_rotation('pitch',e['pitch']))
            r=delta@r
        cam=xyz@r.T;uvz=cam@k.T;uv=uvz[...,:2]/uvz[...,2:]
        uv=uv.reshape(nz*nx,5,2).repeat_interleave(3,dim=1)
        grid=2*(uv+.5)/torch.tensor([960,528])-1
        self.register_buffer('projected_grid',grid.float())   # diagnostics only for learned modes
        self.pool_size=tuple(cfg['pool_size']); self.j=j
        self.local=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,j*3))
        self.global_offset=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2))
        self.out=nn.Linear(d,d);self.norm=nn.LayerNorm(d)
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.global_offset[-1].weight);nn.init.zeros_(self.global_offset[-1].bias)
        # Symmetric nonzero initial sampling points avoid identical offset paths.
        initial=torch.zeros(j,3);initial[::3,0]=-4/48;initial[2::3,0]=4/48
        with torch.no_grad():self.local[-1].bias.copy_(initial.reshape(-1))
        if self.mode in ('fixed','fixed_wrong'):
            self.register_buffer('reference_grid',grid.float()); self.register_buffer('front', (cam[...,2]>0).reshape(nz*nx,5).repeat_interleave(3,dim=1))
        else:
            # Calibration-free: K/R above are never used by the forward pass in these modes.
            self.register_buffer('front',torch.ones(nz*nx,j,dtype=torch.bool))
            if self.mode=='learned_static':
                self.reference_raw=nn.Parameter(torch.atanh(uniform_lower_image_grid(nz,nx,j).clamp(-.99,.99)))
            elif self.mode=='learned_query':
                self.ref_head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,j*2))
                # Zero-init head: start exactly at the learned_static layout, then let the query move it.
                nn.init.zeros_(self.ref_head[-1].weight);nn.init.zeros_(self.ref_head[-1].bias)
                self.reference_raw=nn.Parameter(torch.atanh(uniform_lower_image_grid(nz,nx,j).clamp(-.99,.99)))
            else:raise ValueError(self.mode)
    def references(self,joint):
        if self.mode in ('fixed','fixed_wrong'):return self.reference_grid[None]
        if self.mode=='learned_static':return self.reference_raw.float().tanh()[None]
        # joint = query + per-frame image context, so references adapt to each frame's view.
        b,n,_=joint.shape
        return (self.reference_raw.float()[None]+self.ref_head(joint).float().reshape(b,n,self.j,2)).tanh()
    def sampling_grid(self,query,tokens):
        context=tokens.mean(1); joint=query+context[:,None,:]
        o=self.local(joint).float().reshape(query.shape[0],query.shape[1],self.j,3)
        scale=torch.tensor([2/960,2/528],device=query.device)
        local=o[...,:2].tanh()*48*scale
        global_o=self.global_offset(context).float().tanh()[:,None,None,:]*96*scale
        return self.references(joint)+local+global_o,o[...,2]
    def forward(self,query,tokens):
        grid,weight=self.sampling_grid(query,tokens)
        visible=(grid.abs()<=1).all(-1)&self.front[None]
        b,n,j,_=grid.shape
        tm=tokens.transpose(1,2).reshape(b,-1,*self.pool_size)
        with torch.autocast(device_type=query.device.type,enabled=False):
            sampled=F.grid_sample(tm.float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False).permute(0,2,3,1)
            weights=weight.masked_fill(~visible,-1e4).softmax(-1)*visible
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
            gathered=(sampled*weights[...,None]).sum(2)
        return self.norm(self.out(gathered.to(tokens.dtype)))*visible.any(-1)[...,None]

class Improved(Baseline):
    def __init__(self,cfg,initialize=True):
        super().__init__(cfg,initialize)
        self.modules_enabled=cfg['modules']
        if 'G' in self.modules_enabled:self.height.geometry_sampler=GeometrySampler(cfg)
        if 'P' in self.modules_enabled:
            self.height.bev_head=None
            self.residual_head=nn.Sequential(nn.Conv2d(cfg['dim'],cfg['dim'],3,padding=1),nn.GroupNorm(8,cfg['dim']),nn.SiLU(),nn.Conv2d(cfg['dim'],1,1))
            self.plane_head=nn.Sequential(nn.Linear(cfg['dim'],cfg['dim']),nn.GELU(),nn.Linear(cfg['dim'],3))
            self.register_buffer('plane_design',design_grid())
    def components(self,image):
        feature=self.backbone(image)
        if 'P' not in self.modules_enabled:
            logits,aux=self.height(feature,return_aux=False)
            return dict(logits=logits,prediction=classify(logits))
        bev=self.height(feature,return_aux=False,return_bev=True)
        coef=self.plane_head(bev.mean((2,3))).float()
        with torch.autocast(device_type=image.device.type,enabled=False):
            plane=torch.einsum('bc,zxc->bzx',coef,self.plane_design.float())
        residual=self.residual_head(bev)[:,0].float()
        return dict(prediction=plane+residual,plane=plane,residual=residual,coefficients=coef)
    def forward(self,image):return self.components(image)['prediction']
    def forward_batch(self,batch):return self(batch['image'])

def loss_components(out,b,cfg):
    with torch.autocast(device_type=b['ele_gt'].device.type,enabled=False):
        return _loss_components_float32(out,b,cfg)

def _loss_components_float32(out,b,cfg):
    gt=b['ele_gt']; mask=b['ele_mask']&(gt>-20)&(gt<20)
    assert mask.any(),'No visible supervised cells'
    if 'P' not in cfg['modules']:
        if not cfg.get('hf_loss'):
            loss=height_loss(out['logits'],gt,mask);return loss,dict(ce=float(loss.detach()))
        return hf_loss(out,gt,mask,cfg['hf_loss'])
    a=design_grid().to(gt.device); true_plane=torch.einsum('bc,zxc->bzx',b['plane_coef'].float(),a)
    true_resid=gt-true_plane
    vals=[F.smooth_l1_loss(out['prediction'][mask],gt[mask],beta=1),F.smooth_l1_loss(out['residual'][mask],true_resid[mask],beta=1),F.smooth_l1_loss(out['plane'][mask],true_plane[mask],beta=1)]
    # Penalize residual plane over the supervised mask; infer never reads this mask.
    mw=mask.float(); gram=torch.einsum('bzx,zxi,zxj->bij',mw,a,a)
    rhs=torch.einsum('bzx,bzx,zxi->bi',mw,out['residual'],a)
    co=torch.linalg.solve(gram+1e-5*torch.eye(3,device=gt.device)[None],rhs)
    residual_plane=torch.einsum('bc,zxc->bzx',co,a)
    vals.append(F.smooth_l1_loss(residual_plane[mask],torch.zeros_like(residual_plane[mask]),beta=1))
    loss=sum(w*v for w,v in zip(cfg['plane_weights'],vals))
    return loss,{n:float(v.detach()) for n,v in zip(['height','residual','plane','residual_plane'],vals)}

class TrainFrames(torch.utils.data.Dataset):
    def __init__(self,cfg,ids=None,augment=True):
        self.cfg=cfg;self.epoch=0;self.augment=augment and 'A' in cfg['modules']
        m=json.loads(Path(cfg['manifest']).read_text()); self.samples=[s for s in m['split']['train'] if ids is None or s['sample_id'] in set(ids)]
        with np.load(cfg['plane_targets']) as f:self.planes=dict(zip(f['sample_ids'].tolist(),f['coefficients'].copy()))
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i];sid=s['sample_id']
        with Image.open(s['image_path']) as f:a=np.array(f.convert('RGB'))
        with np.load(Path(str(self.cfg['gt_root']).replace('/home/lixl/Project/roadbev_5mix/reo/experiments/exp23_rsrd_dense_exp18_exp21_s307_20261014/artifacts','/home/lixl/Project/roadbev_5mix/reo/experiments/rsrd_dense_exact_cache_20261014/artifacts'))/'train'/(sid+'.npz')) as f:
            gt=f['ele_gt'].copy(); mask=f['ele_mask'].copy()
            if self.augment:
                r=f['r_vert2cam'];k=f['K'];centers=f['centers'].copy()
        seed=int(hashlib.sha256(f'{self.cfg["seed"]}:{self.epoch}:{sid}'.encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed); pitch=roll=0.;retention=1.
        if self.augment and rng.random()<self.cfg['augmentation_probability']:
            pitch,roll=rng.uniform(-self.cfg['rotation_range_deg'],self.cfg['rotation_range_deg'],2)
            delta=axis_rotation('roll',roll)@axis_rotation('pitch',pitch)
            a,support=warp_image_a2(a,k,delta)
            xyz=np.c_[centers[:,0],1.1-gt.reshape(-1)/100,centers[:,1]].T
            cam=delta@r@xyz;uv=project_points(k,cam);finite=np.isfinite(uv).all(0)&(cam[2]>0)
            uv=np.where(np.isfinite(uv),uv,-9999); xx=np.floor(uv[0]).astype(int);yy=np.floor(uv[1]).astype(int)
            inside=finite&(xx>=0)&(xx+1<960)&(yy>=0)&(yy+1<528)
            x=np.clip(xx,0,958);y=np.clip(yy,0,526)
            vis=(inside&support[y,x]&support[y+1,x]&support[y,x+1]&support[y+1,x+1]).reshape(164,64)
            original=mask&(gt>-20)&(gt<20); mask&=vis
            retention=float((mask&original).sum()/original.sum())
        else:a=a[:528].copy()
        if self.augment and rng.random()<self.cfg['color_probability']:
            br,ct,sa=rng.uniform(*self.cfg['color_range'],3); a=a.astype(np.float32)*br
            mean=a.mean((0,1),keepdims=True);a=(a-mean)*ct+mean
            gray=a@np.array([.299,.587,.114],np.float32);a=gray[...,None]+(a-gray[...,None])*sa
            a=np.clip(a,0,255)
        return dict(image=torch.from_numpy(a.copy()).permute(2,0,1).float()/127.5-1,ele_gt=torch.from_numpy(gt),ele_mask=torch.from_numpy(mask),plane_coef=torch.from_numpy(self.planes[sid]),sample_id=sid,rotation_deg=torch.tensor([pitch,roll]),retention=retention)


def deplane(v,mask,a):
    """Subtract each frame's own masked least-squares plane (same gram/solve as residual_plane)."""
    mw=mask.float();gram=torch.einsum('bzx,zxi,zxj->bij',mw,a,a)
    rhs=torch.einsum('bzx,bzx,zxi->bi',mw,v,a)
    co=torch.linalg.solve(gram+1e-5*torch.eye(3,device=v.device)[None],rhs)
    return v-torch.einsum('bc,zxc->bzx',co,a)

def hf_loss(out,gt,mask,w):
    """Relief-weighted CE + neighbour-gradient + deplaned-shape loss on the expected height."""
    a=design_grid().to(gt.device);pred=out['prediction']
    gres=deplane(gt.masked_fill(~mask,0),mask,a)
    weight=1+w['relief_weight']*(gres.abs()>w['relief_threshold_cm']).float()
    labels=79-torch.floor((gt[mask]+20)/.5).long()
    ce=F.cross_entropy(out['logits'].float().permute(0,2,3,1)[mask],labels,reduction='none')
    ce=(ce*weight[mask]).sum()/weight[mask].sum()
    mz=mask[:,1:]&mask[:,:-1];mx=mask[:,:,1:]&mask[:,:,:-1]
    grad=F.smooth_l1_loss((pred[:,1:]-pred[:,:-1])[mz],(gt[:,1:]-gt[:,:-1])[mz],beta=w['beta_cm'])+\
         F.smooth_l1_loss((pred[:,:,1:]-pred[:,:,:-1])[mx],(gt[:,:,1:]-gt[:,:,:-1])[mx],beta=w['beta_cm'])
    shape=F.smooth_l1_loss(deplane(pred.masked_fill(~mask,0),mask,a)[mask],gres[mask],beta=w['beta_cm'])
    loss=ce+w['grad']*grad+w['deplane']*shape
    return loss,dict(ce_relief=float(ce.detach()),grad=float(grad.detach()),deplane=float(shape.detach()))
