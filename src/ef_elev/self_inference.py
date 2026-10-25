"""Native self-collected inference class, isolated from training and data paths."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sampler import AttnGuidedSampler
from rsrd_reo import MonoREO
from reo_core import nerf_positional_encoding
from self_core import ReoHeightNet
from models.efficientnet import efficientnet_feature

class SelfSfree(MonoREO):
    def __init__(self,cfg,initialize=True):
        nn.Module.__init__(self)
        self.backbone=efficientnet_feature(stereo=False)
        self.height=ReoHeightNet(64,(80,50,200),100,dim=128,num_heads=4,num_layers=2,
            query_chunk=1024,pool_size=(66,120),attn_grid=(100,40),roi_x=(-1.2,1.2),roi_z=(4.,10.),
            grid_res=(.03,.03),num_cams=1,grad_ckpt=True,use_self_attn=True)
        self.height.aux_depth=None;self.height.aux_rgb=None
        self.sampler=AttnGuidedSampler(128,64,(66,120),(132,240),heads=4,points=4,max_px=24)
        if initialize:
            state=torch.load(cfg['pretrained'],map_location='cpu',weights_only=True)
            remapped={k.replace('features.','l'):v for k,v in state.items() if any(k.startswith('features.%d.'%i) for i in range(5))}
            missing,unexpected=self.backbone.load_state_dict(remapped,strict=False)
            assert remapped and not unexpected and all(k.startswith('final_conv.') for k in missing)

    def components(self,image):
        H=self.height;feature=self.backbone(image);b=feature.shape[0]
        tokens,_=H.image_encoder(feature);nz,nx=H.attn_grid
        coords=H.build_query_grid(b,feature.device,feature.dtype,(nz,nx))
        query=H.query_norm(H.query_embed(nerf_positional_encoding(coords,H.image_encoder.num_freqs,bounds=H.coord_bounds)))
        query=query+H.query_pos_embed[:,:query.shape[1]];outs=[]
        for s0 in range(0,query.shape[1],H.query_chunk):
            piece=query[:,s0:s0+H.query_chunk]
            for block in H.blocks:
                if H.grad_ckpt and self.training and torch.is_grad_enabled():
                    piece=checkpoint(lambda q,m,bl=block:bl(q,m)[0],piece,tokens,use_reentrant=True)
                else:piece,_=block(piece,tokens)
            outs.append(piece)
        query=torch.cat(outs,1)
        bev=F.interpolate(query.transpose(1,2).reshape(b,-1,nz,nx),size=(200,80),mode='bilinear',align_corners=False)
        cell=bev.flatten(2).transpose(1,2);fine=self.sampler.fine_proj(feature)
        if H.grad_ckpt and self.training and torch.is_grad_enabled():
            add,_=checkpoint(self.sampler,cell,tokens,fine,use_reentrant=False)
        else:add,_=self.sampler(cell,tokens,fine)
        logits=H.bev_head(H.bev_refine(bev+add.transpose(1,2).reshape_as(bev)))
        values=(24.75-torch.arange(100,device=logits.device)*.5)[None,:,None,None]
        return dict(logits=logits,prediction=(logits.float().softmax(1)*values).sum(1))

    def forward(self,image):return self.components(image)['prediction']
