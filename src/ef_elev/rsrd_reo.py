"""Isolated left-image-only RSRD adapter, fixed final epoch, resumable training."""
import argparse, hashlib, inspect, json, os, random, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from models.efficientnet import efficientnet_feature
from reo_core import ReoHeightNet

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def save_json(p,x):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,indent=2)); tmp.replace(p)
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def config():
    return dict(seed=307,epochs=50,batch_size=8,num_workers=2,lr=3e-4,backbone_lr=2e-5,
                optimizer='AdamW',weight_decay=1e-4,scheduler='constant',grad_clip=1.0,
                dim=128,num_heads=4,num_layers=2,query_chunk=1024,pool_size=[66,120],attn_grid=[82,32],
                self_attn=True,grad_ckpt=True,amp='bfloat16',freeze_pretrained_bn=True,
                height_range_cm=20,cla_res_cm=0.5,gt_root=str(ROOT/'artifacts/gt_g_v1'),
                pretrained='/home/lixl/Project/roadbev_5mix/generated/efficientnet_b6_lukemelas-c76e70fd.pth',
                manifest=str(ROOT/'artifacts/protocol_manifest_roadbev_paper_v2.json'),
                checkpoint_selection='final_epoch_50_only',validation=False,augmentation='none',
                auxiliary_read=False,auxiliary_compute=False,num_cams=1,drop_last=False,
                differences_from_RoadBEV='REO attention architecture; AdamW 3e-4/2e-5 constant vs official AdamW/OneCycle max8e-4; batch8 matches official; BF16; pretrained BN frozen; int32 GT audit reported')

class Frames(Dataset):
    def __init__(self,split,cfg):
        self.split=split; self.gt_root=Path(cfg['gt_root'])/split
        # Only id and image path survive: no pose, calibration or derived indices enter training loader.
        m=json.loads(Path(cfg['manifest']).read_text())
        self.samples=[{'sample_id':s['sample_id'],'image_path':s['image_path']} for s in m['split'][split]]
        assert len(self.samples)==(1210 if split=='train' else 371)
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i]
        with Image.open(s['image_path']) as f: a=np.array(f.convert('RGB'))
        assert a.shape==(540,960,3)
        image=torch.from_numpy(a[:528].copy()).permute(2,0,1).float()/127.5-1
        with np.load(self.gt_root/(s['sample_id']+'.npz')) as f:
            gt=torch.from_numpy(f['ele_gt'].copy()); mask=torch.from_numpy(f['ele_mask'].copy())
        return {'image':image,'ele_gt':gt,'ele_mask':mask,'sample_id':s['sample_id']}

class MonoREO(nn.Module):
    def __init__(self,cfg,initialize=True):
        super().__init__(); self.backbone=efficientnet_feature(stereo=False)
        self.height=ReoHeightNet(64,(64,40,164),80,dim=cfg['dim'],num_heads=cfg['num_heads'],
                  num_layers=cfg['num_layers'],query_chunk=cfg['query_chunk'],pool_size=cfg['pool_size'],
                  attn_grid=cfg['attn_grid'],roi_x=(-1,.92),roi_z=(2.16,7.08),grid_res=(.03,.03),
                  num_cams=1,grad_ckpt=cfg['grad_ckpt'],use_self_attn=cfg['self_attn'])
        # Remove auxiliary parameters as well as skipping all auxiliary computation.
        self.height.aux_depth=None; self.height.aux_rgb=None
        self.initialization={}
        if initialize:
            state=torch.load(cfg['pretrained'],map_location='cpu',weights_only=True)
            remapped={k.replace('features.','l'):v for k,v in state.items() if any(k.startswith('features.%d.'%i) for i in range(5))}
            missing,unexpected=self.backbone.load_state_dict(remapped,strict=False)
            assert remapped and not unexpected and all(k.startswith('final_conv.') for k in missing)
            self.initialization=dict(path=cfg['pretrained'],sha256=sha(cfg['pretrained']),loaded_tensors=len(remapped),
                                     missing_random_final_conv=missing,unexpected=unexpected,
                                     self_collected_checkpoint_loaded=False)
    def train(self,mode=True):
        super().train(mode)
        if mode:
            for name,m in self.backbone.named_modules():
                if isinstance(m,nn.modules.batchnorm._BatchNorm) and not name.startswith('final_conv'): m.eval()
        return self
    def forward(self,image):
        logits,aux=self.height(self.backbone(image),return_aux=False)
        assert aux is None
        return logits
    def forward_batch(self,batch): return self(batch['image'])

def height_loss(logits,gt,mask):
    mask=mask&(gt>-20)&(gt<20)
    assert mask.any(), 'No supervised cells'
    labels=79-torch.floor((gt[mask]+20)/.5).long()
    return nn.functional.cross_entropy(logits.float().permute(0,2,3,1)[mask],labels)
def prediction(logits):
    v=(19.75-torch.arange(80,device=logits.device)*.5)[None,:,None,None]
    return (logits.float().softmax(1)*v).sum(1)
def checkpoint(path,model,opt,epoch,history,cfg,step=0):
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=step,history=history,config=cfg,
                 torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                 numpy_rng=np.random.get_state(),python_rng=random.getstate())
    tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(path)

def smoke(cfg):
    seed_all(307); torch.set_num_threads(2); model=MonoREO(cfg).cuda().train()
    ds=Frames('train',cfg); b=next(iter(DataLoader(ds,batch_size=cfg['batch_size'])))
    x=b['image'].cuda(); gt=b['ele_gt'].cuda(); mask=b['ele_mask'].cuda()
    t=time.time(); torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(x); loss=height_loss(logits,gt,mask)
    assert logits.shape==(cfg['batch_size'],80,164,64) and torch.isfinite(loss)
    loss.backward(); grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    model.eval()
    class Guard(dict):
        def __getitem__(self,k):
            if k!='image': raise AssertionError('Forbidden model input '+k)
            return super().__getitem__(k)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        p=model.forward_batch(Guard(image=x,pose='clean',r_vert2cam='clean'))
        q=model.forward_batch(Guard(image=x,pose='changed',r_vert2cam='changed',R='changed',T='changed',proj_index='changed'))
    assert torch.equal(p,q)
    report=dict(passed=True,shape=list(logits.shape),loss=float(loss),finite_gradients=True,
                peak_allocated_GB=torch.cuda.max_memory_allocated()/2**30,seconds=time.time()-t,
                a1_max_abs_logits_difference=float((p-q).abs().max()),guarded_metadata_access=False,
                forward_signature=str(inspect.signature(model.forward)),initialization=model.initialization,
                params=sum(p.numel() for p in model.parameters()),config=cfg,
                aux_modules_removed=True,aux_target_reads=False)
    save_json(ROOT/'reports/model_smoke.json',report); print(json.dumps(report,indent=2),flush=True)

def train(cfg):
    torch.set_num_threads(2); seed_all(cfg['seed']); model=MonoREO(cfg).cuda()
    opt=torch.optim.AdamW([dict(params=model.backbone.parameters(),lr=cfg['backbone_lr']),
                          dict(params=model.height.parameters(),lr=cfg['lr'])],weight_decay=cfg['weight_decay'])
    ds=Frames('train',cfg); start_epoch=1; start_step=0; history=[]
    latest=ROOT/'checkpoints/latest.ckpt'; started=time.time()
    if latest.exists():
        c=torch.load(latest,map_location='cpu',weights_only=False); assert c['config']==cfg
        model.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']); history=c['history']
        start_epoch=c['epoch']+1 if c['step']==0 else c['epoch']; start_step=c['step']
        torch.set_rng_state(c['torch_rng']); torch.cuda.set_rng_state_all(c['cuda_rng'])
        np.random.set_state(c['numpy_rng']); random.setstate(c['python_rng'])
    else:
        save_json(ROOT/'reports/initialization.json',model.initialization)
    save_json(ROOT/'run_config.json',cfg)
    for epoch in range(start_epoch,51):
        model.train(); t=time.time(); total=0.; seen=0
        gen=torch.Generator().manual_seed(cfg['seed']+epoch)
        loader=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,generator=gen,
                          num_workers=cfg['num_workers'],drop_last=False,pin_memory=True)
        torch.cuda.reset_peak_memory_stats()
        for step,b in enumerate(loader):
            if epoch==start_epoch and step<start_step: continue
            x=b['image'].cuda(non_blocking=True); gt=b['ele_gt'].cuda(non_blocking=True); mask=b['ele_mask'].cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(x); loss=height_loss(logits,gt,mask)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss: no silent skip permitted')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
            opt.step(); total+=float(loss)*len(x); seen+=len(x)
            if (step+1)%100==0:
                print(json.dumps(dict(epoch=epoch,step=step+1,total_steps=len(loader),loss=float(loss),seen=seen,seconds=time.time()-t)),flush=True)
                checkpoint(latest,model,opt,epoch,history,cfg,step+1)
        row=dict(epoch=epoch,train_ce=total/seen,frames=seen,seconds=time.time()-t,
                 peak_allocated_GB=torch.cuda.max_memory_allocated()/2**30,resumed_partial=(epoch==start_epoch and start_step>0))
        history.append(row); checkpoint(latest,model,opt,epoch,history,cfg)
        save_json(ROOT/'logs/train_history.json',history); print(json.dumps(row),flush=True)
        if epoch%10==0: checkpoint(ROOT/('checkpoints/epoch%02d.ckpt'%epoch),model,opt,epoch,history,cfg)
        save_json(ROOT/'logs/progress.json',dict(stage='training',epoch=epoch,epochs=50,last_epoch=row))
    checkpoint(ROOT/'checkpoints/final_epoch50.ckpt',model,opt,50,history,cfg)
    save_json(ROOT/'reports/training_complete.json',dict(epochs=50,history=history,checkpoint=str(ROOT/'checkpoints/final_epoch50.ckpt'),
                   checkpoint_sha256=sha(ROOT/'checkpoints/final_epoch50.ckpt'),run_seconds_this_process=time.time()-started,
                   no_test_evaluation_during_training=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['smoke','train']); args=parser.parse_args()
    cfg=config()
    if (ROOT/'run_config.json').exists(): cfg=json.loads((ROOT/'run_config.json').read_text())
    (smoke if args.action=='smoke' else train)(cfg)
