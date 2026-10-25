"""exp29: attention-guided fine sampling (calibration-free).

For every full-resolution BEV cell (164x64) a per-frame image reference point is the
soft-argmax of a dedicated attention over the pooled image tokens (centroid of the
token grid weighted by attention). Around that reference, J learned offsets sample
the un-pooled 1/4 backbone feature map; the result is added to the cell feature
before bev_refine/bev_head. No K, R, pose or projection index enters forward: the
reference is a function of image content and the learned BEV query only.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from reo_core import nerf_positional_encoding
from improved import Improved
from rsrd_reo import prediction as classify


class AttnGuidedSampler(nn.Module):
    def __init__(self, dim, in_ch, pool_size, fine_hw, heads=4, points=4, max_px=24):
        super().__init__()
        self.h, self.j, self.dim = heads, points, dim
        self.fine_proj = nn.Conv2d(in_ch, dim, 1)
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim)
        ph, pw = pool_size
        # Token centres in grid_sample(align_corners=False) coords: pooled bins of the fine map.
        ys = -1 + (2 * torch.arange(ph).float() + 1) / ph; xs = -1 + (2 * torch.arange(pw).float() + 1) / pw
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('token_xy', torch.stack([gx, gy], -1).reshape(-1, 2), persistent=False)
        fh, fw = fine_hw
        self.register_buffer('px_scale', torch.tensor([2 * max_px / fw, 2 * max_px / fh]), persistent=False)
        self.offset = nn.Linear(dim, heads * points * 2)
        self.weight = nn.Linear(dim, heads * points)
        nn.init.zeros_(self.offset.weight)
        stencil = torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])[:points] * (2 / max_px)  # +-2 fine px
        with torch.no_grad():
            self.offset.bias.copy_(torch.atanh(stencil.clamp(-.99, .99)).repeat(heads, 1).reshape(-1))
        self.out = nn.Linear(dim, dim); self.norm = nn.LayerNorm(dim)

    def references(self, query, tokens):
        b, n, d = query.shape; h = self.h; dh = d // h
        q = self.q(query).view(b, n, h, dh).transpose(1, 2)
        k = self.k(tokens).view(b, -1, h, dh).transpose(1, 2)
        v = torch.zeros(b, h, tokens.shape[1], dh, device=query.device, dtype=q.dtype)
        v[..., :2] = self.token_xy.to(q.dtype)
        # Soft-argmax through SDPA keeps memory linear in N (no N x T matrix stored).
        return F.scaled_dot_product_attention(q, k, v)[..., :2].float()  # [B,h,N,2]

    def forward(self, query, tokens, fine):
        b, n, _ = query.shape
        ref = self.references(query, tokens)
        off = self.offset(query).float().view(b, n, self.h, self.j, 2).tanh() * self.px_scale
        grid = ref.permute(0, 2, 1, 3)[:, :, :, None, :] + off           # [B,N,h,J,2]
        w = self.weight(query).float().softmax(-1)                        # [B,N,hJ]
        with torch.autocast(device_type=query.device.type, enabled=False):
            s = F.grid_sample(fine.float(), grid.reshape(b, n, -1, 2), mode='bilinear',
                              padding_mode='zeros', align_corners=False)  # [B,D,N,hJ]
            g = (s * w[:, None]).sum(-1).transpose(1, 2)                  # [B,N,D]
        return self.norm(self.out(g.to(query.dtype))), ref


class Exp29(Improved):
    def __init__(self, cfg, initialize=True):
        super().__init__(cfg, initialize)
        H = self.height
        self.sampler = AttnGuidedSampler(cfg['dim'], 64, H.pool_size, tuple(cfg['fine_hw']),
                                         heads=cfg['sampler_heads'], points=cfg['sampler_points'], max_px=cfg['sampler_max_px'])

    def components(self, image, return_ref=False):
        H = self.height; feature = self.backbone(image); b = feature.shape[0]
        tokens, _ = H.image_encoder(feature)
        nz, nx = H.attn_grid
        coords = H.build_query_grid(b, feature.device, feature.dtype, (nz, nx))
        query = H.query_norm(H.query_embed(nerf_positional_encoding(coords, H.image_encoder.num_freqs, bounds=H.coord_bounds)))
        query = query + H.query_pos_embed[:, :query.shape[1]]
        outs = []
        for s0 in range(0, query.shape[1], H.query_chunk):
            piece = query[:, s0:s0 + H.query_chunk]
            for block in H.blocks:
                if H.grad_ckpt and self.training and torch.is_grad_enabled():
                    piece = checkpoint(lambda q, m, bl=block: bl(q, m)[0], piece, tokens, use_reentrant=True)
                else:
                    piece, _ = block(piece, tokens)
            outs.append(piece)
        query = torch.cat(outs, 1)
        bev = F.interpolate(query.transpose(1, 2).reshape(b, -1, nz, nx), size=(H.num_grids_z, H.num_grids_x),
                            mode='bilinear', align_corners=False)
        cell = bev.flatten(2).transpose(1, 2)                                   # [B,10496,D]
        fine = self.sampler.fine_proj(feature)
        if H.grad_ckpt and self.training and torch.is_grad_enabled():
            add, ref = checkpoint(self.sampler, cell, tokens, fine, use_reentrant=False)
        else:
            add, ref = self.sampler(cell, tokens, fine)
        bev = bev + add.transpose(1, 2).reshape_as(bev)
        logits = H.bev_head(H.bev_refine(bev))
        out = dict(logits=logits, prediction=classify(logits))
        if return_ref: out['ref'] = ref
        return out


def project_reference(gt, mask, r, k, centers, fine_hw):
    """Training-only target: grid_sample coords of each BEV cell centre at its GT height
    (0 where unlabeled) under the frame's (possibly augmented) rotation. Never used at inference."""
    import numpy as np
    h = np.where(mask, gt, 0.0).reshape(-1)
    xyz = np.c_[centers[:, 0], 1.1 - h / 100, centers[:, 1]].T
    cam = r @ xyz; uvz = k @ cam; uv = uvz[:2] / uvz[2:]
    # image px (960x528 crop) -> grid_sample coords on the same image extent (align_corners=False)
    gx = 2 * (uv[0] + .5) / 960 - 1; gy = 2 * (uv[1] + .5) / 528 - 1
    ok = (cam[2] > 0) & np.isfinite(gx) & np.isfinite(gy) & (np.abs(gx) < 1) & (np.abs(gy) < 1)
    return np.stack([gx, gy], -1).astype(np.float32), ok
