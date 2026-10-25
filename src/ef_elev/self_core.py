"""Minimal calibration-free BEV height model (REO-style) for the reo experiment.

Design (see notes/DECISIONS.md for the evidence behind each choice)
-------------------------------------------------------------------
Calibration-free forward path
  The network never receives K, R, t, a baseline, or any per-frame pose.
  BEV query (x, z) come from the fixed ROI grid with y = 0 (the plane the
  residual GT is defined against), so no extrinsic enters the forward pass.

Three supervision heads
  main   BEV height classification over the ROI grid   <- the task target
  depth  per-pixel depth (REO's auxiliary geometry task) <- gives image
         features a notion of 3D position so cross-attention can place them
  rgb    per-pixel RGB reconstruction (REO's regulariser)
  Semantics are deliberately omitted: the cloud only carries a ground /
  non-ground label, so a semantic head would add noise rather than signal.

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def nerf_positional_encoding(coords, num_freqs=8, temperature=1000.0, bounds=None):
    """Fourier features for 3D query coordinates.

    ``coords``: [B, N, 3] -> [B, N, 3 * 2 * num_freqs]

    The previous version applied the original NeRF schedule
    (``temperature=1000``) directly to metre-valued coordinates. Its highest
    frequency is only ~1 rad/m while the BEV grid pitch is 3 cm, so adjacent
    queries were 0.99998 cosine-similar and all 16000 queries read essentially
    the same content; the head then collapsed to a constant.

    Coordinates are therefore normalised to [-1, 1] per axis (using ``bounds``
    when given, else the batch extent) and the frequency ladder is 2^k * pi,
    which resolves the grid pitch at every scale. Measured effect on the real
    ROI grid: adjacent-query cosine drops 0.99998 -> 0.785.
    """
    if bounds is not None:
        low = torch.as_tensor(bounds[0], device=coords.device, dtype=coords.dtype)
        high = torch.as_tensor(bounds[1], device=coords.device, dtype=coords.dtype)
    else:
        low = coords.amin(dim=1, keepdim=True)
        high = coords.amax(dim=1, keepdim=True)
    normalised = 2.0 * (coords - low) / (high - low).clamp_min(1.0e-6) - 1.0
    freq = (2.0 ** torch.arange(num_freqs, device=coords.device, dtype=coords.dtype)) * math.pi
    scaled = normalised.unsqueeze(-1) * freq
    encoded = torch.cat((scaled.sin(), scaled.cos()), dim=-1)
    return encoded.reshape(coords.shape[0], coords.shape[1], -1)


class CrossAttentionBlock(nn.Module):
    """Self-attention over BEV queries followed by cross-attention to image tokens."""

    def __init__(self, dim=384, num_heads=4, mlp_ratio=1, dropout=0.1, use_self_attn=True):
        super().__init__()
        self.use_self_attn = bool(use_self_attn)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.self_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.cross_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, query, image_tokens):
        if self.use_self_attn:
            attended, _ = self.self_attn(query, query, query)
            query = self.self_norm(attended + query)
        # Cross-attention is the calibration-free 2D->BEV write, so it always
        # runs. Self-attention over every BEV cell is what blows up memory at
        # 16000 queries, so it is optional.
        attended, attn_weights = self.cross_attn(query, image_tokens, image_tokens, need_weights=False)
        query = self.cross_norm(attended + query)
        query = self.ffn_norm(self.ffn(query) + query)
        return query, attn_weights


class ImageTokenEncoder(nn.Module):
    """Turns a feature map into a token sequence carrying 2D positional info."""

    def __init__(self, in_channels, dim=384, num_freqs=8, pool_size=(33, 60)):
        super().__init__()
        self.pool_size = tuple(int(v) for v in pool_size)
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.num_freqs = num_freqs
        self.norm = nn.LayerNorm(dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(2 * 2 * num_freqs, dim), nn.LayerNorm(dim)
        )
        self.token_pos_embed = nn.Parameter(
            0.02 * torch.randn(1, self.pool_size[0] * self.pool_size[1], dim)
        )

    def forward(self, feature):
        projected = self.proj(feature)  # [B, D, H, W]
        # Pool before attention so the token count is bounded. At full
        # resolution the token map is 126720 long, and even a 512-query chunk
        # then needs ~6.5 GB for one attention matrix.
        projected = F.adaptive_avg_pool2d(projected, self.pool_size)
        batch, dim, height, width = projected.shape
        ys = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack((grid_x, grid_y), dim=-1).reshape(1, height * width, 2)
        # 2^k*pi on [-1, 1] coords. The old 1000-temperature ladder left
        # adjacent tokens 0.99999 cosine-similar, and in a calibration-free
        # model the image row is the main cue for distance.
        freq = (2.0 ** torch.arange(self.num_freqs, device=feature.device, dtype=feature.dtype)) * math.pi
        scaled = coords.unsqueeze(-1) * freq
        pos = torch.cat((scaled.sin(), scaled.cos()), dim=-1).reshape(1, height * width, -1)
        tokens = projected.flatten(2).transpose(1, 2)  # [B, HW, D]
        # Learned per-token embedding, as REO adds to its image patches.
        tokens = self.norm(tokens) + self.pos_proj(pos) + self.token_pos_embed
        return tokens, pos


class ReoHeightNet(nn.Module):
    """Calibration-free BEV height predictor with REO-style auxiliary heads."""

    def __init__(
        self,
        in_channels,
        num_grids,
        num_classes,
        dim=256,
        num_heads=4,
        num_layers=2,
        num_freqs=8,
        use_self_attn=True,
        query_chunk=1024,
        pool_size=(66, 120),
        attn_grid=(40, 100),
        roi_x=(-1.2, 1.2),
        roi_z=(4.0, 10.0),
        grid_res=(0.03, 0.03),
        aux_dim=64,
        num_cams=1,
        grad_ckpt=False,
    ):
        super().__init__()
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.num_classes = num_classes
        self.roi_x = roi_x
        self.roi_z = roi_z
        self.register_buffer('grid_res', torch.tensor(grid_res, dtype=torch.float32), persistent=False)

        self.query_chunk = int(query_chunk)
        self.pool_size = tuple(int(v) for v in pool_size)
        self.image_encoder = ImageTokenEncoder(in_channels, dim, num_freqs, pool_size=self.pool_size)
        self.query_norm = nn.LayerNorm(dim)
        self.query_embed = nn.Linear(3 * 2 * num_freqs, dim)
        # Coordinate bounds used to normalise the positional encoding. Fixed
        # from the ROI so the encoding does not depend on the batch extent.
        self.register_buffer(
            'coord_bounds',
            torch.tensor([[roi_x[0], -1.0, roi_z[0]], [roi_x[1], 1.0, roi_z[1]]],
                         dtype=torch.float32),
            persistent=False,
        )
        # Stereo (num_cams=2): left and right tokens are concatenated as the
        # cross-attention memory, each tagged with a learned camera-ID embedding.
        # No extrinsic is used: the network has to learn the L/R correspondence.
        self.num_cams = int(num_cams)
        # Recompute attention blocks in backward: stereo doubles the memory
        # length and bs3 no longer fits in 40 GB otherwise. Numerically identical.
        self.grad_ckpt = bool(grad_ckpt)
        if self.num_cams > 1:
            self.cam_embed = nn.Parameter(0.02 * torch.randn(self.num_cams, 1, dim))
        self.attn_grid = (int(attn_grid[0]), int(attn_grid[1]))
        self.query_pos_embed = nn.Parameter(
            0.02 * torch.randn(1, self.attn_grid[0] * self.attn_grid[1], dim)
        )
        self.bev_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, use_self_attn=use_self_attn)
            for _ in range(num_layers)
        ])

        self.bev_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.SiLU(inplace=True),
            nn.Conv2d(dim, num_classes, 1),
        )

        self.aux_depth = nn.Sequential(
            nn.Conv2d(dim, aux_dim, 3, padding=1), nn.GroupNorm(8, aux_dim), nn.SiLU(inplace=True),
            nn.Conv2d(aux_dim, 1, 1),
        )
        self.aux_rgb = nn.Sequential(
            nn.Conv2d(dim, aux_dim, 3, padding=1), nn.GroupNorm(8, aux_dim), nn.SiLU(inplace=True),
            nn.Conv2d(aux_dim, 3, 1),
        )

    def build_query_grid(self, batch, device, dtype, grid_hw=None):
        """BEV query coordinates on a (possibly coarse) grid.

        ``grid_hw`` = (n_z, n_x) lets attention run on a coarse grid while the
        output is later upsampled to the full 200x80 ROI. Running cross-attention
        at the full 3 cm grid needs 16000 queries, which forced self-attention
        off and batch size down to 1.

        The vertical coordinate is exactly 0: the GT payload is
        100 * (plane_y(x, z) - y_vertical), so a residual of 0 means "on the
        fitted plane". A learned offset here had no supervision, because the
        residual GT has zero mean.
        """
        n_z, n_x = grid_hw if grid_hw is not None else (self.num_grids_z, self.num_grids_x)
        res_x, res_z = float(self.grid_res[0]), float(self.grid_res[1])
        # Sample the coarse grid at cell centres of the full ROI so the coarse
        # queries tile the same physical extent.
        xs = self.roi_x[0] + (torch.arange(n_x, device=device, dtype=dtype) + 0.5) * (
            (self.roi_x[1] - self.roi_x[0]) / n_x
        )
        zs = self.roi_z[1] - (torch.arange(n_z, device=device, dtype=dtype) + 0.5) * (
            (self.roi_z[1] - self.roi_z[0]) / n_z
        )
        grid_z, grid_x = torch.meshgrid(zs, xs, indexing='ij')
        offset = torch.zeros(n_z * n_x, device=device, dtype=dtype)
        coords = torch.stack((grid_x.reshape(-1), offset, grid_z.reshape(-1)), dim=-1)
        return coords.unsqueeze(0).expand(batch, -1, -1)

    def forward(self, image_feature, return_aux=True, image_feature_right=None):
        batch = image_feature.shape[0]
        device, dtype = image_feature.device, image_feature.dtype

        tokens, _ = self.image_encoder(image_feature)
        memory = tokens
        if self.num_cams > 1:
            if image_feature_right is None:
                raise ValueError('num_cams=2 model needs image_feature_right')
            tokens_right, _ = self.image_encoder(image_feature_right)
            memory = torch.cat((tokens + self.cam_embed[0], tokens_right + self.cam_embed[1]), dim=1)

        # Coarse attention grid, then convolutional upsampling to the ROI.
        attn_nz = min(self.attn_grid[0], self.num_grids_z)
        attn_nx = min(self.attn_grid[1], self.num_grids_x)
        coords = self.build_query_grid(batch, device, dtype, (attn_nz, attn_nx))
        encoded = nerf_positional_encoding(
            coords, self.image_encoder.num_freqs, bounds=self.coord_bounds
        )
        query = self.query_norm(self.query_embed(encoded))
        # Learned per-cell embedding: without it every query is a pure function
        # of its coordinate, which made the head collapse to a constant.
        query = query + self.query_pos_embed[:, : query.shape[1]]

        chunk = max(int(self.query_chunk), 1)
        outputs = []
        for start in range(0, query.shape[1], chunk):
            piece = query[:, start:start + chunk]
            for block in self.blocks:
                if self.grad_ckpt and self.training and torch.is_grad_enabled():
                    piece = checkpoint(lambda q, m, b=block: b(q, m)[0], piece, memory)
                else:
                    piece, _ = block(piece, memory)
            outputs.append(piece)
        query = torch.cat(outputs, dim=1)

        coarse = query.transpose(1, 2).reshape(batch, -1, attn_nz, attn_nx)
        # Upsample the attended BEV field to the full ROI resolution.
        bev = F.interpolate(
            coarse, size=(self.num_grids_z, self.num_grids_x),
            mode='bilinear', align_corners=False,
        )
        bev = self.bev_refine(bev)
        bev_logits = self.bev_head(bev)  # [B, num_classes, Z, X]

        if not return_aux:
            return bev_logits, None

        pool_h, pool_w = self.image_encoder.pool_size
        # Aux heads read the left tokens only (the pseudo depth is left-camera).
        token_map = tokens.transpose(1, 2).reshape(batch, -1, pool_h, pool_w)
        # Only tensors the losses consume are returned; keeping attention or
        # token maps costs more memory than the rest of the network.
        aux = {
            'depth': self.aux_depth(token_map),
            'rgb': self.aux_rgb(token_map),
        }
        return bev_logits, aux
