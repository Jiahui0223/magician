"""C1 stochastic latent update adapter (configs.belief.C1Network).

    z_new = z_old + sigmoid(g) * delta,   (delta, g) = f(z_old, condition, noise, code, step)

on the dense latent grid of the frozen generator (factor 8, C channels; the grid of `Net.randn_like` that the
frozen VAE decoder reads). The network predicts an update of one old world hypothesis, not a new world:
- every latent cell is a token: in_proj(z_old, condition, noise) + 3D sine position (AbsolutePositionEmbedder
  of models.sparse.dit);
- blocks: adaLN-modulated windowed self-attention (cubic windows of `window_size` cells, shifted by half a window
  in odd blocks) -> cross-attention from the world tokens to observation tokens (the condition pooled to an
  obs_grid^3 coarse grid, so every cell sees the whole new observation) -> MLP. The modulation vector is the
  per-particle Gaussian code (MLP) + the observation step (TimestepEmbedder of models.sparse.dit);
- randomness: a per-cell Gaussian noise field (input) and a per-particle Gaussian code (modulation); both come from
  the caller, so an update is reproducible from its seeds and differentiable in the parameters;
- zero initialisation (adaLN modulation and output head, as DiT / ControlNet zero_module): at initialisation every
  block is the identity and delta = 0, so z_new == z_old exactly.
Attention uses torch scaled_dot_product_attention on padded windows (CPU and GPU). The official sparse attention
(models.sparse.attention) needs flash-attn / xformers and torchsparse SparseTensors (GPU only); the dense latent grid
does not need sparsity. No training loss is defined here.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from seen2scene.models.backbone.controlnet import zero_module
from seen2scene.models.sparse.dit import AbsolutePositionEmbedder, TimestepEmbedder
from seen2scene.models.sparse.transformer.blocks import LayerNorm32


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """DiT adaLN; x [B, ..., d], shift / scale [B, d]."""
    view = (x.shape[0],) + (1,) * (x.dim() - 2) + (x.shape[-1],)
    return x * (1 + scale.view(view)) + shift.view(view)


class Attention(nn.Module):
    """Multi-head self- (context None) or cross-attention with optional key mask (True = attend)."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "model_channels must be divisible by num_heads"
        self.h, self.dh = num_heads, dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_gain = nn.Parameter(torch.ones(self.dh))
            self.k_gain = nn.Parameter(torch.ones(self.dh))

    def _rms(self, x, gain):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * gain.to(x.dtype)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, d = x.shape
        ctx = x if context is None else context
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            q, k = self._rms(q, self.q_gain), self._rms(k, self.k_gain)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(o.transpose(1, 2).reshape(B, N, d))


def window_partition(x: torch.Tensor, w: int, offset: int):
    """[B, X, Y, Z, d] -> windows [B * nW, w^3, d], valid [B * nW, w^3] (False = padding), meta for the merge.
    The grid is padded by `offset` cells in front (shifted windows) and up to a multiple of w at the end."""
    B, X, Y, Z, d = x.shape
    pads = [(offset, (-(n + offset)) % w) for n in (X, Y, Z)]
    flat = (0, 0, pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1])
    xp = F.pad(x, flat)
    valid = F.pad(torch.ones(B, X, Y, Z, 1, device=x.device), flat)[..., 0] > 0.5
    nx, ny, nz = xp.shape[1] // w, xp.shape[2] // w, xp.shape[3] // w
    win = xp.view(B, nx, w, ny, w, nz, w, d).permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, w ** 3, d)
    valid = valid.view(B, nx, w, ny, w, nz, w).permute(0, 1, 3, 5, 2, 4, 6).reshape(-1, w ** 3)
    return win, valid, (B, X, Y, Z, d, w, pads, nx, ny, nz)


def window_merge(win: torch.Tensor, meta) -> torch.Tensor:
    B, X, Y, Z, d, w, pads, nx, ny, nz = meta
    xp = win.view(B, nx, ny, nz, w, w, w, d).permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, nx * w, ny * w, nz * w, d)
    return xp[:, pads[0][0]:pads[0][0] + X, pads[1][0]:pads[1][0] + Y, pads[2][0]:pads[2][0] + Z]


class C1Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float, cross: bool, qk_norm: bool):
        super().__init__()
        self.w = window_size
        self.norm1 = LayerNorm32(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, qk_norm)
        self.cross = Attention(dim, num_heads, qk_norm) if cross else None
        self.norm2 = LayerNorm32(dim, elementwise_affine=False, eps=1e-6) if cross else None
        self.norm3 = LayerNorm32(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor, obs: Optional[torch.Tensor], shift: bool) -> torch.Tensor:
        """x [B, X, Y, Z, d], c [B, d], obs [B, M, d] or None."""
        (sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3) = self.adaLN(c).chunk(9, dim=-1)
        B, X, Y, Z, d = x.shape
        gv = (B, 1, 1, 1, d)
        h = modulate(self.norm1(x), sh1, sc1)
        win, valid, meta = window_partition(h, self.w, self.w // 2 if shift else 0)
        valid = valid.clone()
        valid[~valid.any(1), 0] = True  # all-padding windows: attend to one (cropped) token, no NaN
        h = window_merge(self.attn(win, key_mask=valid), meta)
        x = x + g1.view(gv) * h
        if self.cross is not None and obs is not None:
            h = modulate(self.norm2(x), sh2, sc2).reshape(B, X * Y * Z, d)
            h = self.cross(h, context=obs).reshape(B, X, Y, Z, d)
            x = x + g2.view(gv) * h
        h = self.mlp(modulate(self.norm3(x), sh3, sc3))
        return x + g3.view(gv) * h


class C1LatentUpdateNet(nn.Module):
    def __init__(self, latent_channels: int = 8, cond_channels: int = 15, model_channels: int = 256,
                 num_blocks: int = 4, num_heads: int = 8, window_size: int = 4, mlp_ratio: float = 4.0,
                 noise_channels: int = 8, global_noise_dim: int = 64, obs_grid: int = 8, qk_norm: bool = True,
                 use_checkpoint: bool = False):
        super().__init__()
        d = model_channels
        self.latent_channels, self.cond_channels, self.noise_channels = latent_channels, cond_channels, noise_channels
        self.global_noise_dim, self.obs_grid, self.use_checkpoint = global_noise_dim, obs_grid, use_checkpoint
        self.in_proj = nn.Linear(latent_channels + cond_channels + noise_channels, d)
        self.pos = AbsolutePositionEmbedder(d, 3)
        self.obs_proj = nn.Linear(cond_channels, d) if obs_grid > 0 else None
        self.step_embed = TimestepEmbedder(d)
        self.code_mlp = nn.Sequential(nn.Linear(global_noise_dim, d), nn.SiLU(), nn.Linear(d, d)) \
            if global_noise_dim > 0 else None
        self.blocks = nn.ModuleList([C1Block(d, num_heads, window_size, mlp_ratio, obs_grid > 0, qk_norm)
                                     for _ in range(num_blocks)])
        self.head_norm = LayerNorm32(d, elementwise_affine=False, eps=1e-6)
        self.head_mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.head = nn.Linear(d, latent_channels + 1)
        self.initialize_weights()

    def initialize_weights(self, identity: bool = True, std: float = 0.02):
        """Xavier linear layers; identity=True zeroes adaLN modulations and the head (z_new == z_old), else they get
        small random values (tests / ablations of a non-trivial update)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.step_embed.mlp[0].weight, std=std)
        nn.init.normal_(self.step_embed.mlp[2].weight, std=std)
        zeroed = [b.adaLN[-1] for b in self.blocks] + [self.head_mod[-1], self.head]
        for m in zeroed:
            if identity:
                zero_module(m)
            else:
                nn.init.normal_(m.weight, std=std)
                nn.init.normal_(m.bias, std=std)

    def _obs_tokens(self, cond: torch.Tensor) -> Optional[torch.Tensor]:
        if self.obs_proj is None:
            return None
        g = self.obs_grid
        B, _, X, Y, Z = cond.shape
        pooled = F.adaptive_avg_pool3d(cond.float(), g)  # [B, Cc, g, g, g]
        tok = self.obs_proj(pooled.flatten(2).transpose(1, 2).to(self.obs_proj.weight.dtype))
        ax = [(torch.arange(g, device=cond.device) + 0.5) * n / g - 0.5 for n in (X, Y, Z)]
        pos = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        return tok + self.pos(pos)[None].to(tok.dtype)

    def forward(self, z_old: torch.Tensor, cond: torch.Tensor, noise: Optional[torch.Tensor] = None,
                code: Optional[torch.Tensor] = None, step: Optional[torch.Tensor] = None,
                update_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """z_old [B, C, X, Y, Z], cond [B, Cc, X, Y, Z], noise [B, Cn, X, Y, Z] (None = 0), code [B, G] (None = 0),
        step [B] observation count (None = 0), update_mask [B, 1, X, Y, Z] in [0, 1] (None = everywhere).
        Returns z_new, delta [B, C, X, Y, Z] and gate [B, 1, X, Y, Z]."""
        B, C, X, Y, Z = z_old.shape
        if C != self.latent_channels or cond.shape[1] != self.cond_channels or cond.shape[2:] != z_old.shape[2:]:
            raise ValueError(f"z_old {tuple(z_old.shape)} / cond {tuple(cond.shape)} do not match the network")
        dev = z_old.device
        if noise is None:
            noise = torch.zeros(B, self.noise_channels, X, Y, Z, device=dev, dtype=z_old.dtype)
        if code is None and self.code_mlp is not None:
            code = torch.zeros(B, self.global_noise_dim, device=dev, dtype=z_old.dtype)
        step = torch.zeros(B, device=dev) if step is None else step.to(dev)
        inp = torch.cat([z_old, cond.to(z_old.dtype), noise.to(z_old.dtype)], 1).permute(0, 2, 3, 4, 1)
        x = self.in_proj(inp)
        idx = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev) for n in (X, Y, Z)], indexing="ij"), -1)
        x = x + self.pos(idx.reshape(-1, 3).float()).reshape(1, X, Y, Z, -1).to(x.dtype)
        c = self.step_embed(step.float())
        if self.code_mlp is not None:
            c = c + self.code_mlp(code.to(c.dtype))
        obs = self._obs_tokens(cond)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                x = checkpoint(blk, x, c, obs, i % 2 == 1, use_reentrant=False)
            else:
                x = blk(x, c, obs, i % 2 == 1)
        shift, scale = self.head_mod(c).chunk(2, dim=-1)
        out = self.head(modulate(self.head_norm(x), shift, scale)).permute(0, 4, 1, 2, 3)
        delta, gate = out[:, :C].to(z_old.dtype), torch.sigmoid(out[:, C:].float()).to(z_old.dtype)
        upd = gate * delta
        if update_mask is not None:
            upd = upd * update_mask.to(upd.dtype)
        return {"z_new": z_old + upd, "delta": delta, "gate": gate}
