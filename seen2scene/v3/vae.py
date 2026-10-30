"""Simple 3D convolutional VAE over the 4x partial-TSDF grid (EXP6 V3, both models).

Encoder: [sdf, band, free, valid] (4 ch at 4x fine voxels) -> stem c0 -> Res3D -> stride-2 c1 -> 2 x Res3D ->
1x1 conv -> (mu, logvar) with latent_ch channels at 8x fine voxels (one latent cell = 8 VS = 2 input cells).
Decoder (training only): z -> c1 -> 2 x Res3D -> nearest x2 + conv c0 -> Res3D -> conv -> [sdf_hat, 3 class logits]
at the input resolution. The decoder reconstructs the observed state of the SAME map; it never completes the scene.
Train: z_rec = mu + exp(0.5 logvar) * eps feeds the decoder; z_cond = mu feeds the ray encoder / flow (so L_FM reaches
mu only, L_rec and KL reach mu and logvar through the reparameterisation; by design).
KL per valid latent unit and channel: mean over valid units of 0.5 (mu^2 + exp(logvar) - logvar - 1), averaged over
channels, so the term does not scale with the grid size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from seen2scene.futureview.model import Res3D, gn


class VAEEncoder(nn.Module):
    def __init__(self, in_ch=4, ch=(32, 64), latent_ch=16, groups=8, act_ckpt=True):
        super().__init__()
        c0, c1 = ch
        self.stem = nn.Conv3d(in_ch, c0, 3, padding=1)
        self.r0 = Res3D(c0, groups)
        self.down = nn.Conv3d(c0, c1, 3, stride=2, padding=1)
        self.r1 = nn.Sequential(Res3D(c1, groups), Res3D(c1, groups))
        self.head = nn.Sequential(gn(c1, groups), nn.SiLU(), nn.Conv3d(c1, 2 * latent_ch, 1))
        self.latent_ch, self.act_ckpt = latent_ch, act_ckpt
        nn.init.zeros_(self.head[-1].bias)

    def _full_res(self, x):
        return self.r0(self.stem(x))

    def forward(self, x, valid):
        """x [B, 3, X4, Y4, Z4], valid [B, 1, X4, Y4, Z4] -> mu, logvar [B, latent_ch, X4/2, Y4/2, Z4/2], valid8 [B, 1, ...]."""
        inp = torch.cat([x, valid.to(x.dtype)], 1)
        if self.act_ckpt and self.training and torch.is_grad_enabled():
            h = torch.utils.checkpoint.checkpoint(self._full_res, inp, use_reentrant=False)
        else:
            h = self._full_res(inp)
        h = self.r1(self.down(F.silu(h)))
        mu, logvar = self.head(h).chunk(2, 1)
        valid8 = (F.avg_pool3d(valid.float(), 2) > 0).to(x.dtype)
        return mu * valid8, logvar.clamp(-10.0, 10.0) * valid8, valid8


class VAEDecoder(nn.Module):
    def __init__(self, latent_ch=16, ch=(32, 64), groups=8, out_ch=4, act_ckpt=True):
        super().__init__()
        c0, c1 = ch
        self.inp = nn.Conv3d(latent_ch, c1, 3, padding=1)
        self.r1 = nn.Sequential(Res3D(c1, groups), Res3D(c1, groups))
        self.up = nn.Conv3d(c1, c0, 3, padding=1)
        self.r0 = Res3D(c0, groups)
        self.out = nn.Sequential(gn(c0, groups), nn.SiLU(), nn.Conv3d(c0, out_ch, 3, padding=1))
        self.act_ckpt = act_ckpt

    def _full_res(self, h):
        return self.out(self.r0(self.up(F.interpolate(h, scale_factor=2, mode="nearest"))))

    def forward(self, z):
        h = self.r1(self.inp(z))
        h = F.silu(h)
        if self.act_ckpt and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._full_res, h, use_reentrant=False)
        return self._full_res(h)


def reparameterize(mu, logvar, generator=None):
    eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu + torch.exp(0.5 * logvar) * eps


def kl_per_unit(mu, logvar, valid8):
    """Mean KL(q || N(0, I)) per valid latent unit and channel (scalar, float32); also the per-channel mean [C]."""
    kl = 0.5 * (mu.float() ** 2 + torch.exp(logvar.float()) - logvar.float() - 1.0)       # [B, C, ...]
    m = valid8.float()
    n = m.sum().clamp(min=1.0)
    per_ch = (kl * m).sum(dim=(0, 2, 3, 4)) / n
    return per_ch.mean(), per_ch


@torch.no_grad()
def posterior_stats(mu, logvar, valid8, active_thr=1e-2):
    """Diagnostics: std of mu over valid units per channel, mean logvar, active units (channels whose mu variance
    across units exceeds active_thr)."""
    m = valid8.bool().expand_as(mu)
    C = mu.shape[1]
    mu_c = mu.float().movedim(1, 0).reshape(C, -1)[:, m.movedim(1, 0).reshape(C, -1)[0]]
    lv_c = logvar.float().movedim(1, 0).reshape(C, -1)[:, m.movedim(1, 0).reshape(C, -1)[0]]
    var = mu_c.var(1) if mu_c.shape[1] > 1 else torch.zeros(C, device=mu.device)
    return {"mu_std_mean": float(var.sqrt().mean()), "logvar_mean": float(lv_c.mean()) if lv_c.numel() else 0.0,
            "active_units": int((var > active_thr).sum()), "latent_ch": int(C)}
