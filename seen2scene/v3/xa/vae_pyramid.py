"""Feature pyramid of the V3 VAE encoder (Experiment XA).

F4   the encoder's full-resolution feature: output of VAEEncoder._full_res = r0(stem([x, valid])), 32 channels on the
     4x grid (the VAE input grid, one cell = 4 fine voxels). An intermediate encoder feature; it is NOT regularised by
     the KL term and is not "mu at a higher resolution".
F8   mu (16 channels, one cell = 8 fine voxels), the KL-regularised latent that conditions the flow (unchanged).
F16  valid-aware 2x pooling of mu: mean of mu over the in-domain children of each 2 x 2 x 2 block (out-of-domain children
     have weight 0), with the block's band / free means, its observed-support fraction and its in-domain fraction kept as
     separate channels, so an unobserved region is never read as free space. The learnable projection of F16 tokens is the
     scale embedding of the attention block.
All three share the model frame of the 4x grid: cell (i, j, k) of a grid with cell size c spans [i c, (i + 1) c).
"""
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from seen2scene.v3 import partial_tsdf as PT


def encode_pyramid(model, x, valid):
    """V3Model.encode with F4 kept: the same modules and operations as VAEEncoder.forward (bitwise the same mu)."""
    enc = model.vae_enc
    inp = torch.cat([x, valid.to(x.dtype)], 1)
    if enc.act_ckpt and enc.training and torch.is_grad_enabled():
        f4 = torch.utils.checkpoint.checkpoint(enc._full_res, inp, use_reentrant=False)
    else:
        f4 = enc._full_res(inp)
    hh = enc.r1(enc.down(F.silu(f4)))
    mu, logvar = enc.head(hh).chunk(2, 1)
    valid8 = (F.avg_pool3d(valid.float(), 2) > 0).to(x.dtype)
    mu, logvar = mu * valid8, logvar.clamp(-10.0, 10.0) * valid8
    band8 = F.max_pool3d(x[:, 1:2].float(), 2)
    free8 = F.max_pool3d(x[:, 2:3].float(), 2)
    return {"mu": mu, "logvar": logvar, "valid8": valid8, "band8": band8, "free8": free8, "ext": PT.extent(x.shape[2:]),
            "f4": f4, "x": x, "valid4": valid}


def pool16(mu, valid8, band8, free8):
    """-> mu16 [1, C, X16, Y16, Z16], band16, free16, obs16 (observed-support fraction), vfrac16 (in-domain fraction),
    all float32; odd dims are padded at the high end with out-of-domain cells."""
    w = valid8.float()
    obs = (((band8 > 0) | (free8 > 0)) & (valid8 > 0)).float()
    pad = []
    for n in reversed(w.shape[2:]):
        pad += [0, int(n) % 2]

    def P(t):
        return F.avg_pool3d(F.pad(t, pad) if any(pad) else t, 2)
    vfrac = P(w)
    den = vfrac.clamp(min=1e-6)
    mu16 = P(mu.float() * w) / den * (vfrac > 0)
    return mu16, P(band8.float() * w) / den, P(free8.float() * w) / den, P(obs), vfrac
