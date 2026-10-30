"""V3Model (EXP6): 3D VAE scene encoder -> latent ray encoder -> 2D rectified-flow U-Net (T0), plus one camera-ray
cross-attention block at the U-Net bottleneck (T1). Tensor conventions follow the V2 model (seen2scene.futureview.
model): views of one state [V, C, 64, 128] (114 columns + zero padding), rays in the model (region / scene) frame.

encode(x, valid)         -> h = dict(mu, logvar, valid8, band8, free8, ext)        (one state)
view_cond(h, o, d, tcap, ch) -> cond [V, C, 64, 128] (+ mem for T1)                (V candidate views of the state)
velocity(x, tau, cond, mem) -> [V, 2, 64, 128]
decode(z)                -> [B, 4, X4, Y4, Z4] = [sdf_hat, logits EMPTY / UNKNOWN / BAND]  (training only)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from seen2scene.futureview import model as M
from seen2scene.v3 import cross_attention as XA
from seen2scene.v3 import partial_tsdf as PT
from seen2scene.v3.flow_unet import FlowUNet
from seen2scene.v3.vae import VAEDecoder, VAEEncoder

GH, GW, GW_PAD = M.GH, M.GW, M.GW_PAD


class LatentRayEncoder(M.RayEncoder):
    """The V2 ray encoder (same Conv1d stack) fed by the VAE latent: per point [mu (C), valid8 * inside]."""

    def forward(self, mu, valid8, ext, o, d, tcap):
        p = self.points(o, d, tcap)
        g = 2.0 * p / ext.to(p.device, p.dtype) - 1.0
        grid = torch.stack([g[..., 2], g[..., 1], g[..., 0]], -1)[None, :, :, None].float()       # [1, R, N, 1, 3]
        with torch.autocast("cuda", enabled=False):
            f0 = F.grid_sample(mu.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            fv = F.grid_sample(valid8.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
        inside = ((g > -1) & (g < 1)).all(-1)[None, None, ..., None].float()
        f = torch.cat([f0, (fv > 0).float() * inside], 1)[0, ..., 0].permute(1, 0, 2)              # [R, C + 1, N]
        h = F.silu(self.c1(f))
        h = F.silu(self.c2(h))
        h = F.silu(self.c3(h))
        return self.fc(h.flatten(1))


class V3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        v, r, u, xa = cfg["vae"], cfg["ray"], cfg["unet"], cfg.get("xattn", {"enabled": False})
        self.vae_enc = VAEEncoder(v["in_ch"], tuple(v["ch"]), v["latent_ch"], v["groups"], v["act_ckpt"])
        self.vae_dec = VAEDecoder(v["latent_ch"], tuple(v["ch"]), v["groups"], 4, v["act_ckpt"])
        self.rays = LatentRayEncoder(r["n_points"], v["latent_ch"], r["hidden"], r["out_ch"])     # base adds the +1 valid channel
        self.x_ch, self.proj_ch, self.ray_ch = cfg["flow"]["x_ch"], cfg["proj_ch"], r["out_ch"]
        self.cond_ch = self.proj_ch + self.ray_ch + 1
        self.xa = dict(xa) if xa.get("enabled") else None
        if self.xa:
            self.xa["mem_dim"] = v["latent_ch"] + 2 + 6 * xa["n_freq"]
            self.xa["q_pe_dim"] = 12 * xa["n_freq"]
        self.unet = FlowUNet(self.x_ch + self.cond_ch, self.x_ch, tuple(u["ch"]), u["blocks_per_level"], tuple(u["attn_levels"]),
                             u["heads"], u["t_emb"], u["groups"], xattn=self.xa)
        self.rays_ckpt = True

    # ------------------------------------------------------------------------------------------------ scene side
    def encode(self, x, valid):
        """x [1, 3, X4, Y4, Z4], valid [1, 1, ...] -> h (mu is the flow condition; logvar for the training terms)."""
        mu, logvar, valid8 = self.vae_enc(x, valid)
        band8 = F.max_pool3d(x[:, 1:2].float(), 2)
        free8 = F.max_pool3d(x[:, 2:3].float(), 2)
        return {"mu": mu, "logvar": logvar, "valid8": valid8, "band8": band8, "free8": free8, "ext": PT.extent(x.shape[2:])}

    def decode(self, z):
        return self.vae_dec(z)

    def memory(self, h):
        return XA.SceneMemory(h["mu"], h["valid8"], h["band8"], h["free8"], h["ext"]) if self.xa else None

    # ------------------------------------------------------------------------------------------------- view side
    def view_cond(self, h, o, d, tcap, ch, mem=None):
        """o [V, 3], d [V, R, 3], tcap [V, R], ch [V, 11, R] -> cond [V, C, 64, 128]; T1 also returns the memory
        tuple (tokens [V, budget, F], mask [V, budget], query pe [V, L, P]) built from mem (SceneMemory of h)."""
        V, R = d.shape[:2]
        args = (h["mu"], h["valid8"], h["ext"], o[:, None].expand(V, R, 3).reshape(-1, 3), d.reshape(-1, 3), tcap.reshape(-1))
        if self.training and torch.is_grad_enabled() and self.rays_ckpt:
            rf = torch.utils.checkpoint.checkpoint(self.rays, *args, use_reentrant=False)
        else:
            rf = self.rays(*args)
        c = torch.cat([ch.to(rf.dtype), rf.reshape(V, R, -1).transpose(1, 2)], 1).reshape(V, -1, GH, GW)
        c = F.pad(c, (0, GW_PAD - GW))
        valid = torch.zeros(V, 1, GH, GW_PAD, device=c.device, dtype=c.dtype)
        valid[..., :GW] = 1
        cond = torch.cat([c, valid], 1)
        if not self.xa:
            return cond
        mem = mem if mem is not None else self.memory(h)
        tok, mask = mem.batch([(o[i], d[i], tcap[i]) for i in range(V)], self.xa["budget"], self.xa["fov_margin_deg"], self.xa["n_freq"])
        qpe = torch.stack([XA.query_pe(o[i], d[i], h["ext"], self.xa["n_freq"]) for i in range(V)])
        return cond, (tok, mask, qpe)

    def velocity(self, x, tau, cond, mem=None, return_features=False):
        return self.unet(torch.cat([x, cond.to(x.dtype)], 1), tau, mem, return_features)


def index_mem(mem, idx):
    return None if mem is None else tuple(m[idx] for m in mem)


@torch.no_grad()
def sample(model, cond, m_unk, eps, mem=None, steps=10, autocast=True, return_features=False):
    """V3 version of futureview.model.sample: Euler integration with the optional scene memory."""
    x = eps * m_unk[:, None]
    feats = None
    for i in range(steps):
        tau = torch.full((x.shape[0],), i / steps, device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = model.velocity(x, tau, cond, mem, return_features=return_features and i == steps - 1)
        if return_features and i == steps - 1:
            out, feats = out
        x = (x + out.float() / steps) * m_unk[:, None]
    return (x, feats) if return_features else x


def common_state(model):
    """State dict of the modules shared by T0 and T1 (everything but the cross-attention block)."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("unet.xattn.")}


def state_hash(sd):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def param_counts(model):
    tot = sum(p.numel() for p in model.parameters())
    xa = sum(p.numel() for n, p in model.named_parameters() if n.startswith("unet.xattn."))
    dec = sum(p.numel() for n, p in model.named_parameters() if n.startswith("vae_dec."))
    return {"total": tot, "xattn": xa, "vae_decoder_train_only": dec, "inference": tot - dec}
