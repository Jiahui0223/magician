"""XAModel (Experiment XA): V3Model with the cross-attention of T1 replaced by one XABlock at the 16 x 32 level of the flow
U-Net (after the second residual block of level 2 on the way down; the enriched feature feeds both the skip connection and
the path to the bottleneck). Every other module, tensor layout and the (cond, mem) interface are those of V3Model, so the
V3 trainer, data code, losses, sampler and predictor run unchanged:

  encode(x, valid)              -> h = V3 h + {f4, x, valid4}
  view_cond(h, o, d, tcap, ch)  -> cond [V, 76, 64, 128], mem = (G, Q, L) each with leading dim V (float32)
  velocity(x, tau, cond, mem)   -> [V, 2, 64, 128]

cfg["xa"] = {memory: "s" | "m", query: "2d" | "ray" | "interval", heads, n_freq, fov_margin_deg, budget_s, budget_f8,
budget_f16, n_local, max_intervals}. All new parameters are in unet.xattn (created last).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from seen2scene.futureview import model as M
from seen2scene.futureview.model import Res2D, UNet2D, timestep_embedding
from seen2scene.v3 import model as V3
from seen2scene.v3.vae import VAEDecoder, VAEEncoder
from seen2scene.v3.xa import memory as MEM
from seen2scene.v3.xa import queries as QR
from seen2scene.v3.xa.attention import XABlock
from seen2scene.v3.xa.vae_pyramid import encode_pyramid

GH, GW, GW_PAD = M.GH, M.GW, M.GW_PAD
N_FREQ_S = 4


def dims(cfg):
    v, r, xa = cfg["vae"], cfg["ray"], cfg["xa"]
    nf = xa["n_freq"]
    ray = r["out_ch"] + cfg["proj_ch"] + 3 + 18 * nf
    return {"g": v["latent_ch"] + 4 + 6 * nf, "q": {"2d": 12 * nf, "ray": ray, "interval": ray + xa["max_intervals"] + 6}[xa["query"]],
            "l": (v["ch"][0] + v["latent_ch"] + 3 + 2 + 2 * N_FREQ_S) if xa["memory"] == "m" else 0, "scales": 2 if xa["memory"] == "m" else 1}


class XAUNet(UNet2D):
    def __init__(self, in_ch, out_ch, ch, blocks, attn_levels, heads, t_emb, groups, make_block):
        super().__init__(in_ch, out_ch, ch, blocks, attn_levels, heads, t_emb, groups)
        self.xa_at = 2 * (blocks + 1) + blocks - 1          # index in self.down of the last residual block of level 2
        self.xattn = make_block(ch[2])                      # created last: the base layers keep the V3 initial values

    def forward(self, x, tau, mem=None, return_features=False):
        assert mem is not None, "XA needs the scene context"
        te = self.temb(timestep_embedding(tau))
        h = self.inp(x)
        hs = [h]
        for i, m in enumerate(self.down):
            if isinstance(m[0], Res2D):
                h = m[0](h, te)
                for a in m[1:]:
                    h = a(h)
            else:
                h = m[0](h)
            if i == self.xa_at:
                h = self.xattn(h, *mem)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, te)), te)
        for m in self.up:
            if isinstance(m[0], Res2D):
                h = m[0](torch.cat([h, hs.pop()], 1), te)
                for a in m[1:]:
                    h = a(h)
            else:
                h = m[1](m[0](h))
        out = self.out(h)
        return (out, h) if return_features else out


class XAModel(V3.V3Model):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        v, r, u = cfg["vae"], cfg["ray"], cfg["unet"]
        self.vae_enc = VAEEncoder(v["in_ch"], tuple(v["ch"]), v["latent_ch"], v["groups"], v["act_ckpt"])
        self.vae_dec = VAEDecoder(v["latent_ch"], tuple(v["ch"]), v["groups"], 4, v["act_ckpt"])
        self.rays = V3.LatentRayEncoder(r["n_points"], v["latent_ch"], r["hidden"], r["out_ch"])
        self.x_ch, self.proj_ch, self.ray_ch = cfg["flow"]["x_ch"], cfg["proj_ch"], r["out_ch"]
        self.cond_ch = self.proj_ch + self.ray_ch + 1
        self.xa = dict(cfg["xa"])
        dm = dims(cfg)
        self.unet = XAUNet(self.x_ch + self.cond_ch, self.x_ch, tuple(u["ch"]), u["blocks_per_level"], tuple(u["attn_levels"]), u["heads"], u["t_emb"],
                           u["groups"], lambda c: XABlock(c, dm["q"], dm["g"], dm["scales"], dm["l"], self.xa["heads"], per_cell=self.xa["query"] != "interval"))
        self.rays_ckpt = True
        self.xa_stats = None                                 # set to [] to collect the interval statistics of every view_cond call

    def encode(self, x, valid):
        return encode_pyramid(self, x, valid)

    def memory(self, h):
        raise RuntimeError("XAModel builds its context in view_cond")

    def context(self, h, o, d, tcap, ch, rf):
        xa = self.xa
        V = d.shape[0]
        nf = xa["n_freq"]
        if "_xa_tok" not in h:                               # once per encoded state (shared by every candidate and sample)
            t8 = MEM.f8_tokens(h)
            h["_xa_tok"] = [(t8, xa["budget_s"], 0)] if xa["memory"] == "s" else [(t8, xa["budget_f8"], 0), (MEM.f16_tokens(h), xa["budget_f16"], 1)]
        Gt = MEM.global_tokens(h["_xa_tok"], [(o[i], d[i], tcap[i]) for i in range(V)], xa["fov_margin_deg"], nf)
        if xa["query"] == "2d":
            Q, aux = QR.spatial_queries(o, d, tcap, h["ext"], nf)
        elif xa["query"] == "ray":
            Q, aux = QR.ray_queries(o, d, tcap, ch, rf, h["ext"], nf)
        else:
            Q, aux = QR.interval_queries(h["x"], h["valid4"], o, d, tcap, ch, rf, h["ext"], nf, xa["max_intervals"])
            if self.xa_stats is not None:
                self.xa_stats.append({k: v.detach().cpu() for k, v in aux["stats"].items()})
        L = QR.local_tokens(h, aux, xa["n_local"], N_FREQ_S) if xa["memory"] == "m" else torch.zeros(V, 1, 1, 1, device=d.device)
        return Gt, Q, L

    def view_cond(self, h, o, d, tcap, ch, mem=None):
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
        return cond, self.context(h, o, d, tcap, ch, rf.reshape(V, R, -1))


def build(cfg):
    return XAModel(cfg) if "xa" in cfg else V3.V3Model(cfg)


def load_core(ckpt_path, device, steps=10, expect_sha=None):
    """EMA checkpoint of the trainer (XA or V3) -> (model, V3Predictor, sha256): seen2scene.v3.predictor.load_core with build()."""
    import hashlib

    from seen2scene.v3.predictor import V3Predictor
    sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    if expect_sha is not None and sha != expect_sha:
        raise RuntimeError(f"checkpoint sha {sha} != expected {expect_sha}")
    ck = torch.load(ckpt_path, map_location=device)
    model = build(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, V3Predictor(model, device, steps), sha
