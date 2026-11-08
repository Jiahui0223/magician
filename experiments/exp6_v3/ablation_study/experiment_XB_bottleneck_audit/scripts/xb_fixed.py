"""Experiment XB / Task A -- fixed-state evaluation of V3-T1, XA-SR and XA-MI with ATTENTION INTERVENTIONS, through the unchanged CT evaluator
(experiment_CT_continued_training/scripts/ct_fixed.py: same replayed states, candidates, K = 4 noise, truth, per-camera metrics, planner scores).
A "checkpoint name" here is BASE@INTERVENTION, e.g. XA_MI@zero_wo. Each is a separate core: the base EMA model deep-copied, with the attention block's
forward replaced by a version that applies the intervention to its inputs and records residual / attention statistics (A1):
  none         unchanged forward (regression: must reproduce the model)
  zero_wo      the attention residual is dropped (h' = h)
  shuffle_tok  scene-token FEATURES permuted among the valid tokens of a view (positional encodings stay in place); M: local sample features permuted along the ray
  shuffle_pe   scene-token POSITIONAL ENCODINGS permuted among the valid tokens (features stay); M: local sample geometry columns permuted along the ray
  drop_band    tokens of cells with observed BAND masked out;  drop_free  tokens of EMPTY-only cells masked out
  drop_f16 / drop_f8 / drop_local   (M memory only) one scale masked out
Outputs work_dirs/.../experiment_XB_bottleneck_audit/states/<group>/<tag>.jsonl (CT row format, ckpt = BASE@IV) and <tag>.a1.json (A1 statistics per core).
Usage: python xb_fixed.py --set indoor|explore|confirm --group G --ckpts NAME,NAME [--only TAG] [--tmax T]"""
import copy
import math
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
for p in ("experiments/exp6_v3/ablation_study/experiment_CT_continued_training/scripts", "experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/scripts",
          "experiments/exp6_v3/shared"):
    sys.path.insert(0, os.path.join(R, p))
import ct_fixed as CT  # noqa: E402

torch, json, time = CT.torch, CT.json, CT.time
from seen2scene.v3 import predictor as V3P  # noqa: E402
from seen2scene.v3.cross_attention import CrossAttnBlock  # noqa: E402
from seen2scene.v3.xa import xa_model as XM  # noqa: E402
from seen2scene.v3.xa.attention import XABlock  # noqa: E402

XBW = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_XB_bottleneck_audit")
CT.W = XBW
BASES = {"T1": ("work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt", "639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb"),
         "XA_SR": ("work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/checkpoints/xa_sr/final.pt", "60069dc4c1419c32231f84f152bc012b72b547d9086cfba86fa667daed3f5ce7"),
         "XA_MI": ("work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/checkpoints/xa_mi/final.pt", "71825e2f02f9b6d1137ff5d15b9ab45b541dddfb186b385e7b30c893c362cc42")}
IVS = ("none", "zero_wo", "shuffle_tok", "shuffle_pe", "drop_band", "drop_free", "drop_f16", "drop_f8", "drop_local")
CT.CFG["checkpoints"] = {f"{b}@{iv}": {"path": BASES[b][0], "sha256": BASES[b][1]} for b in BASES for iv in IVS}
_cur = {"name": None}


def ckpt_path(name):
    _cur["name"] = name
    return os.path.join(R, BASES[name.split("@")[0]][0])


CT.ckpt_path = ckpt_path
_base_cache = {}


def _perm(n, dev, salt):
    g = torch.Generator(device="cpu")
    g.manual_seed(1000003 * salt + n)
    return torch.randperm(n, generator=g).to(dev)


def _acc(self, key, val):
    a = self._xb_acc
    a[key] = a.get(key, 0.0) + float(val)


def _record(self, hmap, res, w, valid_mask, parts):
    """hmap [B, C, ...], res = W_o output [B, C, ...] (same layout), w attention weights [B, H, Nq, Nk] (float), valid_mask [B, Nk] bool, parts: dict name -> [B, Nk] bool."""
    with torch.no_grad():
        rin = hmap.float().pow(2).mean().sqrt()
        rres = res.float().pow(2).mean().sqrt()
        _acc(self, "n_calls", 1); _acc(self, "rms_in", rin); _acc(self, "rms_res", rres); _acc(self, "ratio", rres / rin.clamp(min=1e-12))
        _acc(self, "var_res", res.float().var()); _acc(self, "n_queries", w.shape[2]); _acc(self, "n_keys_valid", valid_mask.float().sum(1).mean())
        wm = w * valid_mask[:, None, None, :].float()
        ent = -(wm * (wm.clamp(min=1e-12)).log()).sum(-1)                        # [B, H, Nq]
        _acc(self, "entropy", ent.mean()); _acc(self, "max_w", wm.max(-1).values.mean())
        _acc(self, "entropy_uniform", valid_mask.float().sum(1).clamp(min=1).log().mean())
        for k, m in parts.items():
            _acc(self, f"mass_{k}", (wm * m[:, None, None, :].float()).sum(-1).mean())


def _xa_forward(self, hmap, G, Q, L):
    iv = getattr(self, "_xb_iv", "none")
    if iv == "zero_wo":
        _acc(self, "n_calls", 1)
        return hmap
    B, C, gh, gw = hmap.shape
    dt = hmap.dtype
    hd = C // self.h
    G, L = G.clone(), L.clone() if self.e_l is not None else L
    nf_cols = G.shape[-1] - 2                                                   # feat + PE
    feat_n = 20                                                                 # [mu 16, band, free, obs, vfrac]
    gmask0 = G[..., -1] > 0.5
    if iv in ("shuffle_tok", "shuffle_pe"):
        for b in range(B):
            idx = torch.nonzero(gmask0[b])[:, 0]
            if idx.numel() > 1:
                p = idx[_perm(idx.numel(), G.device, b)]
                if iv == "shuffle_tok":
                    G[b, idx, :feat_n] = G[b, p, :feat_n]
                else:
                    G[b, idx, feat_n:nf_cols] = G[b, p, feat_n:nf_cols]
        if self.e_l is not None:
            Nl = L.shape[2]
            p = _perm(Nl, L.device, 7)
            if iv == "shuffle_tok":
                L[..., :51] = L[:, :, p, :51]
            else:
                L[..., 51:61] = L[:, :, p, 51:61]
    elif iv == "drop_band":
        G[..., -1] = torch.where(G[..., 16] > 0, torch.zeros_like(G[..., -1]), G[..., -1])
    elif iv == "drop_free":
        G[..., -1] = torch.where((G[..., 16] <= 0) & (G[..., 17] > 0), torch.zeros_like(G[..., -1]), G[..., -1])
    elif iv == "drop_f16":
        G[..., -1] = torch.where(G[..., -2] == 1, torch.zeros_like(G[..., -1]), G[..., -1])
    elif iv == "drop_f8":
        G[..., -1] = torch.where(G[..., -2] == 0, torch.zeros_like(G[..., -1]), G[..., -1])
    elif iv == "drop_local" and self.e_l is not None:
        L[..., -1] = 0.0
    # ---- original math (seen2scene/v3/xa/attention.py), with the weights kept for the statistics
    htok = hmap.flatten(2).transpose(1, 2)
    cell = Q[..., 0].round().long()
    qmask = Q[..., 1] > 0.5
    hq = htok if self.per_cell else htok.gather(1, cell[..., None].expand(-1, -1, C))
    q = self.wq(self.ln_q(hq + self.e_q(Q[..., 2:].to(dt))))
    gf, gs, gmask = G[..., :-2].to(dt), G[..., -2], G[..., -1] > 0.5
    e = self.e_g[0](gf)
    for s in range(1, len(self.e_g)):
        e = torch.where((gs == s)[..., None], self.e_g[s](gf), e)
    kg, vg = self.wk(self.ln_k(e)), self.wv(e)
    heads = lambda t: t.reshape(B, -1, self.h, hd).transpose(1, 2)  # noqa: E731
    qh, kgh, vgh = heads(q), heads(kg), heads(vg)
    Nq, Ng = qh.shape[2], kgh.shape[2]
    logit = (qh.float() @ kgh.float().transpose(-1, -2)) / math.sqrt(hd)
    am = gmask[:, None, None, :].expand(-1, -1, Nq, -1)
    if self.e_l is not None:
        el = self.e_l(L[..., :-1].to(dt))
        Nl = el.shape[2]
        kl = self.wk(self.ln_k(el)).reshape(B, Nq, Nl, self.h, hd).permute(0, 3, 1, 2, 4)
        vl = self.wv(el).reshape(B, Nq, Nl, self.h, hd).permute(0, 3, 1, 2, 4)
        logit = torch.cat([logit, torch.einsum("bhqd,bhqld->bhql", qh.float(), kl.float()) / math.sqrt(hd)], -1)
        am = torch.cat([am, (L[..., -1] > 0.5)[:, None]], -1)
    none = ~am.any(-1, keepdim=True)
    w = torch.softmax(logit.masked_fill(~(am | none), float("-inf")), -1)
    out = w[..., :Ng].to(vgh.dtype) @ vgh
    if self.e_l is not None:
        out = out + torch.einsum("bhql,bhqld->bhqd", w[..., Ng:].to(vl.dtype), vl)
    out = out.masked_fill(none, 0.0).transpose(1, 2).reshape(B, Nq, C)
    qm = qmask[..., None].to(out.dtype)
    if self.per_cell:
        agg = out * qm
    else:
        A = torch.nn.functional.one_hot(cell, gh * gw).to(out.dtype) * qm
        agg = (A.transpose(1, 2) @ out) / A.sum(1).clamp(min=1.0)[..., None]
    res = self.wo(agg).transpose(1, 2).reshape(B, C, gh, gw)
    # statistics: global tokens only for the geometry classes; local mass as its own part
    with torch.no_grad():
        wg = w[..., :Ng]
        parts = {"band": gmask & (G[..., 16] > 0), "free_only": gmask & (G[..., 16] <= 0) & (G[..., 17] > 0), "f16": gmask & (G[..., -2] == 1), "f8": gmask & (G[..., -2] == 0)}
        _record(self, hmap, res, wg, gmask, parts)
        if self.e_l is not None:
            _acc(self, "mass_local", (w[..., Ng:] * (L[..., -1] > 0.5)[:, None].float()).sum(-1).mean())
    return hmap + res


def _t1_forward(self, hmap, mem_tok, mem_mask, qpe):
    iv = getattr(self, "_xb_iv", "none")
    if iv == "zero_wo":
        _acc(self, "n_calls", 1)
        return hmap
    B, C, gh, gw = hmap.shape
    mem_tok, mem_mask = mem_tok.clone(), mem_mask.clone()
    if iv in ("shuffle_tok", "shuffle_pe"):
        for b in range(B):
            idx = torch.nonzero(mem_mask[b])[:, 0]
            if idx.numel() > 1:
                p = idx[_perm(idx.numel(), mem_tok.device, b)]
                if iv == "shuffle_tok":
                    mem_tok[b, idx, :18] = mem_tok[b, p, :18]
                else:
                    mem_tok[b, idx, 18:] = mem_tok[b, p, 18:]
    elif iv == "drop_band":
        mem_mask = mem_mask & ~(mem_tok[..., 16] > 0)
    elif iv == "drop_free":
        mem_mask = mem_mask & ~((mem_tok[..., 16] <= 0) & (mem_tok[..., 17] > 0))
    htok = hmap.flatten(2).transpose(1, 2)
    q = self.wq(self.ln_q(htok + self.q_pe(qpe.to(htok.dtype))))
    kin = self.ln_k(mem_tok.to(htok.dtype))
    k, v = self.wk(kin), self.wv(mem_tok.to(htok.dtype))
    hd = C // self.h
    q = q.reshape(B, -1, self.h, hd).transpose(1, 2)
    k = k.reshape(B, -1, self.h, hd).transpose(1, 2)
    v = v.reshape(B, -1, self.h, hd).transpose(1, 2)
    empty = ~mem_mask.any(1)
    am = mem_mask | empty[:, None]
    logit = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(hd)
    w = torch.softmax(logit.masked_fill(~am[:, None, None, :], float("-inf")), -1)
    out = (w.to(v.dtype) @ v).transpose(1, 2).reshape(B, -1, C)
    out = torch.where(empty[:, None, None], torch.zeros_like(out), out)
    res = self.wo(out).transpose(1, 2).reshape(B, C, gh, gw)
    with torch.no_grad():
        _record(self, hmap, res, w, mem_mask, {"band": mem_mask & (mem_tok[..., 16] > 0), "free_only": mem_mask & (mem_tok[..., 16] <= 0) & (mem_tok[..., 17] > 0)})
    return hmap + res


XABlock.forward = _xa_forward
CrossAttnBlock.forward = _t1_forward


class IVCore:
    """Core for the CT evaluator: BASE@IV -> deep copy of the base EMA model with the intervention set on its attention block."""

    def __init__(self, path, device, expect_sha=None, steps=10):
        name = _cur["name"]
        base = name.split("@")[0]
        if base not in _base_cache:
            _base_cache[base] = XM.load_core(path, device, steps, expect_sha)
        model0, _, sha = _base_cache[base]
        self.model = copy.deepcopy(model0).eval()
        blk = self.model.unet.xattn
        blk._xb_iv = name.split("@")[1]
        blk._xb_acc = {}
        self.pred = V3P.V3Predictor(self.model, device, steps)
        self.ckpt_sha, self.dev, self.ckpt, self.name = sha, device, path, name


CT.FX.PV3.V3Core = IVCore


def _dump_stats(tag, cores, od):
    out = {}
    for n, c in cores.items():
        a = dict(c.model.unet.xattn._xb_acc)
        k = max(1.0, a.get("n_calls", 0.0))
        out[n] = {kk: (vv / k if kk != "n_calls" else vv) for kk, vv in a.items()}
        c.model.unet.xattn._xb_acc = {}
    json.dump(out, open(os.path.join(od, f"{tag}.a1.json"), "w"), indent=1)


_ro, _ri = CT.run_outdoor, CT.run_indoor


def run_outdoor(a, tag, kind, spec, cores, dev):
    rows = _ro(a, tag, kind, spec, cores, dev)
    _dump_stats(tag, cores, a.out or os.path.join(CT.W, "states", a.group))
    return rows


def run_indoor(a, tag, sts, cores, dev):
    rows = _ri(a, tag, sts, cores, dev)
    _dump_stats(tag, cores, a.out or os.path.join(CT.W, "states", a.group))
    return rows


CT.run_outdoor, CT.run_indoor = run_outdoor, run_indoor

if __name__ == "__main__":
    CT.main()
