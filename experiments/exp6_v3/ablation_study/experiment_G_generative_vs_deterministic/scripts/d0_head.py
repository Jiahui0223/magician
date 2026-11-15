"""Experiment G -- the deterministic head D0 and the zero-noise decoder G0 on top of the UNCHANGED V3 code (protocol G-v1,
reports/G0_PROTOCOL.md). Nothing in seen2scene/ is edited: every function here is a drop-in for one V3 function and is installed
only by the Experiment G entry points (train_g.py --head det, the planner child g_out_planner.py --g-mode d0|g0, g_fixed.py).

D0  same V3Model (VAE -> LatentRayEncoder -> FlowUNet, T0 = no cross-attention), same config, same parameters. The U-Net input is
    [x (2) || cond (76)] with x = 0 and tau = 0 for every view; its 2-channel output is read directly as x1_hat = (b_hat, y_hat).
      det_forward   the one U-Net call (the flow's model.velocity(x = 0, tau = 0, cond))
      det_terms     drop-in for losses.fm_terms: sum over labelled UNKNOWN rays of ||x1_hat - x1||^2 (both channels) and the number of
                    such rays -- the fm_terms masks (m_unk on the target, lmask on the error) and the trainer's global 2 x N normalisation
      det_sample    drop-in for v3.model.sample: x1_hat * m_unk (the Euler loop's output masking), one forward, eps / steps ignored
      det_per_view_loss   drop-in for eval_dev.per_view_loss (the eval FM metric of the trainer)
      DetPredictor  V3Predictor whose sample() runs det_sample instead of the Euler integration (same chunking, same output layout)
G0  V3-T0 weights, flow decoding from x0 = 0: ZeroNoisePredictor = V3Predictor with eps replaced by zeros (Euler 10, same code).
"""
import hashlib
import json
import os

import torch

from seen2scene.futureview import model as M
from seen2scene.v3 import model as V3
from seen2scene.v3 import predictor as V3P

HEAD_KEY = "head"           # model_cfg key written into D0 checkpoints ("det"); absent in every V3 checkpoint
DET = "det"
GH, GW_PAD = M.GH, M.GW_PAD


def is_det(model_cfg):
    return model_cfg.get(HEAD_KEY, "flow") == DET


def det_forward(model, cond, mem=None, autocast=True, return_features=False):
    """cond [V, C, 64, 128] -> raw U-Net output [V, 2, 64, 128] for x = 0, tau = 0 (bf16 autocast as in fm_terms / sample)."""
    V = cond.shape[0]
    x = torch.zeros(V, model.x_ch, GH, GW_PAD, device=cond.device)          # float32, like the masked x_tau of fm_terms
    tau = torch.zeros(V, device=cond.device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        return model.velocity(x, tau, cond, mem, return_features=return_features)


def det_terms(model, bt, eps, tau, autocast=True):
    """Drop-in for seen2scene.v3.losses.fm_terms (eps, tau accepted and ignored; the trainer still draws them, so the torch RNG
    stream -- and with it the VAE reparameterisation noise -- is the one of the V3-T0 run)."""
    m = bt["m_unk"][:, None].float()
    x1 = bt["x1"] * m
    xh = det_forward(model, bt["cond"], bt.get("mem"), autocast)
    err = ((xh.float() - x1) ** 2).sum(1)
    lm = bt["lmask"].float()
    return (err * lm).sum(), lm.sum()


def det_per_view_loss(model, bt, eps, tau, autocast=True):
    """Drop-in for seen2scene.v3.eval_dev.per_view_loss (per view: error sum, 2 x labelled rays)."""
    m = bt["m_unk"][:, None].float()
    x1 = bt["x1"] * m
    xh = det_forward(model, bt["cond"], bt.get("mem"), autocast)
    err = ((xh.float() - x1) ** 2).sum(1)
    lm = bt["lmask"].float()
    return (err * lm).sum((1, 2)), 2.0 * lm.sum((1, 2))


@torch.no_grad()
def det_sample(model, cond, m_unk, eps, mem=None, steps=10, autocast=True, return_features=False):
    """Drop-in for seen2scene.v3.model.sample: one forward, output masked to M_unknown rays; eps and steps are ignored."""
    out = det_forward(model, cond, mem, autocast, return_features)
    if return_features:
        out, feats = out
        return out.float() * m_unk[:, None], feats
    return out.float() * m_unk[:, None]


class DetPredictor(V3P.V3Predictor):
    """D0 planner-side predictor: V3Predictor.sample with det_sample in place of the Euler integration (same row order / chunks).
    eps [K, ...] only sets K; with K = 1 (the Table 3 K = 1 path) it is ONE forward per candidate."""

    @torch.no_grad()
    def sample(self, pack, m_unk, eps, return_features=False):
        cond, mem = pack
        V, K = cond.shape[0], eps.shape[0]
        idx = torch.arange(V * K, device=cond.device)
        vi, ki = idx // K, idx % K
        xs, fs = [], []
        for s in range(0, V * K, self.max_views):
            sl = slice(s, min(V * K, s + self.max_views))
            r = det_sample(self.model, cond[vi[sl]], m_unk[vi[sl]], eps[ki[sl]], V3.index_mem(mem, vi[sl]), self.steps, self.autocast, return_features)
            if return_features:
                xs.append(r[0]); fs.append(r[1].float())
            else:
                xs.append(r)
        x = M.unpad_rays(torch.cat(xs)).reshape(V, K, 2, -1)
        if return_features:
            return x[:, :, 0], x[:, :, 1], torch.cat(fs).reshape(V, K, *fs[0].shape[1:])
        return x[:, :, 0], x[:, :, 1]


class ZeroNoisePredictor(V3P.V3Predictor):
    """G0 (diagnostic): the V3-T0 flow decoded from x0 = 0 -- V3Predictor.sample with every noise map replaced by zeros."""

    @torch.no_grad()
    def sample(self, pack, m_unk, eps, return_features=False):
        return super().sample(pack, m_unk, torch.zeros_like(eps), return_features)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(ckpt_path, device, mode, steps=10, expect_sha=None):
    """mode 'flow' (G1 / G4: the unchanged V3Predictor), 'g0' (ZeroNoisePredictor on a flow checkpoint), 'd0' (DetPredictor on a
    D0 checkpoint). The checkpoint's model_cfg must match the mode (head 'det' <-> d0). -> (model, predictor, sha256)."""
    assert mode in ("flow", "g0", "d0"), mode
    sha = sha256_file(ckpt_path)
    if expect_sha is not None and sha != expect_sha:
        raise RuntimeError(f"checkpoint sha {sha} != expected {expect_sha}")
    ck = torch.load(ckpt_path, map_location=device)
    if is_det(ck["model_cfg"]) != (mode == "d0"):
        raise RuntimeError(f"mode {mode} does not match the checkpoint head {ck['model_cfg'].get(HEAD_KEY, 'flow')} ({ckpt_path})")
    model = V3.V3Model(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    cls = {"flow": V3P.V3Predictor, "g0": ZeroNoisePredictor, "d0": DetPredictor}[mode]
    return model, cls(model, device, steps), sha


def code_hashes():
    here = os.path.dirname(os.path.abspath(__file__))
    return {f: sha256_file(os.path.join(here, f)) for f in sorted(os.listdir(here)) if f.endswith((".py", ".sh"))}


if __name__ == "__main__":
    print(json.dumps(code_hashes(), indent=1))
