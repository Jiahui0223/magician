"""Experiment G -- D0 / G0 unit tests on real V3 training views (protocol G-v1 section 6 (b); seen2scene env, 1 GPU).
Writes a JSON report (default work_dirs/exp6_v3/experiment_G/checks/unit_tests.json); exit code 1 if any test fails.
Usage: python test_d0_unit.py [--out PATH]"""
import argparse
import copy
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d0_head as DH  # noqa: E402
from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.v3 import data as D  # noqa: E402
from seen2scene.v3 import losses as L  # noqa: E402
from seen2scene.v3 import model as V3  # noqa: E402
from seen2scene.v3 import predictor as V3P  # noqa: E402
from seen2scene.v3 import trainer as TR  # noqa: E402

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
T0 = os.path.join(R, "work_dirs/exp6_v3/task0_vae/checkpoints/final.pt")
T0_SHA = "7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6"
RES = {}


def check(name, ok, **info):
    RES[name] = {"pass": bool(ok), **{k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in info.items()}}
    print(("PASS " if ok else "FAIL ") + name, json.dumps(RES[name], default=str)[:300], flush=True)


def grad_norms(model):
    out = {}
    for k in ("vae_enc", "vae_dec", "rays", "unet"):
        s = sum(float((p.grad.float() ** 2).sum()) for p in getattr(model, k).parameters() if p.grad is not None)
        out[k] = s ** 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "work_dirs/exp6_v3/experiment_G/checks/unit_tests.json"))
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    t_start = time.time()
    cfg = json.load(open(os.path.join(TR.CFG_DIR, "train_v3.json")))
    mcfg = TR.model_cfg("t0", cfg)
    # real views: first S-c outdoor trajectory + first indoor valsel trajectory (fixed rows, no randomness)
    data_o = D.V3Views("outdoor", D.items("outdoor", "sc")[:1], print)
    data_i = D.V3Views("indoor", D.items("indoor", "valsel")[:1], print)
    bo = data_o.fixed_states("expG|unit|out", 6)
    bi = data_i.fixed_states("expG|unit|in", 6)

    # ---------------------------------------------------------------- model: same init as V3-T0 (seed 0) + head key ignored by V3Model
    m_flow = TR.make_model(mcfg, cfg["seed"], dev)
    mcfg_det = dict(copy.deepcopy(mcfg), **{DH.HEAD_KEY: DH.DET})
    m_det = TR.make_model(mcfg_det, cfg["seed"], dev)
    h_flow, h_det = V3.state_hash(V3.common_state(m_flow)), V3.state_hash(V3.common_state(m_det))
    check("init_identical_to_T0_rule", h_flow == h_det == "6a9d6071a22564c362177bcd2fac0ea8652fe83f3e60de655f299eed0363b513",
          init_hash=h_det, params=V3.param_counts(m_det))
    # realistic weights for the gradient / output tests (zero-initialised output conv blocks upstream gradients at init)
    sd = torch.load(T0, map_location=dev)["ema"]
    m_det.load_state_dict(sd)
    m_det.train()
    bt = D.build_mixed(m_det, [(data_i, bi), (data_o, bo)], dev)
    V = bt["cond"].shape[0]

    # ---------------------------------------------------------------- shapes + masking of the output
    with torch.no_grad():
        raw = DH.det_forward(m_det, bt["cond"])
        xs = DH.det_sample(m_det, bt["cond"], bt["m_unk"], torch.randn(V, 2, M.GH, M.GW_PAD, device=dev))
    check("shapes", tuple(raw.shape) == (V, 2, M.GH, M.GW_PAD) and tuple(xs.shape) == (V, 2, M.GH, M.GW_PAD) and xs.dtype == torch.float32,
          raw=list(raw.shape), sample=list(xs.shape), cond=list(bt["cond"].shape), unet_in_ch=m_det.unet.inp.in_channels)
    outside = ~bt["m_unk"][:, None].expand_as(xs)
    check("sample_zero_outside_m_unk", bool((xs[outside] == 0).all()) and bool((xs[~outside] != 0).any()))
    check("sample_ignores_eps", torch.equal(xs, DH.det_sample(m_det, bt["cond"], bt["m_unk"], torch.zeros(V, 2, M.GH, M.GW_PAD, device=dev))))

    # ---------------------------------------------------------------- loss: hand computation + mask invariance
    m_det.zero_grad(set_to_none=True)
    esum, nray = DH.det_terms(m_det, bt, None, None)
    with torch.no_grad():
        x1m = bt["x1"] * bt["m_unk"][:, None].float()
        lm = bt["lmask"]
        man = float(((raw.float() - x1m) ** 2).sum(1)[lm].sum())
    check("loss_matches_hand", abs(float(esum) - man) <= 1e-6 * max(1.0, man) and int(nray) == int(lm.sum()),
          esum=float(esum), hand=man, nray=float(nray), lmask=int(lm.sum()), fm_mean=float(esum) / (2 * float(nray)))
    bt2 = dict(bt)
    x1p = bt["x1"].clone()
    x1p[~lm[:, None].expand_as(x1p)] += 7.0                     # change targets OUTSIDE lmask only
    bt2["x1"] = x1p
    with torch.no_grad():
        e2, _ = DH.det_terms(m_det, bt2, None, None)
    check("loss_ignores_targets_outside_lmask", float(e2) == float(esum), changed=float(e2), base=float(esum))
    with torch.no_grad():
        pv_e, pv_c = DH.det_per_view_loss(m_det, bt, None, None)
    check("per_view_loss_consistent", abs(float(pv_e.sum()) - float(esum)) <= 1e-4 * max(1.0, float(esum)) and float(pv_c.sum()) == 2 * float(nray))

    # ---------------------------------------------------------------- gradients reach VAE encoder, ray encoder and U-Net (prediction loss alone)
    (esum / (2 * nray)).backward()
    gp = grad_norms(m_det)
    gx = float(m_det.unet.inp.weight.grad[:, :2].abs().max())
    gc = float(m_det.unet.inp.weight.grad[:, 2:].abs().max())
    check("grad_det_loss_reaches_encoder_rays_unet", gp["vae_enc"] > 0 and gp["rays"] > 0 and gp["unet"] > 0 and all(map(lambda v: v == v, gp.values())),
          **{f"gn_{k}": v for k, v in gp.items()})
    check("grad_x_channels_zero", gx == 0.0 and gc > 0, max_grad_x_channels=gx, max_grad_cond_channels=gc)
    # full trainer objective (det + recon + KL): the VAE decoder gets its gradient from L_recon as in T0
    m_det.zero_grad(set_to_none=True)
    bt = D.build_mixed(m_det, [(data_i, bi), (data_o, bo)], dev)
    esum, nray = DH.det_terms(m_det, bt, None, None)
    rec, _ = L.recon_terms(m_det, bt["vae"])
    kl, _ = L.kl_terms(bt["vae"])
    loss = esum / (2 * nray) + cfg["lambda_rec"] * rec + 0.02 * kl
    loss.backward()
    gp = grad_norms(m_det)
    check("grad_full_objective_all_modules", all(v > 0 for v in gp.values()) and torch.isfinite(loss).item(), loss=float(loss), **{f"gn_{k}": v for k, v in gp.items()})
    m_det.zero_grad(set_to_none=True)

    # ---------------------------------------------------------------- determinism
    m_det.eval()
    with torch.no_grad():
        c, mu = bt["cond"], bt["m_unk"]
        a1 = DH.det_sample(m_det, c, mu, None)
        a2 = DH.det_sample(m_det, c, mu, None)
        rep = DH.det_sample(m_det, c.repeat(4, 1, 1, 1), mu.repeat(4, 1, 1), None).reshape(4, V, 2, M.GH, M.GW_PAD)
        one = DH.det_sample(m_det, c[3:4], mu[3:4], None)
    check("deterministic_repeat_bitwise", torch.equal(a1, a2))
    check("K_copies_identical_in_batch", all(torch.equal(rep[0], rep[k]) for k in range(1, 4)),
          note="eval_samples feeds K = 4 copies of every view in ONE batch: they must be bitwise identical (spread / pair term exactly 0)",
          max_abs_diff_vs_batch_of_V=float((rep[0] - a1).abs().max()))
    check("single_view_vs_batch", True, max_abs_diff=float((one[0] - a1[3]).abs().max()), note="informational: bf16 kernels depend on the batch size (same for the flow model); within one batch rows are position-independent")

    # ---------------------------------------------------------------- planner predictors: DetPredictor == det_sample rows, G0 == V3Predictor(eps = 0)
    pred_d = DH.DetPredictor(m_det, dev)
    eps1 = torch.randn(1, 2, M.GH, M.GW_PAD, device=dev)
    b, y = pred_d.sample((c, None), mu, eps1)
    ref = M.unpad_rays(a1)
    check("DetPredictor_matches_det_sample", tuple(b.shape) == (V, 1, M.GH * M.GW) and torch.equal(b[:, 0], ref[:, 0]) and torch.equal(y[:, 0], ref[:, 1]),
          shape=list(b.shape))
    b4, y4 = pred_d.sample((c, None), mu, torch.randn(4, 2, M.GH, M.GW_PAD, device=dev))
    check("DetPredictor_K4_identical_samples", all(torch.equal(b4[:, 0], b4[:, k]) and torch.equal(y4[:, 0], y4[:, k]) for k in range(1, 4)))
    m_t0 = V3.V3Model(mcfg).to(dev)
    m_t0.load_state_dict(sd)
    m_t0.eval()
    pz, pf = DH.ZeroNoisePredictor(m_t0, dev), V3P.V3Predictor(m_t0, dev)
    bz, yz = pz.sample((c, None), mu, eps1)
    bf, yf = pf.sample((c, None), mu, torch.zeros_like(eps1))
    bn, yn = pf.sample((c, None), mu, eps1)
    check("G0_equals_flow_with_zero_noise", torch.equal(bz, bf) and torch.equal(yz, yf) and not torch.equal(bz, bn))

    # ---------------------------------------------------------------- decode rule on D0 outputs (geometry.decode hit rule)
    hit = (b[:, 0] > 0) & (y[:, 0] < 1.0)
    lm_r = M.unpad_rays(bt["lmask"])
    cat = bt["cat"].reshape(V, -1)
    lab = (cat == G.CAT["HIT"]) & lm_r
    nohit = (cat == G.CAT["NOHIT"]) & lm_r
    acc = float(((hit == lab) & lm_r).sum()) / max(1.0, float(lm_r.sum()))
    check("decode_rule_d0_outputs_sane", 0.0 <= acc <= 1.0, hit_rate_on_labelled=float(hit[lm_r].float().mean()), label_hit_rate=float(lab[lm_r].float().mean()),
          false_wall_on_nohit=float(hit[nohit].float().mean()) if nohit.any() else None, accuracy_T0_weights_det_head=acc,
          note="informational: D0 head run on V3-T0 weights (not a trained D0); decode itself = geometry.decode (planner) / eval_samples rule (training eval)")

    # ---------------------------------------------------------------- loader guards (mode vs checkpoint head)
    try:
        DH.load(T0, dev, "d0")
        guard = False
    except RuntimeError:
        guard = True
    _, pr_flow, sha = DH.load(T0, dev, "flow", expect_sha=T0_SHA)
    check("loader_rejects_d0_mode_on_flow_ckpt", guard and type(pr_flow) is V3P.V3Predictor and sha == T0_SHA)

    out = {"tests": RES, "all_pass": all(r["pass"] for r in RES.values()), "n": len(RES), "views": V, "seconds": time.time() - t_start,
           "gpu": torch.cuda.get_device_name(0), "host": os.uname()[1], "time": time.ctime(), "code_sha256": DH.code_hashes()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("ALL PASS" if out["all_pass"] else "SOME FAILED", f"{len(RES)} tests, {out['seconds']:.0f}s")
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
