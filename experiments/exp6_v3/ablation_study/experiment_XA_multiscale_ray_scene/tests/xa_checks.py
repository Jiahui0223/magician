"""Experiment XA -- engineering checks of the six variants on real training states (seen2scene env, 1 GPU). No training result.
  init      common-module hash == the V3-T1 run manifest (same seed -> same initial values of every shared module)
  T0 equal  with the zero-initialised output projection, cond and velocity equal the T0 network (xattn off) bitwise
  shapes    context tensors, masks, token / query / interval counts
  grads     with W_o and the (zero-initialised) flow output conv perturbed: loss.backward is finite and every new parameter
            and the VAE encoder receive a gradient
  sampling  V3Predictor.sample + decode invariants (predicted hits only on UNKNOWN rays); two calls are bitwise equal
  cost      encode + view_cond + one velocity call: seconds, peak memory; parameters; attention token pairs
Usage: python xa_checks.py [--out FILE]"""
import argparse
import json
import os
import sys
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, R)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.v3 import data as D  # noqa: E402
from seen2scene.v3 import losses as L  # noqa: E402
from seen2scene.v3 import model as V3  # noqa: E402
from seen2scene.v3 import predictor as V3P  # noqa: E402
from seen2scene.v3.xa import xa_model as XM  # noqa: E402

X = os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_XA_multiscale_ray_scene")
IDS = ("xa_s2d", "xa_sr", "xa_si", "xa_m2d", "xa_mr", "xa_mi")
T1_COMMON = "6a9d6071a22564c362177bcd2fac0ea8652fe83f3e60de655f299eed0363b513"


def state(domain, dev, t=20, scene=None):
    it = sorted(x for x in D.items(domain, "train") if scene is None or x[0] == scene)[:1]
    data = D.V3Views(domain, it, log=lambda *a: None)
    rows = data.state_rows[(0, t)][:8]
    return data.state_tensors(0, t, 0, rows, dev), it[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(X, "results", "engineering_checks.json"))
    a = ap.parse_args()
    dev = torch.device("cuda")
    states = {"outdoor": state("outdoor", dev, scene="pisa"), "indoor": state("indoor", dev)}
    out = {"states": {k: list(v[1]) for k, v in states.items()}, "variants": {}}
    cfg0 = json.load(open(os.path.join(R, "experiments", "exp6_v3", "configs", "model_v3.json")))
    torch.manual_seed(0)
    t0m = V3.V3Model(cfg0).to(dev).eval()                  # T0 (xattn disabled in the file)
    ref = {}
    for dom, (st, _) in states.items():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = t0m.encode(st["x4"], st["valid4"])
            cond = t0m.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"])
            g = torch.Generator(device=dev).manual_seed(1)
            V = cond.shape[0]
            x = torch.randn(V, 2, 64, 128, device=dev, generator=g) * st["m_unk"][:, None]
            tau = torch.rand(V, device=dev, generator=g)
            ref[dom] = (cond.float().clone(), t0m.velocity(x, tau, cond).float().clone(), x, tau)
    for vid in IDS:
        mc = json.load(open(os.path.join(X, "configs", f"model_{vid}.json")))
        torch.manual_seed(0)
        model = XM.build(mc).to(dev)
        r = {"params": V3.param_counts(model), "common_init_sha256": V3.state_hash(V3.common_state(model))}
        r["common_init_equals_T1"] = r["common_init_sha256"] == T1_COMMON
        r["xattn_params"] = {n: int(p.numel()) for n, p in model.unet.xattn.named_parameters()}
        model.xa_stats = []
        for dom, (st, _) in states.items():
            model.eval()
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(); tb = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.encode(st["x4"], st["valid4"])
                cond, mem = model.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"])
                torch.cuda.synchronize(); t1 = time.time()
                v = model.velocity(ref[dom][2], ref[dom][3], cond, mem).float()
                torch.cuda.synchronize(); t2 = time.time()
            Gt, Q, Lt = mem
            V = cond.shape[0]
            d = {"views": V, "G": list(Gt.shape), "Q": list(Q.shape), "L": list(Lt.shape), "dtypes": [str(x.dtype) for x in mem],
                 "global_tokens_per_view": Gt[..., -1].sum(1).tolist(), "queries_per_view": Q[..., 1].sum(1).tolist(),
                 "global_by_scale": [float(((Gt[..., -2] == s) & (Gt[..., -1] > 0)).sum(1).float().mean()) for s in (0, 1)],
                 "local_valid_share": float(Lt[..., -1].mean()) if Lt.shape[-1] > 1 else None,
                 "finite": bool(torch.isfinite(Gt).all() and torch.isfinite(Q).all() and torch.isfinite(Lt).all() and torch.isfinite(v).all()),
                 "cond_equals_T0": bool(torch.equal(cond.float(), ref[dom][0])), "velocity_equals_T0_at_init": bool(torch.equal(v, ref[dom][1])),
                 "encode_cond_s": t1 - tb, "velocity_s": t2 - t1, "peak_gb_eval": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                 "attention_pairs_per_view": int(Q.shape[1] * (Gt.shape[1] + (Lt.shape[2] if Lt.shape[-1] > 1 else 0)))}
            if model.xa["query"] == "interval":
                s = model.xa_stats[-1]
                d["interval"] = {k: v_.float().mean().item() for k, v_ in s.items()}
                d["interval"]["ray_coverage"] = float((s["rays_covered"].float() / s["rays_with_interval"].float().clamp(min=1)).mean())
                d["interval"]["selected_share"] = float((s["selected"].float() / s["intervals"].float().clamp(min=1)).mean())
                cell = Q[..., 0].long()
                d["interval"]["max_queries_per_cell"] = int(torch.stack([torch.bincount(cell[i][Q[i, :, 1] > 0.5], minlength=512).max() for i in range(V)]).max())
            # determinism of the context and of the velocity
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h2 = model.encode(st["x4"], st["valid4"])
                cond2, mem2 = model.view_cond(h2, st["o"], st["d"], st["tcap"], st["ch"])
                v2 = model.velocity(ref[dom][2], ref[dom][3], cond2, mem2).float()
            d["deterministic"] = bool(torch.equal(v, v2) and all(torch.equal(x_, y_) for x_, y_ in zip(mem, mem2)))
            # gradients (train mode, real loss terms)
            model.train()
            torch.manual_seed(3)
            with torch.no_grad():
                w0, o0 = model.unet.xattn.wo.weight.clone(), model.unet.out[-1].weight.clone()
                model.unet.xattn.wo.weight.normal_(0, 0.02)
                model.unet.out[-1].weight.normal_(0, 0.02)       # the flow output conv is zero-initialised too (V2 / V3): without this only it gets a gradient
            torch.cuda.reset_peak_memory_stats(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.encode(st["x4"], st["valid4"])
                cond, mem = model.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"])
            bt = {"m_unk": st["m_unk"], "x1": st["x1"], "lmask": st["lmask"], "cond": cond, "mem": mem}
            eps = torch.randn(V, 2, 64, 128, device=dev)
            tau = torch.rand(V, device=dev)
            es, nr = L.fm_terms(model, bt, eps, tau)
            stt = [{"mu": h["mu"], "logvar": h["logvar"], "valid8": h["valid8"], "x4": st["x4"], "valid4": st["valid4"]}]
            rec, _ = L.recon_terms(model, stt)
            kl, _ = L.kl_terms(stt)
            loss = es / (2 * nr.clamp(min=1)) + rec + 0.02 * kl
            model.zero_grad(set_to_none=True)
            loss.backward()
            gx = {n: float(p.grad.float().norm()) if p.grad is not None else None for n, p in model.unet.xattn.named_parameters()}
            d["loss"] = float(loss)
            d["grad_xattn_all_nonzero_finite"] = all(v_ is not None and np.isfinite(v_) and v_ > 0 for v_ in gx.values())
            d["grad_xattn_zero"] = [n for n, v_ in gx.items() if not (v_ is not None and v_ > 0)]
            d["grad_vae_enc"] = float(sum(p.grad.float().pow(2).sum() for p in model.vae_enc.parameters() if p.grad is not None) ** 0.5)
            d["grad_all_finite"] = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            d["peak_gb_train_1state"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
            # sampling + decode with the perturbed block (the block is active)
            model.eval()
            pred = V3P.V3Predictor(model, dev, 10)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.encode(st["x4"], st["valid4"])
                cond, mem = model.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"])
            g = torch.Generator(device=dev).manual_seed(5)
            e4 = torch.stack([torch.randn(2, 64, 128, device=dev, generator=g) for _ in range(4)])
            torch.cuda.synchronize(); tb = time.time()
            b, y = pred.sample((cond, mem), st["m_unk"], e4)
            torch.cuda.synchronize()
            d["sample_s_K4_E10"] = time.time() - tb
            b2, y2 = pred.sample((cond, mem), st["m_unk"], e4)
            d["sample_deterministic"] = bool(torch.equal(b, b2) and torch.equal(y, y2))
            mu = M.unpad_rays(st["m_unk"])
            d["sample_shape"] = list(b.shape)
            d["sample_zero_outside_unknown"] = bool((b[:, :, :][~mu[:, None].expand_as(b)] == 0).all())
            d["sample_finite"] = bool(torch.isfinite(b).all() and torch.isfinite(y).all())
            with torch.no_grad():
                model.unet.xattn.wo.weight.copy_(w0)
                model.unet.out[-1].weight.copy_(o0)
            r[dom] = d
            del h, cond, mem, bt, loss
            torch.cuda.empty_cache()
        out["variants"][vid] = r
        ok = all(r[dom][k] for dom in states for k in ("finite", "cond_equals_T0", "velocity_equals_T0_at_init", "deterministic", "grad_xattn_all_nonzero_finite", "grad_all_finite", "sample_deterministic", "sample_zero_outside_unknown", "sample_finite")) and r["common_init_equals_T1"]
        r["PASS"] = bool(ok)
        print(vid, "PASS" if ok else "FAIL", "params", r["params"]["total"], "xattn", r["params"]["xattn"], "| outdoor", {k: r["outdoor"][k] for k in ("G", "Q", "L", "attention_pairs_per_view", "encode_cond_s", "velocity_s", "sample_s_K4_E10", "peak_gb_train_1state", "grad_xattn_zero")},
              "| indoor", {k: r["indoor"][k] for k in ("encode_cond_s", "sample_s_K4_E10", "peak_gb_train_1state", "local_valid_share", "global_by_scale")}, r["outdoor"].get("interval"), flush=True)
        if not ok:
            print({dom: {k: r[dom][k] for k in ("finite", "cond_equals_T0", "velocity_equals_T0_at_init", "deterministic", "grad_xattn_all_nonzero_finite", "grad_all_finite", "sample_deterministic", "sample_zero_outside_unknown", "sample_finite")} for dom in states}, r["common_init_equals_T1"])
        del model
        torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
