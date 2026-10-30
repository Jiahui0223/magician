"""EXP5 Task 1.3 view-latent diagnostics (protocol §4.8; diagnostic only, never enters the closed loop).

  extract : (seen2scene env, 1 GPU) final U-Net decoder features (last Euler step, sample k = 0, C = 64) of up to 128
            valid labelled M_unknown rays per view (seeded draw), with their labels (b, y) and the sample's own x_1:
            S-c calibration states (fit set; labels from the T1.1 shards) and S-a states (report set; labels from the
            evaluation-side Dev t*, protocol §2.4 encode_labels). Noise seeds = those of the S-c / S-a evaluations.
  probe   : (CPU) ridge probe features -> b and features -> y (y on HIT rays only), fitted on S-c, R^2 on S-a (and
            in-sample on S-c); reference: R^2 of the sample's own x_1. Per-state Spearman of U_feat with U_raw, DistU
            (delta frozen) and the true one-step gain on S-c / S-a / S-b.
U_feat is the K-sample variance of these features: it is NOT called geometric uncertainty or information gain.
Output W/fixed_state/latent/{feat_sc.npz, feat_sa.npz, latent.json}
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402

import numpy as np  # noqa: E402

OUT = os.path.join(VF.WORK, "fixed_state", "latent")
N_RAYS = 128


def extract():
    import torch
    import m_common as MC
    import m_p1
    import t5_diagnostic as TD
    import t3_common as C3
    import t12_eval_sc as EV
    import t13_fixed_state as FS
    import v2_plan_core as PC
    from seen2scene.futureview import geometry as G
    from seen2scene.futureview import model as M
    from seen2scene.futureview import predictor as PR
    from seen2scene.futureview import rays as RY
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    core = FS.load_core(dev)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(RY.sha_seed("exp5-V2|latent|rays") % (2 ** 63))

    def run(geo, h, legal_R, legal_T, eps, labels, acc):
        views = [geo.view(R, T) for R, T in zip(legal_R, legal_T)]
        if not views:
            return
        cond, m_unk = core.pred.cond(h, [(v["o"], v["d"], v["tcap"], v["pr"]) for v in views])
        b, y, f = core.pred.sample(cond, m_unk, eps, return_features=True)
        for m, v in enumerate(views):
            lb, ly = labels(m, v)
            ok = (v["pr"]["cls"] == G.CLS_UNKNOWN) & (lb != 0)
            idx = torch.nonzero(ok)[:, 0].cpu().numpy()
            if len(idx) == 0:
                continue
            idx = np.sort(rng.choice(idx, min(N_RAYS, len(idx)), replace=False))
            it = torch.as_tensor(idx, device=dev)
            fm = M.unpad_rays(f[m])[0]                       # [C, R], sample k = 0
            acc["feat"].append(fm[:, it].T.cpu().numpy().astype(np.float32))
            acc["b"].append(lb[it].float().cpu().numpy())
            acc["y"].append(ly[it].float().cpu().numpy())
            acc["xb"].append(b[m][0, it].cpu().numpy())
            acc["xy"].append(y[m][0, it].cpu().numpy())

    # S-c (fit): P1 replay exactly as t12_eval_sc, labels from the shard
    acc = {k: [] for k in ("feat", "b", "y", "xb", "xy")}
    times = EV.SC["Sc_times"]
    for scene, traj in EV.sc_items():
        fr, dims, dom, win = TD.scene_geometry(scene)
        shard = np.load(os.path.join(VF.WORK, "data", "views", scene, f"{traj}.npz"))
        fs = EV.records(scene, traj)
        fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
        for t in range(max(times) + 1):
            rec = MC.load_service_record(fs[t])
            fu.integrate_record(rec)
            if t not in times:
                continue
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            eps = PR.noise(dev, 4, lambda k: EV.seed(f"exp5-V2|sc|{scene}|{traj}|t{t}|k{k}"))
            rows = np.flatnonzero(shard["t"] == t)
            lab = lambda m, v: (torch.as_tensor(shard["b"][rows[m]], device=dev), torch.as_tensor(shard["y"][rows[m]], device=dev).float())  # noqa: E731
            run(geo, h, rec["legal_R"], rec["legal_T"], eps, lab, acc)
            del geo, h
        print("sc", scene, traj, sum(len(x) for x in acc["b"]), flush=True)
    np.savez_compressed(os.path.join(OUT, "feat_sc.npz"), **{k: np.concatenate(v) for k, v in acc.items()})
    # S-a (report): labels from the evaluation-side Dev t*
    import abl_common as AC
    acc = {k: [] for k in ("feat", "b", "y", "xb", "xy")}
    for s, k, t in C3.STATES:
        st = C3.planner_state(s, k, t, dev)
        geo = PC.Geometry(st["frame"], st["dims"], st["dom"], st["vol"], st["structure"], dev)
        h = core.pred.encode(geo.coarse())
        rec = MC.load_service_record(TD.svc_path(s, k, t))
        eps = PR.noise(dev, 4, lambda kk: FS.seed(f"exp5-V2|sa|{s}|s{k}|t{t}|k{kk}"))
        dt = FS.dev_tstar("sa", s, k)
        ts = dt["tstar"][dt["t"] == t]

        def lab(m, v, ts=ts, geo=geo):
            tq = torch.as_tensor(ts[m], device=dev, dtype=v["d"].dtype)
            tq = torch.where(torch.isfinite(tq), tq, torch.full_like(tq, float("inf")))
            e = G.encode_labels(v["o"], v["d"], v["pr"], tq, v["dz"], geo.fr.k_M, geo.struct, geo.domain)
            return e["b"], e["y"].float()
        run(geo, h, rec["legal_R"], rec["legal_T"], eps, lab, acc)
        print("sa", s, k, t, sum(len(x) for x in acc["b"]), flush=True)
        del geo, h, st
    np.savez_compressed(os.path.join(OUT, "feat_sa.npz"), **{k: np.concatenate(v) for k, v in acc.items()})
    _ = AC


def r2(pred, y):
    return float(1.0 - ((pred - y) ** 2).sum() / max(1e-12, ((y - y.mean()) ** 2).sum()))


def ridge(X, y, lam=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    A = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ y)
    return lambda Xn: np.c_[(Xn - mu) / sd, np.ones(len(Xn))] @ w


def spearman(a, b):
    import t13_analyze as TA
    return TA.spearman(a, b)


def probe():
    fit, rep = np.load(os.path.join(OUT, "feat_sc.npz")), np.load(os.path.join(OUT, "feat_sa.npz"))
    res = {"n_rays_fit_Sc": int(len(fit["b"])), "n_rays_report_Sa": int(len(rep["b"])), "C": int(fit["feat"].shape[1])}
    fb = ridge(fit["feat"].astype(np.float64), fit["b"])
    res["R2_b"] = {"Sc_in_sample": r2(fb(fit["feat"]), fit["b"]), "Sa": r2(fb(rep["feat"]), rep["b"]),
                   "ref_own_x1_Sa": r2(rep["xb"], rep["b"])}
    hf, hr = fit["b"] > 0, rep["b"] > 0
    fy = ridge(fit["feat"][hf].astype(np.float64), fit["y"][hf])
    res["R2_y_on_HIT"] = {"Sc_in_sample": r2(fy(fit["feat"][hf]), fit["y"][hf]), "Sa": r2(fy(rep["feat"][hr]), rep["y"][hr]),
                          "ref_own_x1_Sa": r2(rep["xy"][hr], rep["y"][hr])}
    dl = VF.frozen_values()["scoring"]["delta_vox"]
    corr = {}
    sets = {"S-c": glob.glob(os.path.join(VF.WORK, "eval_sc", "main", "states_*.jsonl")),
            "S-a": glob.glob(os.path.join(VF.WORK, "fixed_state", "sa", "part*.jsonl")),
            "S-b": glob.glob(os.path.join(VF.WORK, "fixed_state", "sb", "sb_*.jsonl"))}
    gains = gain_lookup()
    for nm, ps in sets.items():
        rows = [json.loads(x) for p in sorted(ps) for x in open(p)]
        out = {"u_raw": [], "dist_u": [], "true_gain": []}
        for r in rows:
            uf = r.get("u_feat")
            if not uf or len(uf) < 2:
                continue
            for key, other in (("u_raw", r.get(f"u_raw@{dl}")), ("dist_u", r.get(f"dist_u@{dl}")),
                               ("true_gain", gains(nm, r))):
                if other is not None and len(other) == len(uf):
                    c = spearman(uf, other)
                    if c is not None:
                        out[key].append(c)
        corr[nm] = {k: {"mean": float(np.mean(v)) if v else None, "states": len(v)} for k, v in out.items()}
    res["spearman_Ufeat"] = corr
    res["delta_vox"] = dl
    res["note"] = "U_feat = variance of decoder features over the K samples; diagnostic only, not uncertainty / information gain"
    json.dump(res, open(os.path.join(OUT, "latent.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def gain_lookup():
    """True one-step gains per state: S-c from S-GAIN-C; S-a / S-b via t13_analyze's loaders."""
    sc = {}
    for p in glob.glob(os.path.join(VF.WORK, "fixed_state", "sc_gain", "gain_*.json")):
        g = json.load(open(p))
        for st in g["states"]:
            sc[(g["scene"], g["traj"], st["t"])] = st["g"]
    import t3_evalside as EV
    import v2_common as LC
    diag = {}
    for s in LC.SCENES:
        for k in LC.STARTS:
            dp = os.path.join(LC.B0_ROOT, "oracle_diag", f"diag_A2_{s}_s{k}_r0.json")
            if os.path.exists(dp):
                diag[(s, k)] = {x["t"]: x.get("g") for x in json.load(open(dp))["steps"]}

    def f(nm, r):
        if nm == "S-c":
            return sc.get((r["scene"], r["traj"], r["t"]))
        if nm == "S-a":
            return EV.gains_for(r["scene"], r["start"], r["t"], r["keys"])
        return diag.get((r["scene"], r["start"]), {}).get(r["t"])
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "probe"])
    a = ap.parse_args()
    VF.verify_frozen()
    extract() if a.mode == "extract" else probe()
