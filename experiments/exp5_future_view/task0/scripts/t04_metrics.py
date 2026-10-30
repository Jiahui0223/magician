"""EXP5 Task 0.4 -- shared (cross-architecture) metrics of the indoor fixed states (CPU, evaluation side).

Inputs per state: W/fixed_state/{frozen,C,V2}/<state>.npz|json (planner side) and labels/<state>.npz (GT side).
Rays: the 64 x 114 scoring rays of every legal candidate; the evaluated rays are the P1 M_unknown rays (V2 projection
class UNKNOWN; identical P1 / region for all methods); segment end tmax = V2 projection t_max (box exit / range).
Label y = t* if t* <= tmax else no-hit (value tmax).  Sample value x_k:
  world models : cast first hit if hit and <= tmax else tmax          (K4 = worlds 0..3, K1 = world 0)
  V2           : decoded UNKNOWN hit, else the known surface t_known if <= tmax, else tmax   (K4 / K1 = sample 0)
Per method (all values in voxels, VS = 0.011 model units):
  crps     fair CRPS per ray (K = 1: absolute error), mean over rays
  es_view  per-view energy score (EXP5 T1.2 definition: rms over rays per sample, fair pair term), mean over views
  brier    hit Brier score (p = fraction of samples with a hit before tmax)
  false_conf  fraction of rays where all K samples agree on hit / no-hit and are wrong (K4 only)
  spread   mean pairwise |x_i - x_j| (K4 only)
  hit_rate_pred / hit_rate_label
Planning per state: DistP1 (K4 / K1), V2 U_raw / DistU (delta frozen), Frontier: Spearman with the true gain,
top-1, regret (argmax, ties -> lowest index; all scores 0 -> seeded uniform fallback), dead state (max gain < 0.1).
Output W/analysis/t04_shared.json (per state x method rows + stratum / house aggregates).
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

W = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp5_future_view/task0"
VS = 0.011
CLS_UNKNOWN = 0


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return None

    def rk(x):
        r = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rk(a), rk(b))[0, 1])


def pick(scores, sid):
    s = np.asarray(scores, float)
    if s.max() <= 0:
        rng = np.random.default_rng(int(hashlib.sha256(f"exp5-T0|fallback|{sid}".encode()).hexdigest()[:15], 16))
        return int(rng.integers(len(s))), True
    return int(np.flatnonzero(s == s.max())[0]), False


def one_step(scores, gain, sid):
    g = np.asarray(gain, float)
    k, fb = pick(scores, sid)
    return {"spearman": spearman(scores, g), "top1": float(g[k] == g.max()), "regret": float(g.max() - g[k]),
            "fallback": fb, "dead": bool(g.max() < 0.1)}


def depth_metrics(X, y, lab_hit, ph):
    """X [K, n] sample values, y [n] label values (voxels), lab_hit [n], ph [K, n] sample hit flags."""
    K = X.shape[0]
    mae = np.abs(X - y[None]).mean(0)
    if K > 1:
        pair = sum(np.abs(X[i] - X[j]) for i in range(K) for j in range(K) if i != j) / (K * (K - 1))
        crps = mae - 0.5 * pair
    else:
        pair, crps = np.zeros_like(mae), mae
    p = ph.mean(0)
    out = {"crps": float(crps.mean()), "mae": float(mae.mean()), "brier": float(((p - lab_hit) ** 2).mean()),
           "hit_rate_pred": float(p.mean()), "hit_rate_label": float(lab_hit.mean())}
    if K > 1:
        agree = (ph.all(0) | (~ph).all(0))
        out["false_conf"] = float((agree & (ph[0] != lab_hit)).mean())
        out["spread"] = float(pair.mean())
    return out


def view_es(X, y):
    K = X.shape[0]
    rms = lambda a: np.sqrt((a ** 2).mean())   # noqa: E731
    e1 = np.mean([rms(X[k] - y) for k in range(K)])
    if K == 1:
        return float(e1)
    pair = sum(rms(X[i] - X[j]) for i in range(K) for j in range(K) if i != j)
    return float(e1 - pair / (2 * K * (K - 1)))


def state_rows(sid, root):
    lab = np.load(os.path.join(root, "labels", f"{sid}.npz"))
    v2 = np.load(os.path.join(root, "V2", f"{sid}.npz"))
    v2j = json.load(open(os.path.join(root, "V2", f"{sid}.json")))
    gain = lab["gain"]
    cls, tmax, tk = v2["cls"], v2["tmax"].astype(np.float64), v2["t_known"].astype(np.float64)
    ts = lab["tstar"].astype(np.float64)
    mu = (cls == CLS_UNKNOWN) & ~np.isnan(ts)          # R protocol: NaN = no real sample -> not evaluated
    lab_hit = (ts <= tmax) & mu
    y = np.where(ts <= tmax, ts, tmax)
    preds = {}
    # V2
    td = v2["t"].astype(np.float64)                                   # [n, 4, R]
    hit = np.isfinite(td) & (td <= tmax[:, None])
    known_ok = np.isfinite(tk) & (tk <= tmax)
    xv = np.where(hit, td, np.where(known_ok[:, None], tk[:, None], tmax[:, None]))
    phv = hit | known_ok[:, None]
    preds["V2_K4"], preds["V2_K1"] = (xv, phv), (xv[:, :1], phv[:, :1])
    scores = {"V2_K4": v2j["distp1_K4"], "V2_K1": v2j["distp1_K1"], "Frontier": v2j["frontier_distp1"]}
    d = v2j["delta"]
    scores.update({"V2_Uraw": v2j[f"u_raw@{d}"], "V2_DistU": v2j[f"dist_u@{d}"]})
    for m in ("frozen", "C"):
        p = os.path.join(root, m, f"{sid}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        j = json.load(open(os.path.join(root, m, f"{sid}.json")))
        if j["availability"] != "A1":
            scores[f"{m}_K4"] = scores[f"{m}_K1"] = [0.0] * len(gain)
            continue
        tw = z["t"].astype(np.float64)
        hw = z["hit"] & (tw <= tmax[:, None])
        xw = np.where(hw, tw, tmax[:, None])
        preds[f"{m}_K4"], preds[f"{m}_K1"] = (xw, hw), (xw[:, :1], hw[:, :1])
        scores[f"{m}_K4"], scores[f"{m}_K1"] = j["distp1_K4"], j["distp1_K1"]
    rows = {}
    for m, (X, ph) in preds.items():
        Xs = [X[a][:, mu[a]] / VS for a in range(len(gain))]
        ys = [y[a][mu[a]] / VS for a in range(len(gain))]
        Xa = np.concatenate(Xs, 1)
        ya = np.concatenate(ys)
        pa = np.concatenate([ph[a][:, mu[a]] for a in range(len(gain))], 1)
        la = np.concatenate([lab_hit[a][mu[a]] for a in range(len(gain))])
        r = depth_metrics(Xa, ya, la, pa) if len(ya) else {}
        r["es_view"] = float(np.mean([view_es(Xs[a], ys[a]) for a in range(len(gain)) if len(ys[a])])) if len(ya) else None
        r["n_rays"] = int(len(ya))
        rows[m] = r
    plan = {m: one_step(s, gain, sid) for m, s in scores.items()}
    return rows, plan, float(gain.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(W, "fixed_state"))
    ap.add_argument("--states", default=os.path.join(W, "manifests", "indoor_states.json"))
    ap.add_argument("--out", default=os.path.join(W, "analysis", "t04_shared.json"))
    a = ap.parse_args()
    sts = json.load(open(a.states))["states"]
    per, missing = [], []
    for st in sts:
        if not (os.path.exists(os.path.join(a.root, "labels", f"{st['id']}.npz"))
                and os.path.exists(os.path.join(a.root, "V2", f"{st['id']}.npz"))):
            missing.append(st["id"])
            continue
        rows, plan, gmax = state_rows(st["id"], a.root)
        per.append({"state": st, "depth": rows, "plan": plan, "gain_max": gmax})
    agg = {}
    for grp in ["all"] + sorted({p["state"]["stratum"] for p in per}):
        sel = [p for p in per if grp == "all" or p["state"]["stratum"] == grp]
        methods = sorted({m for p in sel for m in p["depth"]} | {m for p in sel for m in p["plan"]})
        agg[grp] = {"n_states": len(sel), "houses": len({p["state"]["house"] for p in sel})}
        for m in methods:
            dm = [p["depth"][m] for p in sel if m in p["depth"] and p["depth"][m]]
            pm = [p["plan"][m] for p in sel if m in p["plan"]]
            agg[grp][m] = {**{k: float(np.mean([x[k] for x in dm if x.get(k) is not None])) for k in
                              ("crps", "mae", "es_view", "brier", "false_conf", "spread", "hit_rate_pred", "hit_rate_label")
                              if any(x.get(k) is not None for x in dm)},
                           **{f"plan_{k}": float(np.nanmean([np.nan if x[k] is None else x[k] for x in pm])) for k in
                              ("spearman", "top1", "regret") if pm}, "n": len(dm) or len(pm)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"doc": __doc__, "per_state": per, "aggregate": agg, "missing": missing}, open(a.out, "w"), indent=1, default=float)
    for grp, v in agg.items():
        print(grp, v["n_states"], {m: {k: round(x, 3) for k, x in v[m].items() if k in ("crps", "es_view", "brier", "plan_spearman", "plan_regret")}
                                     for m in v if isinstance(v[m], dict)})


if __name__ == "__main__":
    main()
