"""EXP5 Task 0.7 supplementary analysis (CPU, evaluation side; descriptive decomposition of the pre-registered shared
metrics, added for final-report questions 6 and 7, not a new endpoint).
1. Error types on the M_unknown rays of every candidate (sample values built exactly as t04_metrics.state_rows):
   miss      = label hit, sample no hit           (share of label-hit rays, averaged over the K samples)
   false_hit = label no hit, sample hit           (share of label-no-hit rays)
   both hit  : signed error x - y in voxels; early = x < y - 4, late = x > y + 4 (4 voxels = the frozen delta)
   MAE share = contribution of miss / false hit / both-hit rays to the per-sample MAE
2. Depth accuracy vs ranking quality: per method, Spearman over states between the state's CRPS and its DistP1
   Spearman / regret (cluster bootstrap over houses for the CI); and the method orderings by CRPS, Brier and regret.
Protocols M, R, O; methods frozen / C / V2 (K4). Output W/analysis/t047.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t04_metrics as TM  # noqa: E402
import t045_analyze as TA  # noqa: E402

import numpy as np  # noqa: E402

VS, TAU = TM.VS, 4.0
METHODS = ("frozen_K4", "C_K4", "V2_K4")


def samples(sid, root):
    """{method: (X [n_cand, K, R] sample values, ph hit flags)}, y, lab_hit, mu -- as t04_metrics.state_rows."""
    lab = np.load(os.path.join(root, "labels", f"{sid}.npz"))
    v2 = np.load(os.path.join(root, "V2", f"{sid}.npz"))
    cls, tmax, tk = v2["cls"], v2["tmax"].astype(np.float64), v2["t_known"].astype(np.float64)
    ts = lab["tstar"].astype(np.float64)
    mu = (cls == TM.CLS_UNKNOWN) & ~np.isnan(ts)
    lab_hit = (ts <= tmax) & mu
    y = np.where(ts <= tmax, ts, tmax)
    td = v2["t"].astype(np.float64)
    hit = np.isfinite(td) & (td <= tmax[:, None])
    known_ok = np.isfinite(tk) & (tk <= tmax)
    out = {"V2_K4": (np.where(hit, td, np.where(known_ok[:, None], tk[:, None], tmax[:, None])), hit | known_ok[:, None])}
    for m in ("frozen", "C"):
        p = os.path.join(root, m, f"{sid}.npz")
        if not os.path.exists(p) or json.load(open(os.path.join(root, m, f"{sid}.json")))["availability"] != "A1":
            continue
        z = np.load(p)
        tw = z["t"].astype(np.float64)
        hw = z["hit"] & (tw <= tmax[:, None])
        out[f"{m}_K4"] = (np.where(hw, tw, tmax[:, None]), hw)
    return out, y, lab_hit, mu


def err_types(X, ph, y, lh):
    """X, ph [K, n]; y, lh [n] (M_unknown rays only); voxels."""
    e = (X - y[None]) / VS
    a = np.abs(e)
    miss, fh, bh = lh[None] & ~ph, ~lh[None] & ph, lh[None] & ph
    tot = a.sum()
    r = {"miss": float(miss.sum() / max(1, lh.sum() * X.shape[0])),
         "false_hit": float(fh.sum() / max(1, (~lh).sum() * X.shape[0])),
         "mae_share_miss": float(a[miss].sum() / tot) if tot > 0 else None,
         "mae_share_false_hit": float(a[fh].sum() / tot) if tot > 0 else None,
         "mae_share_both_hit": float(a[bh].sum() / tot) if tot > 0 else None}
    if bh.any():
        eb = e[bh]
        r.update({"bothhit_median_abs": float(np.median(np.abs(eb))), "bothhit_early": float((eb < -TAU).mean()),
                  "bothhit_late": float((eb > TAU).mean()), "bothhit_within": float((np.abs(eb) <= TAU).mean())})
    return r


def spearman_states(a, b, g, n=2000, seed=0):
    ok = [i for i in range(len(a)) if a[i] is not None and b[i] is not None and np.isfinite(a[i]) and np.isfinite(b[i])]
    a, b, g = np.asarray(a, float)[ok], np.asarray(b, float)[ok], [g[i] for i in ok]
    if len(a) < 5:
        return None
    rho = TM.spearman(a, b)
    cl = sorted(set(g))
    idx = {c: [i for i in range(len(g)) if g[i] == c] for c in cl}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        sel = np.concatenate([idx[c] for c in rng.choice(cl, len(cl))])
        v = TM.spearman(a[sel], b[sel])
        if v is not None:
            bs.append(v)
    return {"rho": rho, "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "n": int(len(a))}


def main():
    res = {"doc": __doc__}
    for proto, (root, sf) in TA.ROOTS.items():
        rd = os.path.join(TM.W, root)
        sts = json.load(open(os.path.join(TM.W, "manifests", sf)))["states"]
        rows = []
        for st in sts:
            if not os.path.exists(os.path.join(rd, "V2", f"{st['id']}.npz")):
                continue
            sm, y, lh, mu = samples(st["id"], rd)
            r = {"state": st, "err": {}}
            for m, (X, ph) in sm.items():
                Xa = np.concatenate([X[c][:, mu[c]] for c in range(len(y))], 1)
                pa = np.concatenate([ph[c][:, mu[c]] for c in range(len(y))], 1)
                if Xa.shape[1]:
                    r["err"][m] = err_types(Xa, pa, np.concatenate([y[c][mu[c]] for c in range(len(y))]),
                                            np.concatenate([lh[c][mu[c]] for c in range(len(y))]))
            rows.append(r)
        pm = {r["state"]["id"]: r for r in TA.load(proto)}
        agg = {}
        for grp in ["all"] + TA.STRATA:
            sel = [r for r in rows if grp == "all" or r["state"]["stratum"] == grp]
            g = [r["state"]["house"] for r in sel]
            agg[grp] = {m: {k: TA.cboot([(r["err"].get(m) or {}).get(k) for r in sel], g)
                            for k in ("miss", "false_hit", "mae_share_miss", "mae_share_false_hit", "mae_share_both_hit",
                                      "bothhit_median_abs", "bothhit_early", "bothhit_late", "bothhit_within")} for m in METHODS}
        cons = {}
        ids = [r["state"]["id"] for r in rows if r["state"]["id"] in pm]
        g = [pm[i]["state"]["house"] for i in ids]
        for m in METHODS:
            crps = [(pm[i]["depth"].get(m) or {}).get("crps") for i in ids]
            cons[m] = {"crps_vs_spearman": spearman_states(crps, [(pm[i]["plan"].get(m) or {}).get("spearman") for i in ids], g),
                       "crps_vs_regret": spearman_states(crps, [(pm[i]["plan"].get(m) or {}).get("regret") for i in ids], g),
                       "brier_vs_regret": spearman_states([(pm[i]["depth"].get(m) or {}).get("brier") for i in ids],
                                                          [(pm[i]["plan"].get(m) or {}).get("regret") for i in ids], g)}
        res[proto] = {"aggregate": agg, "consistency": cons, "per_state": [{"id": r["state"]["id"], "err": r["err"]} for r in rows]}
        print(proto, len(rows), "states", flush=True)
        for m in METHODS:
            a = agg["all"][m]
            print("  ", m, {k: round(v["mean"], 3) for k, v in a.items() if v}, flush=True)
            print("  ", m, {k: (round(v["rho"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in cons[m].items() if v}, flush=True)
    od = os.path.join(TM.W, "analysis")
    json.dump(res, open(os.path.join(od, "t047.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
