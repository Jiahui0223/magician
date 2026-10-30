"""EXP5 Task 1.2 analysis (CPU, evaluation side): learning curves, the protocol §3.7 judgement on S-c (STOP rules (a)-(c)
and the report-only distribution-quality strata), hit reliability (K = 16), and the LOSO decomposition (§3.6).
Inputs: W/training/<run>/{log,eval}.jsonl, W/eval_sc/<name>/{states,rays,rel}_*, W/eval_sets/*.json, sim_meta.json
(GT coverage, evaluation side, stratification only). Output: W/analysis/t12.json + figures.
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

CAT_HIT, CAT_NOHIT, CAT_KNOWN = 3, 1, 2
OUT = os.path.join(VF.WORK, "analysis")
T_BINS = [("t=0", 0, 0), ("t1-5", 1, 5), ("t6-20", 6, 20), ("t21-50", 21, 50), ("t51-99", 51, 99)]
LU_BINS = [("Lu<20", 0, 20), ("Lu20-100", 20, 100), ("Lu100-300", 100, 300), ("Lu>=300", 300, 1e9)]
COV_BINS = [("cov<0.10", 0, 0.10), ("cov0.10-0.20", 0.10, 0.20), ("cov0.20-0.40", 0.20, 0.40), ("cov>=0.40", 0.40, 2)]


def coverage(scene, traj, t):
    m = json.load(open(os.path.join(VF.DATA_V1, "trajectories", scene, traj, "sim_meta.json")))
    return float(m["steps"][t]["cov_gt_diagnostic"])


def load_rays(name):
    parts = sorted(glob.glob(os.path.join(VF.WORK, "eval_sc", name, "rays_*.npz")))
    out, off = {}, 0
    for p in parts:
        z = np.load(p)
        for k in z.files:
            v = z[k]
            if k == "view_id":
                v = v + off
            out.setdefault(k, []).append(v)
        off += len(z["keys"])
    return {k: np.concatenate(v) for k, v in out.items()}


def view_metrics(r):
    """Per-view fair ES (ray distance, voxels), collapsed ES, spread, skill, PIT ranks on valid M_unknown rays."""
    vid = r["view_id"]
    order = np.argsort(vid, kind="stable")
    bounds = np.flatnonzero(np.diff(vid[order])) + 1
    groups = np.split(order, bounds)
    X, y = r["t_vox"].astype(np.float64), r["tstar_vox"].astype(np.float64)
    K = X.shape[1]
    rows = []
    for g in groups:
        x, yy = X[g], y[g]
        rms = lambda a: np.sqrt((a ** 2).mean(0))   # noqa: E731
        e1 = np.mean([rms(x[:, k] - yy) for k in range(K)])
        pair = sum(rms(x[:, i] - x[:, j]) for i in range(K) for j in range(K) if i != j)
        es = e1 - pair / (2 * K * (K - 1))
        es_c = rms(x[:, 0] - yy)
        rows.append({"view": int(vid[g[0]]), "es": es, "es_collapsed": es_c, "n": len(g)})
    spread = np.mean([np.abs(X[:, i] - X[:, j]) for i in range(K) for j in range(i + 1, K)], 0)
    skill = np.abs(X.mean(1) - y)
    return rows, spread, skill


def judge(name):
    r = load_rays(name)
    lab = r["lab"]
    hit = r["hit"].astype(bool)
    s = r["s"].astype(np.float64)
    out = {"model": name, "rays": int(len(lab))}
    hs = s[hit]
    out["a_s_std_hits"] = float(hs.std()) if len(hs) else None
    nh = lab == CAT_NOHIT
    out["b_false_wall"] = float(hit[nh].mean()) if nh.any() else None
    out["c_hit_ratio"] = float(hit.mean() / max(1e-12, (lab == CAT_HIT).mean()))
    out["ambiguous_bit_rate"] = float((np.abs(r["b"].astype(np.float64)) < 0.5).mean())
    out["b_y_conflict_rate"] = float(((r["b"].astype(np.float64) > 0) & (s >= 1.0)).mean())
    th = json.load(open(os.path.join(VF.T1, "configs", "train_main.json")))["stop"]
    out["stop_a"] = out["a_s_std_hits"] is not None and out["a_s_std_hits"] < th["s_std_min"]
    out["stop_b"] = out["b_false_wall"] is not None and out["b_false_wall"] > th["false_wall_max"]
    out["stop_c"] = not (th["hit_ratio_range"][0] <= out["c_hit_ratio"] <= th["hit_ratio_range"][1])
    out["stop"] = out["stop_a"] or out["stop_b"] or out["stop_c"]
    vm, spread, skill = view_metrics(r)
    keys = r["keys"]
    meta = []
    covc = {}
    for kk in keys:
        sc, tj, t, j = kk.split("|")
        if (sc, tj, t) not in covc:
            covc[(sc, tj, t)] = coverage(sc, tj, int(t))
        meta.append((sc, tj, int(t), covc[(sc, tj, t)]))
    vid = r["view_id"]
    lu = r["lu_vox"]

    def agg(mask_views, mask_rays, nm):
        es = [v["es"] for v in vm if mask_views[v["view"]]]
        esc = [v["es_collapsed"] for v in vm if mask_views[v["view"]]]
        sp, sk = spread[mask_rays], skill[mask_rays]
        hm = hit[mask_rays]
        lb = lab[mask_rays]
        Xr, yr = r["t_vox"][mask_rays], r["tstar_vox"][mask_rays]
        ranks = ((Xr < yr[:, None]).sum(1))[lb == CAT_HIT] if (lb == CAT_HIT).any() else np.zeros(0)
        d = {"views": len(es), "rays": int(mask_rays.sum()), "fair_es_vox": float(np.mean(es)) if es else None,
             "es_over_collapsed": float(np.mean(es) / np.mean(esc)) if es and np.mean(esc) > 0 else None,
             "spread_vox": float(sp.mean()) if len(sp) else None, "skill_vox": float(sk.mean()) if len(sk) else None,
             "spread_skill": float(sp.mean() / sk.mean()) if len(sk) and sk.mean() > 0 else None,
             "false_wall": float(hm[lb == CAT_NOHIT].mean()) if (lb == CAT_NOHIT).any() else None,
             "hit_ratio": float(hm.mean() / max(1e-12, (lb == CAT_HIT).mean())) if len(lb) else None,
             "pit_hist": np.bincount(ranks, minlength=hit.shape[1] + 1).tolist()}
        return d
    nv = len(keys)
    strata = {"all": agg(np.ones(nv, bool), np.ones(len(lab), bool), "all")}
    for nm, lo, hi in T_BINS:
        mv = np.array([lo <= m[2] <= hi for m in meta])
        strata[nm] = agg(mv, mv[vid], nm)
    for nm, lo, hi in COV_BINS:
        mv = np.array([lo <= m[3] < hi for m in meta])
        strata[nm] = agg(mv, mv[vid], nm)
    for nm, lo, hi in LU_BINS:
        mr = (lu >= lo) & (lu < hi)
        mv = np.zeros(nv, bool)
        mv[np.unique(vid[mr])] = True
        strata[nm] = agg(mv, mr, nm)
    for sc in sorted({m[0] for m in meta}):
        mv = np.array([m[0] == sc for m in meta])
        strata[sc] = agg(mv, mv[vid], sc)
    # mode-collapse / low-uncertainty / noise classification (report only)
    cls = {}
    errs = {k: v["skill_vox"] for k, v in strata.items() if k.startswith("Lu") and v["skill_vox"] is not None}
    if errs:
        worst = max(errs, key=errs.get)
        w = strata[worst]
        cls = {"largest_error_stratum": worst, "spread_skill_there": w["spread_skill"],
               "true_mode_collapse": w["spread_skill"] is not None and w["spread_skill"] < 0.2,
               "noise_like_strata": [k for k, v in strata.items() if v["es_over_collapsed"] is not None and v["es_over_collapsed"] >= 1.0]}
    out["strata"] = strata
    out["diversity_classification"] = cls
    rel = sorted(glob.glob(os.path.join(VF.WORK, "eval_sc", name, "rel_*.npz")))
    if rel:
        p = np.concatenate([np.load(x)["p16"] for x in rel])
        lb = np.concatenate([np.load(x)["lab"] for x in rel]).astype(float)
        bins = np.minimum((p * 10).astype(int), 9)
        curve = [{"bin": b / 10, "n": int((bins == b).sum()), "p_mean": float(p[bins == b].mean()) if (bins == b).any() else None,
                  "freq": float(lb[bins == b].mean()) if (bins == b).any() else None} for b in range(10)]
        ece = sum(c["n"] * abs(c["p_mean"] - c["freq"]) for c in curve if c["n"]) / len(p)
        out["reliability_K16"] = {"rays": int(len(p)), "brier": float(((p - lb) ** 2).mean()), "ece": float(ece), "curve": curve}
    return out


def curves(run):
    d = os.path.join(VF.WORK, "training", run)
    ev = [json.loads(x) for x in open(os.path.join(d, "eval.jsonl"))] if os.path.exists(os.path.join(d, "eval.jsonl")) else []
    lg = [json.loads(x) for x in open(os.path.join(d, "log.jsonl"))] if os.path.exists(os.path.join(d, "log.jsonl")) else []
    return ev, lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="main,loso_pisa")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for m in a.models.split(","):
        if glob.glob(os.path.join(VF.WORK, "eval_sc", m, "rays_*.npz")):
            res[m] = judge(m)
            print(m, json.dumps({k: res[m][k] for k in ("a_s_std_hits", "b_false_wall", "c_hit_ratio", "stop")}), flush=True)
        ev, lg = curves(m)
        res.setdefault(m, {})["curves"] = {"eval": ev, "train_loss": [(x["step"], x["loss"]) for x in lg]}
    for p in glob.glob(os.path.join(VF.WORK, "eval_sets", "*.json")):
        res.setdefault("eval_sets", {})[os.path.basename(p)[:-5]] = json.load(open(p))
    json.dump(res, open(os.path.join(OUT, "t12.json"), "w"), indent=1, default=float)
    figures(res)


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for m, st in (("main", "-"), ("loso_pisa", "--"), ("scale3", ":")):
        c = res.get(m, {}).get("curves")
        if not c:
            continue
        tl = np.asarray(c["train_loss"])
        if len(tl):
            ax[0].plot(tl[:, 0], tl[:, 1], st, label=f"{m} train (batch)", alpha=0.6)
        ev = c["eval"]
        for key in ("fm_train_sub", "fm_sc_sub", "fm_pisa_sub"):
            pts = [(e["step"], e[key]) for e in ev if key in e]
            if pts:
                ax[1].plot(*zip(*pts), st, marker="o", ms=3, label=f"{m} {key[3:]}")
        for key in ("samp_train_sub", "samp_sc_sub", "samp_pisa_sub"):
            pts = [(e["step"], e[key]["fair_es_arclength_vox"]) for e in ev if key in e]
            if pts:
                ax[2].plot(*zip(*pts), st, marker="s", ms=3, label=f"{m} {key[5:]}")
    ax[0].set_title("training loss (50-step mean)")
    ax[1].set_title("EMA deterministic FM loss")
    ax[2].set_title("EMA fair ES, unknown-arclength coords (vox)")
    for x in ax:
        x.set_xlabel("step")
        x.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "t12_learning_curves.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
