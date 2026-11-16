"""Experiment G2 items 1-2 -- metrics, strata, statistics and figures from g2_prob.py / g2_pool.py outputs (protocol G2-v1 sections 1-2;
CPU only). Sets: VALSEL-in (prob/in_val), S-c-out (prob/out_sc), F-pool (pool/*.npz; strata edges = label-only quartiles of the pooled
arrays, see logs/EVENTS.md). Arms: D0, G1 (= sample k0), G4 (k0..3), G16 (k0..15).
Statistics: trajectory-cluster bootstrap, 10,000 draws, seed 0, 95 % percentile; ratio metrics as ratio of cluster sums; paired differences
on the same rays / views; AUROC / AP from per-cluster score histograms (G: exact K + 1 levels; D0: 3,000 bins of b_hat on [-1.5, 1.5]);
point estimates of AUROC / AP exact (sklearn); Spearman CI = cluster-mean version.
Writes results/g2/{item1_*.csv, item2_*.csv, g2_items12.json, G2_ITEMS12.md} and figures/g2/{reliability,spread_skill}_<set>.{pdf,png}.
Usage: python g2_analyze.py [--sets in_val,out_sc,pool] [--root DIR --out DIR (tests)]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
GD = os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic")
WG2 = os.path.join(R, "work_dirs/exp6_v3/experiment_G/g2")
OUT, FIG = os.path.join(GD, "results/g2"), os.path.join(GD, "figures/g2")
NB = 10000
SETNAME = {"in_val": "VALSEL-in", "out_sc": "S-c-out", "pool": "F-pool"}
ARMS = ["D0", "G1", "G4", "G16"]
EPS_LL = 0.01
TAU_COV = 1.0


# ------------------------------------------------------------------------------------------------ loading
def load_set(name):
    if name in ("in_val", "out_sc"):
        d = os.path.join(WG2, "prob", name)
        z = np.load(os.path.join(d, "rays.npz"))
        rays = {k: z[k] for k in z.files}
        vz = np.load(os.path.join(d, "views.npz"))
        vr = vz["rows"]
        views = {"cluster": vr[:, 1].astype(int), "es": {"G1": vr[:, 3], "G4": vr[:, 4], "G16": vr[:, 5], "D0": vr[:, 6]}}
        edges = json.load(open(os.path.join(d, "edges.json")))
        reg = json.load(open(os.path.join(d, "regression.json")))
        return rays, views, edges, reg
    fs = sorted(f for f in glob.glob(os.path.join(WG2, "pool", "*.npz")) if not f.endswith("_fig.npz"))
    parts, vparts = [], {"cluster": [], "es": {a: [] for a in ARMS}}
    for ci, f in enumerate(fs):
        z = np.load(f)
        p = {k: z[k] for k in z.files}
        p["cluster"] = np.full(len(p["lu"]), ci, np.int32)
        parts.append(p)
        vj = json.load(open(f[:-4] + "_views.json"))["views"]
        vparts["cluster"].append(np.full(len(vj), ci))
        for a in ARMS:
            vparts["es"][a].append(np.asarray([v[f"es_{a}"] for v in vj]))
    rays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    views = {"cluster": np.concatenate(vparts["cluster"]).astype(int), "es": {a: np.concatenate(v) for a, v in vparts["es"].items()}}
    lab = rays["label"]
    edges = {"set": "pool", "A_lu_quartiles": [float(np.quantile(rays["lu"], q)) for q in (0.25, 0.5, 0.75)],
             "B_lstar_quartiles": [float(np.quantile(rays["lstar"][lab], q)) for q in (0.25, 0.5, 0.75)], "n_labelled": int(len(lab)), "n_gt_hit": int(lab.sum()),
             "n_trajectories": len(fs), "note": "label-only quartiles of the stored pool arrays (deviation: computed in the analysis, not before the forward)"}
    return rays, views, edges, {"trajectories": [os.path.basename(f)[:-4] for f in fs]}


# ------------------------------------------------------------------------------------------------ per-ray quantities
def pair_sum(x):
    """sum_{i<j} |x_i - x_j| per row (sorted trick), x [N, K]."""
    K = x.shape[1]
    s = np.sort(x, 1)
    w = 2 * np.arange(1, K + 1) - K - 1
    return s @ w


def arm_arrays(rays, arm):
    if arm == "D0":
        ell, hit = rays["d_ell"][:, None].astype(np.float64), rays["d_hit"][:, None]
        p = np.clip((rays["d_b"].astype(np.float64) + 1) / 2, 0, 1)
        score = rays["d_b"].astype(np.float64)
    else:
        K = {"G1": 1, "G4": 4, "G16": 16}[arm]
        ell, hit = rays["g_ell"][:, :K].astype(np.float64), rays["g_hit"][:, :K]
        p = hit.mean(1)
        score = p
    K = ell.shape[1]
    lt = rays["lstar"].astype(np.float64)
    lab = rays["label"]
    err = np.abs(ell - lt[:, None])
    t1 = err.mean(1)
    ps = pair_sum(ell) if K > 1 else np.zeros(len(lt))
    crps = t1 - ps / (K * (K - 1)) if K > 1 else t1
    spread = 2 * ps / (K * (K - 1)) if K > 1 else np.zeros(len(lt))
    tp = hit & lab[:, None]
    cov = np.where(lab, (hit & (err <= TAU_COV)).any(1), (~hit).any(1))
    o = {"K": K, "crps": crps, "err1": t1, "spread": spread, "p": p, "score": score, "hit_mean": hit.mean(1), "tp_n": tp.sum(1), "hit_n": hit.sum(1),
         "tp_err": np.where(tp, err, 0).sum(1), "cover": cov.astype(np.float64)}
    pc = np.clip(p, EPS_LL, 1 - EPS_LL)
    o["brier"] = (p - lab) ** 2
    o["logloss"] = -(lab * np.log(pc) + (~lab) * np.log(1 - pc))
    if arm == "D0":
        o["brier_hard"] = (hit[:, 0].astype(np.float64) - lab) ** 2
    return o


# ------------------------------------------------------------------------------------------------ cluster bootstrap machinery
class Boot:
    def __init__(self, cluster, n=NB, seed=0):
        self.u, self.ci = np.unique(cluster, return_inverse=True)
        C = len(self.u)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, C, (n, C))
        self.W = np.zeros((n, C))
        for b in range(n):
            self.W[b] = np.bincount(idx[b], minlength=C)
        self.C = C

    def sums(self, x, mask=None):
        x = np.asarray(x, np.float64)
        if mask is not None:
            x = np.where(mask, x, 0.0)
        return np.bincount(self.ci, weights=x, minlength=self.C)

    def ratio(self, num, den, mask=None):
        """-> point, boot [n] of sum(num) / sum(den)."""
        a, b = self.sums(num, mask), self.sums(den, mask)
        with np.errstate(invalid="ignore", divide="ignore"):
            return a.sum() / b.sum(), (self.W @ a) / (self.W @ b)


def ci(bv):
    bv = bv[np.isfinite(bv)]
    return (float(np.percentile(bv, 2.5)), float(np.percentile(bv, 97.5))) if len(bv) else (None, None)


def hist_auc_ap(pos_h, neg_h):
    """pos_h, neg_h [..., B] counts per score bin in ASCENDING score order -> AUROC, AP (ties within a bin = one threshold)."""
    P, N = pos_h.sum(-1), neg_h.sum(-1)
    cneg = np.cumsum(neg_h, -1) - neg_h                       # negatives strictly below
    auc = ((pos_h * (cneg + 0.5 * neg_h)).sum(-1)) / (P * N)
    tp = np.cumsum(pos_h[..., ::-1], -1)
    fp = np.cumsum(neg_h[..., ::-1], -1)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P[..., None]
    drec = np.diff(np.concatenate([np.zeros(rec.shape[:-1] + (1,)), rec], -1), axis=-1)
    ap = (drec * prec).sum(-1)
    return auc, ap


def score_bins(arm, score, K):
    if arm == "D0":
        e = np.linspace(-1.5, 1.5, 3001)
        return np.clip(np.digitize(score, e) - 1, 0, 2999), 3000
    return np.rint(score * K).astype(int), K + 1


# ------------------------------------------------------------------------------------------------ analysis of one set
def analyze(name):
    rays, views, edges, reg = load_set(name)
    lab = rays["label"]
    bt = Boot(rays["cluster"])
    btv = Boot(views["cluster"])
    A = {a: arm_arrays(rays, a) for a in ARMS}
    one = np.ones(len(lab))
    res = {"set": SETNAME[name], "n_rays": int(len(lab)), "n_gt_hit": int(lab.sum()), "n_views": int(len(views["cluster"])), "n_clusters": bt.C, "edges": edges,
           "regression": reg, "item1": {}, "paired": {}, "item2": {}, "h2": {}}
    # ---------------- item 1 ratio metrics (point + boot per arm)
    defs = {"crps": lambda o: (o["crps"], one), "brier_soft": lambda o: (o["brier"], one), "logloss": lambda o: (o["logloss"], one),
            "hit_ratio": lambda o: (o["hit_mean"], lab.astype(float))}
    boots = {}
    for a in ARMS:
        o = A[a]
        r = {}
        for m, f in defs.items():
            pt, bv = bt.ratio(*f(o))
            r[m] = pt
            boots[(a, m)] = bv
        if a == "D0":
            pt, bv = bt.ratio(o["brier_hard"], one)
            r["brier_hard"] = pt
            boots[(a, "brier_hard")] = bv
        # fair ES per view
        pt, bv = btv.ratio(views["es"][a], np.ones(len(views["es"][a])))
        r["fair_es"] = pt
        boots[(a, "fair_es")] = bv
        r["abs_1_minus_hit_ratio"] = abs(1 - r["hit_ratio"])
        boots[(a, "abs_1_minus_hit_ratio")] = np.abs(1 - boots[(a, "hit_ratio")])
        # reliability + ECE (10 bins)
        pb = np.clip((o["p"] * 10).astype(int), 0, 9)
        n_b = np.stack([bt.sums(pb == b) for b in range(10)], 1)          # [C, 10]
        sp_b = np.stack([bt.sums(np.where(pb == b, o["p"], 0)) for b in range(10)], 1)
        sy_b = np.stack([bt.sums((pb == b) & lab) for b in range(10)], 1)
        nb, spb, syb = n_b.sum(0), sp_b.sum(0), sy_b.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r["reliability"] = {"n": nb.tolist(), "mean_p": (spb / nb).tolist(), "freq": (syb / nb).tolist()}
            r["ece"] = float(np.nansum(nb / nb.sum() * np.abs(spb / nb - syb / nb)))
            Bn, Bp, By = bt.W @ n_b, bt.W @ sp_b, bt.W @ sy_b
            boots[(a, "ece")] = np.nansum(Bn / Bn.sum(1, keepdims=True) * np.abs(Bp / Bn - By / Bn), 1)
        # AUROC / AP: exact point + histogram bootstrap
        from sklearn.metrics import average_precision_score, roc_auc_score
        r["auroc"] = float(roc_auc_score(lab, o["score"]))
        r["ap"] = float(average_precision_score(lab, o["score"]))
        sb, nbins = score_bins(a, o["score"], o["K"])
        ph = np.zeros((bt.C, nbins))
        nh = np.zeros((bt.C, nbins))
        np.add.at(ph, (bt.ci[lab], sb[lab]), 1)
        np.add.at(nh, (bt.ci[~lab], sb[~lab]), 1)
        auc_b, ap_b = hist_auc_ap(bt.W @ ph, bt.W @ nh)
        auc_h, ap_h = hist_auc_ap(ph.sum(0), nh.sum(0))
        r["auroc_hist"], r["ap_hist"] = float(auc_h), float(ap_h)
        boots[(a, "auroc")], boots[(a, "ap")] = auc_b, ap_b
        # spread-skill (G with K > 1)
        if o["K"] > 1:
            from scipy.stats import spearmanr
            rho = spearmanr(o["spread"], o["err1"]).correlation
            per = []
            for c in range(bt.C):
                m = bt.ci == c
                if m.sum() > 50 and np.ptp(o["spread"][m]) > 0:
                    per.append(spearmanr(o["spread"][m], o["err1"][m]).correlation)
            per = np.asarray([x for x in per if np.isfinite(x)])
            rng = np.random.default_rng(0)
            pb_ = per[rng.integers(0, len(per), (NB, len(per)))].mean(1) if len(per) else np.array([np.nan])
            qs = np.unique(np.quantile(o["spread"], np.linspace(0, 1, 11)))
            bi = np.clip(np.digitize(o["spread"], qs[1:-1]), 0, len(qs) - 2)
            r["spread_skill"] = {"spearman_pooled": float(rho), "spearman_cluster_mean": float(per.mean()) if len(per) else None, "ci": ci(pb_),
                                 "n_clusters_rho": int(len(per)), "bin_edges": qs.tolist(),
                                 "bins": [{"n": int((bi == b).sum()), "spread": float(o["spread"][bi == b].mean()), "error": float(o["err1"][bi == b].mean())}
                                          for b in range(len(qs) - 1) if (bi == b).any()]}
        for m in list(r):
            if (a, m) in boots:
                r[f"{m}_ci"] = ci(boots[(a, m)])
        res["item1"][a] = r
    # ---------------- paired differences (x - y), positive = y lower for "lower is better" metrics
    for x, y in (("D0", "G4"), ("D0", "G1"), ("D0", "G16"), ("G1", "G4"), ("G4", "G16")):
        pr = {}
        for m in ("crps", "fair_es", "brier_soft", "logloss", "ece", "auroc", "ap", "abs_1_minus_hit_ratio"):
            if (x, m) in boots and (y, m) in boots:
                d = res["item1"][x][m] - res["item1"][y][m]
                pr[m] = {"diff": float(d), "ci": ci(boots[(x, m)] - boots[(y, m)])}
        if x == "D0":
            d = res["item1"]["D0"]["brier_hard"] - res["item1"][y]["brier_soft"]
            pr["brier_hard_D0_vs_soft"] = {"diff": float(d), "ci": ci(boots[("D0", "brier_hard")] - boots[(y, "brier_soft")])}
        res["paired"][f"{x}-{y}"] = pr
    # ---------------- item 2 strata
    qa, qb = edges["A_lu_quartiles"], edges["B_lstar_quartiles"]
    sa = np.digitize(rays["lu"], qa)                                  # 0..3 -> A1..A4
    sbq = np.digitize(rays["lstar"], qb)
    strata = {f"A{i + 1}": sa == i for i in range(4)}
    strata.update({f"B{i + 1}": lab & (sbq == i) for i in range(4)})
    strata["B0"] = ~lab
    strata["C-edge"], strata["C-int"] = rays["edgeC"], ~rays["edgeC"]
    if "edgeD" in rays:
        strata["D-edge"], strata["D-smooth"] = rays["edgeD"], lab & ~rays["edgeD"]
    sboot = {}
    for sname, m in strata.items():
        row = {"n_rays": int(m.sum()), "n_gt_hit": int((m & lab).sum())}
        for a in ARMS:
            o = A[a]
            K = o["K"]
            met = {"recall": bt.ratio(o["tp_n"] / K, lab.astype(float), m), "crps": bt.ratio(o["crps"], one, m),
                   "tp_err": bt.ratio(o["tp_err"], o["tp_n"].astype(float), m), "cover_hit": bt.ratio(o["cover"], one, m & lab),
                   "cover_nohit": bt.ratio(o["cover"], one, m & ~lab)}
            if not sname.startswith("B") and not sname.startswith("D"):
                met["precision"] = bt.ratio(o["tp_n"].astype(float), o["hit_n"].astype(float), m)
            row[a] = {k: (float(v[0]) if np.isfinite(v[0]) else None) for k, v in met.items()}
            row[a].update({f"{k}_ci": ci(v[1]) for k, v in met.items()})
            sboot[(sname, a)] = met["crps"][1]
        dd = sboot[(sname, "D0")] - sboot[(sname, "G4")]
        row["delta_crps_D0_minus_G4"] = {"diff": row["D0"]["crps"] - row["G4"]["crps"] if row["D0"]["crps"] is not None else None, "ci": ci(dd)}
        dd1 = sboot[(sname, "D0")] - sboot[(sname, "G1")]
        row["delta_crps_D0_minus_G1"] = {"diff": row["D0"]["crps"] - row["G1"]["crps"] if row["D0"]["crps"] is not None else None, "ci": ci(dd1)}
        res["item2"][sname] = row
    for cname, (hi, lo) in {"A4-A1": ("A4", "A1"), "B4-B1": ("B4", "B1"), "Cedge-Cint": ("C-edge", "C-int"), "Dedge-Dsmooth": ("D-edge", "D-smooth")}.items():
        if hi not in strata:
            continue
        d_hi = sboot[(hi, "D0")] - sboot[(hi, "G4")]
        d_lo = sboot[(lo, "D0")] - sboot[(lo, "G4")]
        pt = res["item2"][hi]["delta_crps_D0_minus_G4"]["diff"] - res["item2"][lo]["delta_crps_D0_minus_G4"]["diff"]
        res["h2"][cname] = {"contrast": float(pt), "ci": ci(d_hi - d_lo)}
    return res


# ------------------------------------------------------------------------------------------------ verdicts, tables, figures
def verdict(cis, positive=True):
    """cis: list of (lo, hi) per set -> supported / contradicted / inconclusive."""
    if not cis:
        return "未计算"
    ok = [c[0] is not None and (c[0] > 0 if positive else c[1] < 0) for c in cis]
    bad = [c[0] is not None and (c[1] < 0 if positive else c[0] > 0) for c in cis]
    return "支持" if all(ok) else ("反对" if any(bad) else "不确定")


def f3(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def fci(c, nd=4):
    return "—" if c is None or c[0] is None else f"[{c[0]:+.{nd}f}, {c[1]:+.{nd}f}]"


def figures(allres):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    col = {"D0": "#2471a3", "G1": "#7d3c98", "G4": "#c0392b", "G16": "#e67e22"}
    fig, axes = plt.subplots(1, len(allres), figsize=(4.2 * len(allres), 3.8))
    for ax, (nm, r) in zip(np.atleast_1d(axes), allres.items()):
        ax.plot([0, 1], [0, 1], color="k", lw=0.7, ls="--")
        for a in ARMS:
            rl = r["item1"][a]["reliability"]
            mp, fr, n = np.asarray(rl["mean_p"], float), np.asarray(rl["freq"], float), np.asarray(rl["n"], float)
            ok = n > 0
            ax.plot(mp[ok], fr[ok], marker="o", ms=3.5, lw=1.3, color=col[a], label=f"{a} (ECE {r['item1'][a]['ece']:.3f})")
        ax.set_title(r["set"], fontsize=10)
        ax.set_xlabel("predicted hit probability")
        ax.set_ylabel("observed hit frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"reliability.{ext}"), dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(allres), figsize=(4.2 * len(allres), 3.6))
    for ax, (nm, r) in zip(np.atleast_1d(axes), allres.items()):
        mx = 0
        for a in ("G4", "G16"):
            ss = r["item1"][a]["spread_skill"]
            xs = [b["spread"] for b in ss["bins"]]
            ys = [b["error"] for b in ss["bins"]]
            mx = max(mx, max(xs + ys))
            ax.plot(xs, ys, marker="o", ms=3.5, lw=1.3, color=col[a], label=f"{a} (ρ {ss['spearman_pooled']:.2f})")
        ax.plot([0, mx], [0, mx], color="k", lw=0.7, ls="--")
        ax.set_title(r["set"], fontsize=10)
        ax.set_xlabel("sample spread (arc-length vox)")
        ax.set_ylabel("mean |ℓ_k − ℓ*| (vox)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"spread_skill.{ext}"), dpi=200)
    plt.close(fig)


def write(allres):
    os.makedirs(OUT, exist_ok=True)
    json.dump(allres, open(os.path.join(OUT, "g2_items12.json"), "w"), indent=1, default=float)
    md = ["# G2 items 1–2 — probabilistic quality and ambiguity strata (protocol G2-v1 §1–§2)", "",
          "Trajectory-cluster bootstrap, 10,000 draws, seed 0, 95 % CI. Lower is better for CRPS / ES / Brier / log-loss / ECE / |1−hit_ratio|; higher for AUROC / AP.", ""]
    rows = []
    for nm, r in allres.items():
        md += [f"## {r['set']} ({r['n_rays']:,} labelled UNKNOWN rays, {r['n_gt_hit']:,} GT hits, {r['n_views']} views, {r['n_clusters']} trajectories)", ""]
        if "regression_T0" in r["regression"]:
            md.append(f"Regression vs eval.jsonl step 40000: G4 {r['regression'].get('regression_T0_pass')}, D0 {r['regression'].get('regression_D0_pass')}.")
            md.append("")
        md += ["| Arm | fair ES | CRPS | hit_ratio | Brier (soft) | Brier (D0 hard) | log-loss | ECE | AUROC | AP |", "|---|---|---|---|---|---|---|---|---|---|"]
        for a in ARMS:
            x = r["item1"][a]
            md.append(f"| {a} | {f3(x['fair_es'], 3)} | {f3(x['crps'], 3)} | {f3(x['hit_ratio'], 3)} | {f3(x['brier_soft'])} | {f3(x.get('brier_hard'))} | "
                      f"{f3(x['logloss'])} | {f3(x['ece'])} | {f3(x['auroc'])} | {f3(x['ap'])} |")
            rows.append([r["set"], a] + [x.get(k) for k in ("fair_es", "crps", "hit_ratio", "brier_soft", "brier_hard", "logloss", "ece", "auroc", "ap")])
        md += ["", "| Paired | fair ES | CRPS | Brier soft | log-loss | ECE | AUROC | AP | |1−hit_ratio| |", "|---|---|---|---|---|---|---|---|---|"]
        for p, v in r["paired"].items():
            md.append(f"| {p} | " + " | ".join(f"{v[m]['diff']:+.4f} {fci(v[m]['ci'])}" if m in v else "—"
                                                for m in ("fair_es", "crps", "brier_soft", "logloss", "ece", "auroc", "ap", "abs_1_minus_hit_ratio")) + " |")
        if "brier_hard_D0_vs_soft" in r["paired"].get("D0-G4", {}):
            v = r["paired"]["D0-G4"]["brier_hard_D0_vs_soft"]
            md.append(f"\nD0 hard Brier − G4 soft Brier: {v['diff']:+.4f} {fci(v['ci'])}")
        md += ["", "Spread–skill: " + "; ".join(f"{a} pooled ρ {r['item1'][a]['spread_skill']['spearman_pooled']:.3f}, cluster-mean ρ "
                                              f"{f3(r['item1'][a]['spread_skill']['spearman_cluster_mean'], 3)} {fci(r['item1'][a]['spread_skill']['ci'], 3)}" for a in ("G4", "G16")), ""]
        md += [f"### Strata (edges A {[round(x, 1) for x in r['edges']['A_lu_quartiles']]}, B {[round(x, 1) for x in r['edges']['B_lstar_quartiles']]} vox)", "",
               "| Stratum | rays | CRPS D0 / G1 / G4 / G16 | ΔCRPS D0−G4 [CI] | recall D0 / G4 | precision D0 / G4 | TP error D0 / G4 | covered (GT hit) D0 / G1 / G4 / G16 | covered (GT no-hit) D0 / G4 / G16 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for s, x in r["item2"].items():
            g = lambda a, k: x[a].get(k)  # noqa: E731
            md.append(f"| {s} | {x['n_rays']:,} | {' / '.join(f3(g(a, 'crps'), 2) for a in ARMS)} | {f3(x['delta_crps_D0_minus_G4']['diff'], 3)} {fci(x['delta_crps_D0_minus_G4']['ci'], 3)} | "
                      f"{f3(g('D0', 'recall'), 3)} / {f3(g('G4', 'recall'), 3)} | {f3(g('D0', 'precision'), 3)} / {f3(g('G4', 'precision'), 3)} | "
                      f"{f3(g('D0', 'tp_err'), 2)} / {f3(g('G4', 'tp_err'), 2)} | {' / '.join(f3(g(a, 'cover_hit'), 3) for a in ARMS)} | "
                      f"{' / '.join(f3(g(a, 'cover_nohit'), 3) for a in ('D0', 'G4', 'G16'))} |")
        md += ["", "H2 contrasts (Δ = CRPS D0 − CRPS G4; contrast = Δ(high) − Δ(low)): " +
               "; ".join(f"{k} {v['contrast']:+.3f} {fci(v['ci'], 3)}" for k, v in r["h2"].items()), ""]
    # verdicts
    sets = list(allres.values())
    h1 = {m: verdict([s["paired"]["D0-G4"][m]["ci"] for s in sets]) for m in ("crps", "fair_es")}
    h1d = verdict([s["item1"]["G4"]["spread_skill"]["ci"] for s in sets])
    h2 = {}
    for c in ("A4-A1", "B4-B1", "Cedge-Cint", "Dedge-Dsmooth"):
        cis = [s["h2"][c]["ci"] for s in sets if c in s["h2"]]
        h2[c] = verdict(cis)
    md += ["## 预注册假设的判定（协议 §1、§2）", "", f"- H1（D0 − G4 > 0，三个集都 CI 不含 0）：CRPS **{h1['crps']}**，fair ES **{h1['fair_es']}**",
           f"- H1d（G4 的 spread–skill ρ > 0）：**{h1d}**", "- H2（ΔCRPS 随歧义增大）：" + "；".join(f"{k} **{v}**" for k, v in h2.items()), ""]
    open(os.path.join(OUT, "G2_ITEMS12.md"), "w").write("\n".join(md) + "\n")
    import csv
    with open(os.path.join(OUT, "item1_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "arm", "fair_es", "crps", "hit_ratio", "brier_soft", "brier_hard", "logloss", "ece", "auroc", "ap"])
        w.writerows(rows)
    with open(os.path.join(OUT, "item2_strata.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "stratum", "arm", "n_rays", "crps", "recall", "precision", "tp_err", "cover_hit", "cover_nohit"])
        for r in allres.values():
            for s, x in r["item2"].items():
                for a in ARMS:
                    w.writerow([r["set"], s, a, x["n_rays"]] + [x[a].get(k) for k in ("crps", "recall", "precision", "tp_err", "cover_hit", "cover_nohit")])
    print("\n".join(md))
    return {"H1": h1, "H1d": h1d, "H2": h2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="in_val,out_sc,pool")
    ap.add_argument("--root", default=None, help="test only: input root instead of work_dirs/exp6_v3/experiment_G/g2")
    ap.add_argument("--out", default=None, help="test only: output dir for tables and figures")
    a = ap.parse_args()
    global WG2, OUT, FIG
    if a.root:
        WG2 = a.root
    if a.out:
        OUT, FIG = a.out, a.out
    allres = {s: analyze(s) for s in a.sets.split(",")}
    v = write(allres)
    figures(allres)
    json.dump(v, open(os.path.join(OUT, "g2_verdicts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
