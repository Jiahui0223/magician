"""Experiment U Phase D -- analysis of the fixed-state rows (CPU; seen2scene env). Reads work_dirs/exp6_v3/experiment_U/fixed/states/*.jsonl
(u_fixed.py) and writes results/fixed/*.csv|json and figures/U_fixed_*.png. GT enters only through the existing oracle labels
(official one-step coverage gain, Experiment F rows) and the offline gt_attr fields.
Pick rule = m_score.choose exactly (argmax, ties -> lowest index, max == 0 -> seeded MC.fallback_choice(scene, start, t, n)).
Statistics: scene-equal means (state -> scene mean -> mean over scenes), paired scene bootstrap 10,000 draws, seed 0 (cvpr_lib.boot_ci).
Usage: python u_analyze_fixed.py"""
import csv
import glob
import json
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
HERE = os.path.dirname(os.path.abspath(__file__))
UD = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "scripts"))
import cvpr_lib as L  # noqa: E402

sys.path.insert(0, os.path.join(R, "experiments", "exp3b_macaronspp_transfer"))
import m_common as MC  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402

W = os.path.join(R, "work_dirs", "exp6_v3", "experiment_U", "fixed")
OUT = os.path.join(UD, "results", "fixed")
FIG = os.path.join(UD, "figures")
PLANNER = ["D1", "C1", "U1", "U2", "U3"]                       # the frozen / pre-registered scores
OFFLINE = ["U2rel", "U3sum", "U1sum", "Chamfer", "1-Jacc"]     # diagnostics only, never planner scores
COL = {"D1": "#2a78d6", "C1": "#1baf7a", "U1": "#eb6834", "U2": "#6250d6", "U3": "#e34948"}


def choose(s, scene, start, t):
    s = np.asarray(s, np.float64)
    if s.max() == 0.0:
        return MC.fallback_choice(scene, start, t, len(s)), True
    return int(np.flatnonzero(s == s.max())[0]), False


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(W, "states", "*.jsonl"))):
        rows += [json.loads(x) for x in open(f)]
    return rows


def scores_of(r):
    u = r["u"]
    sc = {"D1": r["d1"], "C1": r["c1"], "U1": [m["u1"] for m in u], "U2": [m["u2"] for m in u], "U3": [m["u3"] for m in u],
          "U2rel": [m.get("u2_rel") or 0.0 for m in u], "U3sum": [m["u3_sum"] for m in u], "U1sum": [m["u1_sum"] for m in u],
          "Chamfer": [m.get("chamfer_rt") or 0.0 for m in u], "1-Jacc": [1.0 - m["jaccard_pair_mean"] if m.get("jaccard_pair_mean") is not None else 0.0 for m in u]}
    return {k: np.asarray(v, np.float64) for k, v in sc.items()}


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else None


def scene_mean(vals, scenes):
    """{scene: mean of the state values (None dropped)}"""
    d = {}
    for v, s in zip(vals, scenes):
        if v is not None and np.isfinite(v):
            d.setdefault(s, []).append(v)
    return {s: float(np.mean(x)) for s, x in d.items()}


def se_mean(d):
    return float(np.mean(list(d.values()))) if d else None


def paired_ci(da, db):
    ss = sorted(set(da) & set(db))
    m, lo, hi = L.boot_ci([da[s] - db[s] for s in ss])
    return m, lo, hi, len(ss), int(sum(da[s] > db[s] for s in ss)), int(sum(da[s] < db[s] for s in ss))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    assert rows, "no fixed-state rows"
    names = PLANNER + OFFLINE
    reg = {"n_states": len(rows), "keys_eq_F": sum(r["regression"]["keys_eq_F"] for r in rows), "paths_eq_F": sum(r["regression"]["paths_eq_F"] for r in rows),
           "d1_bitwise_T4V2_T0": sum(1 for r in rows if r["regression"]["d1_bitwise_T4V2_T0"] is True),
           "d1_ref_missing": sum(1 for r in rows if r["regression"]["d1_bitwise_T4V2_T0"] is None),
           "c1_equals_meanV1k": sum(1 for r in rows if np.allclose(r["c1"], np.mean(r["V1k"], 1), rtol=0, atol=0))}
    st, cand = [], []
    for i, r in enumerate(rows):
        sc = scores_of(r)
        o = np.asarray(r["oracle_pp"], np.float64)
        omax = float(np.nanmax(o))
        rec = {"sid": i, "tag": r["tag"], "scene": r["scene"], "split": L.SPLIT[r["scene"]], "set": r["set"], "start": r["start"], "t": r["t"],
               "n_legal": r["n_legal"], "oracle_max": omax, "oracle_mean": float(np.nanmean(o))}
        for n in names:
            p, fb = choose(sc[n], r["scene"], int(r["start"]), int(r["t"]))
            rec[f"pick_{n}"], rec[f"fb_{n}"] = p, fb
            rec[f"gain_{n}"] = float(o[p])
            rec[f"geo_{n}"] = float(r["geo1"][p]) if r.get("geo1") is not None else None      # secondary label: F geometric one-step gain (pp)
            rec[f"norm_{n}"] = float(o[p] / omax) if omax > 0 else None
            rec[f"top1_{n}"] = float(o[p] >= omax - 1e-12)
            rec[f"rho_{n}"] = sp(sc[n], o)
            rec[f"allzero_{n}"] = float(sc[n].max() == 0.0)
            rec[f"zero_frac_{n}"] = float((sc[n] == 0.0).mean())
            rec[f"cv_{n}"] = float(sc[n].std() / sc[n].mean()) if sc[n].mean() > 0 else None
            rec[f"ndist_{n}"] = int(len(np.unique(sc[n])))
        for n in ("U1", "U2", "U3", "D1"):
            rec[f"rhoC1_{n}"] = sp(sc[n], sc["C1"])
        nrc = [m["n_rc"] for m in r["u"]]
        for n in ("U1", "U2", "U3", "C1"):
            rec[f"rhoNrc_{n}"] = sp(sc[n], nrc)
        rec["rhoNcells_U3"] = sp(sc["U3"], [m["n_cells_union"] for m in r["u"]])
        rec["rhoNcells_U3sum"] = sp(sc["U3sum"], [m["n_cells_union"] for m in r["u"]])
        rec["rhoDist_U2"] = sp(sc["U2"], [m["u2_mean_hit_dist_rt"] or 0.0 for m in r["u"]])
        rec["unknown_frac_mean"] = float(np.mean(nrc) / r["u"][0]["n_rays"])
        # cross-view noise-index agreement (U4 rule)
        ag = r["agree"]
        for k_ in ("same_num", "same_den", "cross_num", "cross_den", "n_overlap"):
            rec[f"ag_{k_}"] = float(sum(a[k_] for a in ag))
        rec["n_paths"] = len(ag)
        st.append(rec)
        for j in range(r["n_legal"]):
            m = r["u"][j]
            g = (r.get("gt_attr") or [None] * r["n_legal"])[j] or {}
            c = {"sid": i, "scene": r["scene"], "t": r["t"], "j": j, "oracle_pp": float(o[j]), **{n: float(sc[n][j]) for n in names},
                 "n_rc": m["n_rc"], "n_r2": m["n_r2"], "n_cells_union": m["n_cells_union"], "n_cells_allk": m["n_cells_allk"],
                 "u2_mean_hit_dist_rt": m.get("u2_mean_hit_dist_rt"), "u2_cv": m.get("u2_cv"), "n_disagree_rays": m["n_disagree_rays"],
                 "n_anyhit_rays": m["n_anyhit_rays"], **{f"gt_{k}": v for k, v in g.items()}}
            cand.append(c)
    L.write_csv(os.path.join(OUT, "fixed_states.csv"), list(st[0].keys()), [list(x.values()) for x in st])
    keys = sorted({k for c in cand for k in c}, key=lambda k: (k.startswith("gt_"), k))
    L.write_csv(os.path.join(OUT, "fixed_candidates.csv"), keys, [[c.get(k) for k in keys] for c in cand])
    scenes = [x["scene"] for x in st]
    # ---------------- Table U3 (decision quality on the fixed-state pool)
    groups = {"All (10 scenes)": sorted(set(scenes)), "Dev-5": [s for s in L.GROUPS["Dev-5"] if s in scenes], "Train (5 of Train-6)": [s for s in L.GROUPS["Train-6"] if s in scenes]}
    tab = []
    summ = {"regression": reg, "n_states": len(st), "n_candidates": len(cand), "scenes": {s: scenes.count(s) for s in sorted(set(scenes))}}
    for gname, gs in groups.items():
        idx = [i for i, s in enumerate(scenes) if s in gs]
        sub = [st[i] for i in idx]
        ssub = [scenes[i] for i in idx]
        for n in names:
            row = {"group": gname, "scorer": n, "role": "planner" if n in PLANNER else "offline diagnostic", "n_states": len(sub)}
            for met in ("gain", "geo", "norm", "top1", "rho", "allzero", "zero_frac", "fb"):
                d = scene_mean([x[f"{met}_{n}"] if x[f"{met}_{n}"] is None else float(x[f"{met}_{n}"]) for x in sub], ssub)
                row[met] = se_mean(d)
            row["agree_pick_C1"] = se_mean(scene_mean([float(x[f"pick_{n}"] == x["pick_C1"]) for x in sub], ssub))
            row["agree_pick_D1"] = se_mean(scene_mean([float(x[f"pick_{n}"] == x["pick_D1"]) for x in sub], ssub))
            for ref in ("C1", "D1"):
                da = scene_mean([x[f"gain_{n}"] for x in sub], ssub)
                db = scene_mean([x[f"gain_{ref}"] for x in sub], ssub)
                m_, lo, hi, ns, w, l_ = paired_ci(da, db)
                row[f"dgain_vs_{ref}"], row[f"dgain_vs_{ref}_lo"], row[f"dgain_vs_{ref}_hi"], row[f"wins_vs_{ref}"], row[f"losses_vs_{ref}"] = m_, lo, hi, w, l_
            row["oracle_max_mean"] = se_mean(scene_mean([x["oracle_max"] for x in sub], ssub))
            row["random_gain"] = se_mean(scene_mean([x["oracle_mean"] for x in sub], ssub))
            tab.append(row)
    L.write_csv(os.path.join(OUT, "table_U3_decision_quality.csv"), list(tab[0].keys()), [list(x.values()) for x in tab])
    allrows = {x["scorer"]: x for x in tab if x["group"] == "All (10 scenes)"}
    # ---------------- U-best (pre-registered rule: highest scene-equal mean official one-step gain of the pick, all pool states)
    order = ["U1", "U2", "U3"]
    best = max(order, key=lambda n: (round(allrows[n]["gain"], 12), -order.index(n)))
    summ["u_best"] = {"rule": "argmax scene-equal mean official one-step gain (pp) of the choose() pick on the 755-state pool; exact tie -> U1 < U2 < U3",
                      "values": {n: allrows[n]["gain"] for n in order}, "U_best": best}
    # ---------------- U4 rule: same-index vs cross-index agreement on shared cells
    per = {}
    for x in st:
        d = per.setdefault(x["scene"], {"sn": 0.0, "sd": 0.0, "cn": 0.0, "cd": 0.0, "paths": 0, "ov": 0.0})
        d["sn"] += x["ag_same_num"]; d["sd"] += x["ag_same_den"]; d["cn"] += x["ag_cross_num"]; d["cd"] += x["ag_cross_den"]
        d["paths"] += x["n_paths"]; d["ov"] += x["ag_n_overlap"]
    sc_same = {s: d["sn"] / d["sd"] for s, d in per.items() if d["sd"] > 0}
    sc_cross = {s: d["cn"] / d["cd"] for s, d in per.items() if d["cd"] > 0}
    m_, lo, hi, ns, w, l_ = paired_ci(sc_same, sc_cross)
    same_se, cross_se = se_mean(sc_same), se_mean(sc_cross)
    ratio = same_se / cross_se if cross_se else None
    passed = bool(lo is not None and lo > 0 and ratio is not None and ratio >= 1.25)
    summ["u4_rule"] = {"same_index_agreement": same_se, "cross_index_agreement": cross_se, "diff": m_, "ci": [lo, hi], "ratio": ratio,
                       "scenes_same_gt_cross": w, "scenes": ns, "PASS": passed, "per_scene": {s: [sc_same.get(s), sc_cross.get(s)] for s in per},
                       "n_paths": int(sum(d["paths"] for d in per.values())), "pooled_overlap_cells": float(sum(d["ov"] for d in per.values()))}
    # ---------------- six questions: aggregates
    q = {}
    C = {k: np.asarray([c[k] if c[k] is not None else np.nan for c in cand], float) for k in ("U1", "U2", "U3", "C1", "D1", "n_rc", "n_cells_union", "oracle_pp", "u2_mean_hit_dist_rt", "U3sum", "U2rel")}
    for n in ("U1", "U2", "U3"):
        v = C[n]
        q[n] = {"cand_zero_frac": float((v == 0).mean()), "quantiles": [float(x) for x in np.nanpercentile(v, [5, 25, 50, 75, 95])],
                "state_allzero_frac": float(np.mean([x[f"allzero_{n}"] for x in st])),
                "within_state_cv_median": float(np.nanmedian([x[f"cv_{n}"] for x in st if x[f"cv_{n}"] is not None])),
                "pooled_rho_oracle": sp(v, C["oracle_pp"]), "pooled_rho_C1": sp(v, C["C1"]), "pooled_rho_nrc": sp(v, C["n_rc"]),
                "within_rho_oracle_se": allrows[n]["rho"], "within_rho_C1_se": se_mean(scene_mean([x[f"rhoC1_{n}"] for x in st], scenes)),
                "within_rho_nrc_se": se_mean(scene_mean([x[f"rhoNrc_{n}"] for x in st], scenes))}
    for n in ("U1", "U2", "U3"):                      # state level: mean disagreement of the state vs its UNKNOWN fraction / step
        mu_ = [float(np.mean([c[n] for c in cand if c["sid"] == x["sid"]])) for x in st]
        q[n]["state_rho_unknown_frac"] = sp(mu_, [x["unknown_frac_mean"] for x in st])
        q[n]["state_rho_t"] = sp(mu_, [x["t"] for x in st])
        q[n]["state_max_lt_0p01_frac"] = float(np.mean([max(c[n] for c in cand if c["sid"] == x["sid"]) < 0.01 for x in st]))
    q["C1"] = {"within_rho_nrc_se": se_mean(scene_mean([x["rhoNrc_C1"] for x in st], scenes)), "within_rho_oracle_se": allrows["C1"]["rho"]}
    q["U3"]["within_rho_ncells_se"] = se_mean(scene_mean([x["rhoNcells_U3"] for x in st], scenes))
    q["U3sum"] = {"within_rho_ncells_se": se_mean(scene_mean([x["rhoNcells_U3sum"] for x in st], scenes)), "pooled_rho_oracle": sp(C["U3sum"], C["oracle_pp"])}
    q["U2"]["within_rho_meandist_se"] = se_mean(scene_mean([x["rhoDist_U2"] for x in st], scenes))
    q["U2"]["pooled_rho_meandist"] = sp(C["U2"], C["u2_mean_hit_dist_rt"])
    q["U2rel"] = {"pooled_rho_meandist": sp(C["U2rel"], C["u2_mean_hit_dist_rt"]), "pooled_rho_U2": sp(C["U2rel"], C["U2"])}
    # Q3 GT attribution (offline)
    ga = [c for c in cand if c.get("gt_n_rc") is not None]
    if ga:
        tot1 = sum(c["U1"] * c["n_rc"] for c in ga)

        def share(k, tot):
            return float(sum(c.get(k) or 0.0 for c in ga) / tot) if tot else None
        q["gt_attr"] = {"n_candidates": len(ga),
                        "u1_mass_share": {g: share(f"gt_u1mass_{g}", tot1) for g in ("HIT", "NOHIT", "KNOWN", "OTHER")},
                        "u1_mass_singleton_share": share("gt_u1mass_singleton", tot1),
                        "ray_share_rc": {g: float(sum(c.get(f"gt_n_{g}") or 0 for c in ga) / max(1, sum(c["gt_n_rc"] for c in ga))) for g in ("HIT", "NOHIT", "KNOWN", "OTHER")},
                        "hit_precision_on_gthit_rays": float(sum(c.get("gt_n_hits_correct") or 0 for c in ga) / max(1, sum(c.get("gt_n_hits_on_gthit") or 0 for c in ga))),
                        "hits_on_gt_nohit_share": float(sum(c.get("gt_n_hits_on_gtnohit") or 0 for c in ga) / max(1, sum((c.get("gt_n_hits_on_gtnohit") or 0) + (c.get("gt_n_hits_on_gthit") or 0) for c in ga)))}
        tot2 = sum(sum(c.get(f"gt_u2mass_{g}") or 0.0 for g in ("HIT", "NOHIT", "KNOWN", "OTHER")) for c in ga)
        q["gt_attr"]["u2_mass_share"] = {g: share(f"gt_u2mass_{g}", tot2) for g in ("HIT", "NOHIT", "KNOWN", "OTHER")}
        rr = [c.get("gt_u2_rho_spread_err_gthit") for c in ga if c.get("gt_u2_rho_spread_err_gthit") is not None]
        q["gt_attr"]["u2_rho_spread_vs_error_median"] = float(np.median(rr)) if rr else None
        t3 = sum((c.get("gt_u3mass_true") or 0.0) + (c.get("gt_u3mass_phantom") or 0.0) for c in ga)
        q["gt_attr"]["u3_mass_true_share"] = share("gt_u3mass_true", t3)
        q["gt_attr"]["u3_cells_true_share"] = float(sum(c.get("gt_n_cells_true") or 0 for c in ga) / max(1, sum((c.get("gt_n_cells_true") or 0) + (c.get("gt_n_cells_phantom") or 0) for c in ga)))
    # Q4 / Q6: where do the U picks land relative to the oracle
    for n in ("U1", "U2", "U3", "C1", "D1"):
        lowgain = [x for x in st if x["oracle_max"] > 0 and x[f"norm_{n}"] is not None and x[f"norm_{n}"] <= 0.1]
        q.setdefault("picks", {})[n] = {"low_gain_pick_frac(<=10% of max)": len(lowgain) / len(st),
                                        "pick_nrc_frac_mean": float(np.mean([rows[x["sid"]]["u"][x[f"pick_{n}"]]["n_rc"] / rows[x["sid"]]["u"][0]["n_rays"] for x in st]))}
    cases = []
    for x in st:
        for n in ("U1", "U2", "U3"):
            if x["oracle_max"] > 0 and x[f"norm_{n}"] is not None and x[f"norm_{n}"] <= 0.1 and (x["norm_C1"] or 0) >= 0.9:
                cases.append({"scorer": n, "tag": x["tag"], "t": x["t"], "scene": x["scene"], "pick": x[f"pick_{n}"], "pick_C1": x["pick_C1"],
                              "gain_pick": x[f"gain_{n}"], "gain_C1": x["gain_C1"], "oracle_max": x["oracle_max"]})
    q["high_u_low_gain_cases_count"] = {n: sum(1 for c in cases if c["scorer"] == n) for n in ("U1", "U2", "U3")}
    L.write_csv(os.path.join(OUT, "high_u_low_gain_cases.csv"), list(cases[0].keys()) if cases else ["none"], [list(c.values()) for c in cases])
    summ["questions"] = q
    summ["table_U3_all"] = allrows
    json.dump(summ, open(os.path.join(OUT, "fixed_summary.json"), "w"), indent=1, default=float)
    figures(st, cand, allrows, tab)
    print(json.dumps({"regression": reg, "u_best": summ["u_best"], "u4": {k: v for k, v in summ["u4_rule"].items() if k != "per_scene"}}, indent=1, default=float))


def figures(st, cand, allrows, tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    # ---- candidate-ranking quality per scorer (planner scores only), All / Dev-5 / Train
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    gl = ["All (10 scenes)", "Dev-5", "Train (5 of Train-6)"]
    for ax, met, ttl in zip(axs, ("gain", "top1", "rho"), ("official gain of the pick (pp)", "Top-1 rate", "within-state Spearman vs gain")):
        w = 0.15
        for i, n in enumerate(PLANNER):
            vals = [next(x[met] for x in tab if x["group"] == g and x["scorer"] == n) for g in gl]
            vals = [np.nan if v is None else v for v in vals]
            ax.bar(np.arange(3) + (i - 2) * w, vals, w * 0.9, color=COL[n], label=n)
        if met == "gain":
            rv = [next(x["random_gain"] for x in tab if x["group"] == g and x["scorer"] == "D1") for g in gl]
            rv = [np.nan if v is None else v for v in rv]
            ax.scatter(np.arange(3), rv, marker="_", s=400, color="#555555", zorder=3, label="uniform random (expected)")
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(["All 755", "Dev-5", "Train"]); ax.set_title(ttl)
    axs[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Experiment U Phase D -- candidate ranking on the fixed-state pool (V3-T0, scene-equal)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "U_fixed_ranking.png"), dpi=160); plt.close(fig)
    # ---- Figure U3: coverage score vs disagreement vs GT gain (within-state normalised ranks)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    by = {}
    for c in cand:
        by.setdefault(c["sid"], []).append(c)

    def nr(v):
        v = np.asarray(v, float)
        if len(v) < 2:
            return np.zeros_like(v)
        from scipy.stats import rankdata
        return (rankdata(v) - 1) / (len(v) - 1)
    pairs = [("C1", "oracle_pp"), ("U1", "oracle_pp"), ("U2", "oracle_pp"), ("U3", "oracle_pp")]
    for ax, (a, b) in zip(axs, pairs):
        xs, ys = [], []
        for sid, cs in by.items():
            if len(cs) < 3:
                continue
            xs += list(nr([c[a] for c in cs])); ys += list(nr([c[b] for c in cs]))
        H, xe, ye = np.histogram2d(xs, ys, bins=6, range=[[0, 1], [0, 1]])
        ax.imshow(H.T / H.sum(), origin="lower", extent=[0, 1, 0, 1], cmap="Blues", vmin=0)
        ax.set_xlabel(f"within-state rank of {a}"); ax.set_ylabel("within-state rank of official gain")
        r = sp(xs, ys)
        ax.set_title(f"{a}: pooled rank rho {r:.2f}" if r is not None else a)
        ax.grid(False)
    fig.suptitle("Figure U3 -- predicted coverage (C1) and disagreement (U1-U3) vs GT official one-step gain (755 states)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "U3_coverage_vs_disagreement.png"), dpi=160); plt.close(fig)
    # ---- size dependence: U vs number of UNKNOWN rays (pooled candidates)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    nrc = np.asarray([c["n_rc"] for c in cand], float) / 7296.0
    for ax, n in zip(axs, ("U1", "U2", "U3")):
        v = np.asarray([c[n] for c in cand], float)
        ax.scatter(nrc, v, s=3, alpha=0.25, color=COL[n], edgecolors="none")
        ax.set_xlabel("fraction of rays in R_c (UNKNOWN)"); ax.set_ylabel(n)
        if n == "U2":
            ax.set_yscale("symlog", linthresh=1.0)
        r = sp(nrc, v)
        ax.set_title(f"{n} vs UNKNOWN size: rho {r:.2f}" if r is not None else n)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "U_fixed_size_dependence.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
