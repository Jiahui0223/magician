"""EXP5 Task 1.4 analysis (CPU, evaluation side; protocol §5.5): per arm Cov@100, AUC@100, Cov@10/20/50, curves, the
15-start table (mean and range of the 3 noise replicates), replicate variance; paired differences vs B0, MAGICIAN and
between V2 arms (replicates averaged within a start first; bootstrap over the 15 starts, 10,000, seed 0) and scene-level
means; one-step diagnostics along the executed trajectories (oracle_diag), view distance, cost and integrity checks.
Output W/analysis/t14.json + figures.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as LC  # noqa: E402

import numpy as np  # noqa: E402

OUT = os.path.join(LC.WORK, "analysis")
ABL = os.path.join(LC.REPO, "work_dirs", "exp4_multiworld_belief", "ablation_study")
T7A = os.path.join(LC.REPO, "work_dirs", "exp3b_macaronspp_transfer", "task7a_baseline_v2", "attempt2")
KEYS = [(s, k) for s in LC.SCENES for k in LC.STARTS]
AT = (10, 20, 50)


def metrics(c):
    c = np.asarray(c, float)
    return {"cov_final": float(c[100]), "auc": float(np.trapz(c, dx=1.0) / 100), **{f"cov{t}": float(c[t]) for t in AT},
            "curve": c.tolist()}


def boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "n_pos": int((d > 0).sum()), "n": int(len(d))}


def baselines():
    out = {}
    pe = json.load(open(os.path.join(T7A, "analysis", "t7a_results.json")))["per_episode"]
    out["MAGICIAN"] = {x: metrics(pe["magician"][f"{x[0]}/{x[1]}"]["curve_mean"]) for x in KEYS}
    groups = {"B0": [os.path.join(ABL, "abl_1", "T1C_FRindep_DistP1_Greedy_R10_K4", "dev100", "episodes", "ep_A2_{s}_s{k}_r0.json")],
              "B1": [os.path.join(ABL, "abl_2", "Frozen_FRindep_DistP1_Greedy_R10_K4", "dev100", "episodes", "ep_B1_{s}_s{k}_r0.json")],
              "B2": [os.path.join(ABL, "abl_2", "T1C_FRindep_DistP1_Greedy_R10_K1", f"chain_k{c}", "dev100", "episodes",
                                  f"ep_B2k{c}" + "_{s}_s{k}_r0.json") for c in range(4)]}
    for g, pats in groups.items():
        out[g] = {}
        for s, k in KEYS:
            cs = [json.load(open(p.format(s=s, k=k)))["coverage"] for p in pats if os.path.exists(p.format(s=s, k=k))]
            out[g][(s, k)] = metrics(np.mean(cs, 0)) if cs else None
    return out


def view_dists(svc_dir, s, k):
    """Median camera -> observed sample point distance of every executed view (abl_1 abl_analyze definition)."""
    out = []
    for t in range(100):
        p = os.path.join(svc_dir, f"svc_{s}_s{k}_t{t}.npz")
        if os.path.exists(p):
            z = np.load(p)
            if len(z["sample_points"]) >= 20:
                out.append(float(np.median(np.linalg.norm(z["sample_points"] - z["cam_center"][None], axis=1))))
    return out


BASE_ROOTS = {"B0": [(os.path.join(ABL, "abl_1", "T1C_FRindep_DistP1_Greedy_R10_K4", "dev100"), "A2")],
              "B2": [(os.path.join(ABL, "abl_2", "T1C_FRindep_DistP1_Greedy_R10_K1", f"chain_k{c}", "dev100"), f"B2k{c}")
                     for c in range(4)]}


def base_cost(g):
    """Archived B0 (abl_1 A2) / B2 (abl_2, 4 chains) episodes: wall seconds, GPU-h, world-cache hits (protocol §4.7,
    reported separately from the V2 arms)."""
    rows, vd = [], []
    for (root, pre), (s, k) in [(r, x) for r in BASE_ROOTS[g] for x in KEYS]:
        tg = f"{pre}_{s}_s{k}_r0"
        if not os.path.exists(os.path.join(root, "episodes", f"ep_{tg}.json")):
            continue
        e = json.load(open(os.path.join(root, "episodes", f"ep_{tg}.json")))
        pl = json.load(open(os.path.join(root, "planner_logs", f"pl_{tg}.json")))
        ap = os.path.join(root, "acct", f"{tg}.json")
        ac = json.load(open(ap)) if os.path.exists(ap) else None
        refr = [r for r in pl["refreshes"]]
        rows.append({"tag": tg, "wall_s": e["seconds_this_process"], "gpu_h": ac["gpu_hours"] if ac else None,
                     "host": e["host"], "n_refresh": len(refr), "cache_hits": sum(1 for r in refr if "cache_entry" in r),
                     "generation_s_total": float(sum(r.get("generation_s", 0.0) for r in refr)),
                     "decision_s_total": pl.get("decision_s_total"), "peak_mem_gb": pl.get("peak_mem_gb_this_process")})
        vd += view_dists(os.path.join(root, "svc", tg), s, k)
    return {"episodes": rows, "wall_s_mean": float(np.mean([r["wall_s"] for r in rows])),
            "gpu_h_total": float(sum(r["gpu_h"] or 0 for r in rows)),
            "cache_hit_refreshes": int(sum(r["cache_hits"] for r in rows)), "refreshes": int(sum(r["n_refresh"] for r in rows)),
            "generation_s_total": float(sum(r["generation_s_total"] for r in rows)),
            "view_dist_median": float(np.median(vd)), "view_dist_lt10": float(np.mean(np.asarray(vd) < 10)), "view_dists": vd}


def arm(eid):
    root = LC.exp_root(eid)
    per, reps, diag, cost, integ = {}, {}, [], [], {"replay_equal": [], "legal": [], "guard_denied": 0}
    vdist = []
    for s, k in KEYS:
        cs = []
        for r in LC.REPS:
            tg = LC.tag(eid, s, k, r)
            p = os.path.join(root, "episodes", f"ep_{tg}.json")
            if not os.path.exists(p):
                continue
            e = json.load(open(p))
            cs.append(e["coverage"])
            reps.setdefault((s, k), []).append(metrics(e["coverage"]))
            rp = os.path.join(root, "replay", f"rep_{tg}.json")
            if os.path.exists(rp):
                x = json.load(open(rp))
                integ["replay_equal"].append(bool(x.get("bitwise_equal")))
                integ["legal"].append(bool(x.get("all_legal")))
            pl = os.path.join(root, "planner_logs", f"pl_{tg}.json")
            vdist += view_dists(os.path.join(root, "svc", tg), s, k)
            if os.path.exists(pl):
                q = json.load(open(pl))
                integ["guard_denied"] += len(q["guard"]["denied"])
                ac = os.path.join(LC.LOOP, "acct", f"{tg}.json")
                att = json.load(open(ac))["attempts"] if os.path.exists(ac) else []
                cost.append({"tag": tg, "wall_s": e["seconds_this_process"], "decision_s_total": q["decision_s_total"],
                             "attempts": len(att), "gpu_h": sum(a["end"] - a["start"] for a in att) / 3600.0,
                             "resume_overhead_s": sum(a["end"] - a["start"] for a in att[:-1]),
                             "peak_mem_gb": q["peak_mem_gb_this_process"],
                             "decision_s_mean": float(np.mean([x["decision_s"] for x in q["steps"]])),
                             "sampling_s_mean": float(np.mean([x["timing"].get("sampling_s", 0) for x in q["steps"] if x["timing"]])),
                             "projection_s_mean": float(np.mean([x["timing"].get("projection_s", 0) for x in q["steps"] if x["timing"]])),
                             "encode_s_mean": float(np.mean([r_["encode_s"] for r_ in q["refreshes"]])) if q["refreshes"] else None,
                             "fallbacks": sum(1 for x in q["steps"] if x["choice"].get("fallback"))})
            dp = os.path.join(root, "oracle_diag", f"diag_{tg}.json")
            if os.path.exists(dp):
                st = json.load(open(dp))["steps"]
                diag.append({"tag": tg, "spearman": float(np.nanmean([x["spearman"] if x.get("spearman") is not None else np.nan for x in st])),
                             "regret": float(np.mean([x["regret"] for x in st if x.get("regret") is not None])),
                             "top1": float(np.mean([x["pick_is_oracle"] for x in st if "pick_is_oracle" in x])),
                             "dead_share": float(np.mean([x["g_max"] < 0.1 for x in st if "g_max" in x])),
                             "false_promise": float(np.mean([x["score_pick"] > 0 and x["g_pick"] < 0.01 for x in st
                                                             if x.get("score_pick") is not None]))})
        if cs:
            per[(s, k)] = metrics(np.mean(cs, 0))
    integ["view_dists"] = vdist
    return per, reps, diag, cost, integ


def paired(a, b, key):
    ks = [x for x in KEYS if a.get(x) and b.get(x)]
    if not ks:
        return None
    d = [a[x][key] - b[x][key] for x in ks]
    out = boot(d)
    out["by_scene"] = {s: float(np.mean([a[x][key] - b[x][key] for x in ks if x[0] == s])) for s in LC.SCENES}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    B = baselines()
    import hashlib
    ins = [os.path.join(T7A, "analysis", "t7a_results.json")]
    for root, pre in BASE_ROOTS["B0"] + BASE_ROOTS["B2"] + [(os.path.join(ABL, "abl_2", "Frozen_FRindep_DistP1_Greedy_R10_K4", "dev100"), "B1")]:
        ins += [os.path.join(root, "episodes", f"ep_{pre}_{s}_s{k}_r0.json") for s, k in KEYS]
    base_sha = {os.path.relpath(p, LC.REPO): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in ins if os.path.exists(p)}
    res = {"baselines": {g: {f"{x[0]}/{x[1]}": v for x, v in d.items()} for g, d in B.items()}, "arms": {},
           "baseline_inputs_sha256": base_sha,
           "B0_cost": base_cost("B0"), "B2_cost": base_cost("B2")}
    arms = {}
    for e in LC.RUN_ORDER:
        per, reps, diag, cost, integ = arm(e)
        if not per:
            continue
        arms[e] = per
        summ = {m: float(np.mean([v[m] for v in per.values()])) for m in ("cov_final", "auc", "cov10", "cov20", "cov50")}
        rep_sd = {m: float(np.mean([np.std([r[m] for r in rs]) for rs in reps.values() if len(rs) > 1] or [np.nan]))
                  for m in ("cov_final", "auc")}
        res["arms"][e] = {"name": LC.ARMS[e]["name"], "n_starts": len(per), "n_episodes": sum(len(v) for v in reps.values()),
                          "summary": summ, "mean_within_start_sd": rep_sd,
                          "per_start": {f"{x[0]}/{x[1]}": {"mean": {m: per[x][m] for m in ("cov_final", "auc", "cov10", "cov20", "cov50")},
                                                            "range_auc": [min(r["auc"] for r in reps[x]), max(r["auc"] for r in reps[x])],
                                                            "range_cov100": [min(r["cov_final"] for r in reps[x]), max(r["cov_final"] for r in reps[x])]}
                                        for x in per},
                          "curve_mean": np.mean([per[x]["curve"] for x in per], 0).tolist(),
                          "diag": {m: float(np.mean([d[m] for d in diag])) for m in ("spearman", "regret", "top1", "dead_share", "false_promise")} if diag else None,
                          "cost": {"episodes": len(cost), "wall_s_mean": float(np.mean([c["wall_s"] for c in cost])) if cost else None,
                                   "decision_s_mean": float(np.mean([c["decision_s_mean"] for c in cost])) if cost else None,
                                   "sampling_s_mean": float(np.mean([c["sampling_s_mean"] for c in cost])) if cost else None,
                                   "projection_s_mean": float(np.mean([c["projection_s_mean"] for c in cost])) if cost else None,
                                   "peak_mem_gb_max": float(max(c["peak_mem_gb"] for c in cost)) if cost else None,
                                   "fallbacks_mean": float(np.mean([c["fallbacks"] for c in cost])) if cost else None,
                                   "gpu_h_total": float(sum(c["gpu_h"] for c in cost)) if cost else None,
                                   "episodes_resumed": int(sum(c["attempts"] > 1 for c in cost)),
                                   "resume_overhead_s_total": float(sum(c["resume_overhead_s"] for c in cost))},
                          "view_dist_median": float(np.median(integ["view_dists"])) if integ["view_dists"] else None,
                          "view_dist_lt10": float(np.mean(np.asarray(integ["view_dists"]) < 10)) if integ["view_dists"] else None,
                          "integrity": {"replays": len(integ["replay_equal"]), "replay_equal_all": all(integ["replay_equal"]),
                                        "legal_all": all(integ["legal"]), "guard_denied": integ["guard_denied"]}}
        for g in ("B0", "B2", "B1", "MAGICIAN"):
            res["arms"][e][f"vs_{g}"] = {m: paired(per, B[g], m) for m in ("cov_final", "auc", "cov20", "cov50")}
        # protocol §5.3: colosseum s2 has no BAND at t = 0 (B0 / B2 can only fall back there) -> also reported without it
        excl = {x: v for x, v in per.items() if x != ("colosseum", 2)}
        for g in ("B0", "B2"):
            res["arms"][e][f"vs_{g}_excl_colosseum_s2"] = {m: paired(excl, B[g], m) for m in ("cov_final", "auc")}
    pairs = [("E2", "E1"), ("E3", "E1"), ("E4", "E1"), ("E5", "E1"), ("E6", "E1"), ("E4", "E3"), ("E5", "E4")]
    res["arm_pairs"] = {f"{a}-{b}": {m: paired(arms[a], arms[b], m) for m in ("cov_final", "auc", "cov20", "cov50")}
                        for a, b in pairs if a in arms and b in arms}
    for g in ("B0", "B2", "B1", "MAGICIAN"):
        vals = [v for v in B[g].values() if v]
        res["baselines_summary"] = res.get("baselines_summary", {})
        res["baselines_summary"][g] = {m: float(np.mean([v[m] for v in vals])) for m in ("cov_final", "auc", "cov10", "cov20", "cov50")} if vals else None
    json.dump(res, open(os.path.join(OUT, "t14.json"), "w"), indent=1, default=float)
    for e, r in res["arms"].items():
        print(e, r["summary"], "vs B0 AUC", r["vs_B0"]["auc"]["mean"] if r["vs_B0"]["auc"] else None)
    figures(res)


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for g in ("B0", "MAGICIAN"):
        vals = [v["curve"] for v in res["baselines"][g].values() if v]
        if vals:
            ax[0].plot(np.mean(vals, 0), "k--" if g == "B0" else "k:", label=g)
    for e, r in res["arms"].items():
        ax[0].plot(r["curve_mean"], label=f"{e} {r['name']}")
    ax[0].set_xlabel("move")
    ax[0].set_ylabel("coverage (Dev-3 mean of 15 starts)")
    ax[0].legend(fontsize=7)
    names = list(res["arms"])
    d = [res["arms"][e]["vs_B0"]["auc"] for e in names]
    ax[1].errorbar(range(len(names)), [x["mean"] for x in d], yerr=[[x["mean"] - x["ci95"][0] for x in d], [x["ci95"][1] - x["mean"] for x in d]], fmt="o")
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("AUC@100 - B0 (paired, 95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "t14_curves.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
