"""EXP5 Task 3 / E1 -- analysis (seen2scene env, CPU): closed loop B0 vs B1 (protocol §3-§4), B0 reproduction of Task 2,
behaviour / cost, fixed-state geometry and ranking (B0-B3), figures. Output analysis/, reports tables, figures.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T3 = os.path.dirname(HERE)
EXP5 = os.path.dirname(T3)
sys.path.insert(0, os.path.join(EXP5, "ablation_study", "common"))
import ab_common as A  # noqa: E402
import ab_load as AL  # noqa: E402
import ab_style as ST  # noqa: E402
import t13_analyze as T13  # noqa: E402

import numpy as np  # noqa: E402

plt = ST.plt
W3 = os.path.join(A.REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1")
T2E = os.path.join(A.T2W, "eval")
OUT = os.path.join(T3, "analysis")
FIG = os.path.join(T3, "analysis", "figures")
GRP = ("B0", "B1", "B2")
COL = {"B0": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a"}
LAB = {"B0": "B0 Sparse-P1", "B1": "B1 FullDepth-P1", "B2": "B2 Dense-input / Sparse-scoring"}
PAIRS = (("B1", "B0"), ("B2", "B0"), ("B2", "B1"))
TAU = A.TIE_TOL * 100


def boot(vals, n=10000, seed=0):
    g = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    bs = g[rng.integers(0, len(g), (n, len(g)))].mean(1)
    return {"mean": float(g.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "n": int(len(g))}


# ---------------------------------------------------------------------------------------------------- closed loop
def indoor_episodes():
    rows = []
    for ep in A.episodes("indoor"):
        r = {"dataset": "indoor", "scene": ep["scene"], "start": ep["start"], "split": "mp3d_val"}
        for g in GRP:
            d = os.path.join(W3, "closed_loop", "indoor", g, "V2M", ep["tag"])
            e = json.load(open(os.path.join(d, "episode.json")))
            pl = [json.loads(x) for x in open(os.path.join(d, "planner.jsonl"))]
            c = np.asarray(e["coverage"])
            ex = [tuple(p) for p in e["executed"]]
            pos = [p[:3] for p in ex]
            r[g] = {"auc": float(np.trapz(c, dx=1) / e["moves"]), "final": float(c[-1]), "curve": c.tolist(),
                    **{f"cov{x}": float(c[x]) for x in (10, 20, 50, 100)},
                    "revisit_pose": float(np.mean([ex[i] in set(ex[:i]) for i in range(1, len(ex))])),
                    "revisit_position": float(np.mean([pos[i] in set(pos[:i]) for i in range(1, len(pos))])),
                    "translations": int(sum(pos[i] != pos[i - 1] for i in range(1, len(pos)))),
                    "n_cand": float(np.mean([s["n_legal"] for s in e["steps"]])),
                    "time_per_move": float(np.mean([x["timing"]["decision_s"] for x in pl])),
                    "peak_gb": float(max(x.get("peak_mem_gb", 0) for x in pl)), "executed": [list(p) for p in ex]}
        ref = json.load(open(os.path.join(T2E, "indoor", "closed_loop", "V2M", ep["tag"], "episode.json")))
        r["B0_equals_task2"] = bool(ref["coverage"] == r["B0"]["curve"] and ref["executed"] == r["B0"]["executed"])
        rows.append(r)
    return rows


def outdoor_episodes():
    rows = []
    for ep in A.episodes("outdoor"):
        s, k = ep["scene"], ep["start"]
        tg = f"V2M_{s}_s{k}_r0"
        r = {"dataset": "outdoor", "scene": s, "start": k, "split": ep["split"]}
        for g in GRP:
            root = os.path.join(W3, "closed_loop", "outdoor", g, "V2M")
            e = json.load(open(os.path.join(root, "episodes", f"ep_{tg}.json")))
            m = json.load(open(os.path.join(W3, "metrics", "outdoor", g, "V2M", f"{tg}.json")))
            pl = json.load(open(os.path.join(root, "planner_logs", f"pl_{tg}.json")))
            c = np.asarray(m["cov_mono"])
            keys = e["executed_keys"]
            r[g] = {"auc": float(m["auc_mono"]), "final": float(m["cov_mono_final"]), "curve": c.tolist(),
                    "auc_official": float(m["auc_official"]), "final_official": float(m["cov_official_final"]),
                    **{f"cov{x}": float(c[x]) for x in (10, 20, 50)},
                    "revisit_pose": float(np.mean([keys[i] in set(keys[:i]) for i in range(1, len(keys))])),
                    "hover": int(e.get("n_hover", 0)), "n_cand": float(np.mean([st["n_legal"] for st in e["steps"]])),
                    "time_per_move": float(np.mean([st["decision_s"] for st in pl["steps"]])),
                    "peak_gb": float(pl.get("peak_mem_gb_this_process", 0)), "keys": keys, "cov_official": e["coverage"]}
        ref = json.load(open(os.path.join(T2E, "outdoor", "closed_loop", "V2M", "episodes", f"ep_{tg}.json")))
        r["B0_equals_task2"] = bool(ref["coverage"] == r["B0"]["cov_official"] and ref["executed_keys"] == r["B0"]["keys"])
        rows.append(r)
    return rows


def summarize(rows, dataset, hi="B1", lo="B0"):
    keys = ["auc", "final"] + [k for k in rows[0]["B0"] if k.startswith("cov") and isinstance(rows[0]["B0"][k], float)] + ["revisit_pose", "n_cand", "time_per_move", "peak_gb"]
    if dataset == "outdoor":
        keys += ["auc_official", "final_official", "hover"]
    else:
        keys += ["revisit_position", "translations"]
    scenes = sorted({r["scene"] for r in rows}, key=lambda s: (A.SPLIT_OF.get(s, ""), s))
    S = {"n_episodes": len(rows), "B0_equals_task2": sum(r["B0_equals_task2"] for r in rows), "means": {}, "paired": {}, "per_scene": {}}
    for k in keys:
        for g in GRP:
            S["means"].setdefault(k, {})[g] = boot([np.mean([r[g][k] for r in rows if r["scene"] == s]) for s in scenes])
        d = [np.mean([r[hi][k] - r[lo][k] for r in rows if r["scene"] == s]) for s in scenes]
        S["paired"][k] = dict(boot(d), per_scene=dict(zip(scenes, map(float, d))), wins=int(sum(x > 0 for x in d)),
                              losses=int(sum(x < 0 for x in d)),
                              rel=float(np.mean(d) / S["means"][k][lo]["mean"]) if S["means"][k][lo]["mean"] else None)
        if dataset == "outdoor":
            for sp in ("test", "dev"):
                ss = [s for s in scenes if A.SPLIT_OF[s] == sp]
                dd = [np.mean([r[hi][k] - r[lo][k] for r in rows if r["scene"] == s]) for s in ss]
                S["paired"][k][f"mean@{sp}"] = float(np.mean(dd))
                S["paired"][k][f"wins@{sp}"] = int(sum(x > 0 for x in dd))
    for s in scenes:
        S["per_scene"][s] = {g: {k: float(np.mean([r[g][k] for r in rows if r["scene"] == s])) for k in ("auc", "final")} for g in GRP}
    p = S["paired"]
    n_sc = len(scenes)
    need = 4
    meaningful = (p["auc"]["mean"] >= 0.010 and p["final"]["mean"] >= 0.010 and p["auc"]["wins"] >= need
                  and min(p["auc"]["per_scene"].values()) >= -0.030)
    degraded = p["auc"]["mean"] <= -0.010 or p["final"]["mean"] <= -0.010
    S["criteria"] = {"meaningful": bool(meaningful), "degraded": bool(degraded), "wins_needed": f"{need}/{n_sc}",
                     # protocol §4: "AUC 有意义地改善" = the AUC criteria 1, 3, 4 (incl. no scene < -0.030), Final not met
                     "auc_meaningful_only": bool(p["auc"]["mean"] >= 0.010 and p["auc"]["wins"] >= need
                                                 and min(p["auc"]["per_scene"].values()) >= -0.030 and not meaningful)}
    return S


def decision(S, cost_ok):
    mi, mo = S["indoor"]["criteria"]["meaningful"], S["outdoor"]["criteria"]["meaningful"]
    di, do = S["indoor"]["criteria"]["degraded"], S["outdoor"]["criteria"]["degraded"]
    if mi and mo and cost_ok:
        return "Go"
    if di or do or not (mi or mo or S["indoor"]["criteria"]["auc_meaningful_only"] or S["outdoor"]["criteria"]["auc_meaningful_only"]):
        return "No-Go for direct replacement"
    return "Conditional Go"


def curves(rows_in, rows_out):
    fig, axs = plt.subplots(2, 7, figsize=(22, 7.5), gridspec_kw={"width_ratios": [2.2, 1, 1, 1, 1, 1, 1]})
    for i, (rows, nm, xmax) in enumerate(((rows_in, "MP3D (C_obs)", 150), (rows_out, "Macarons++ (monotone coverage)", 100))):
        ax = axs[i, 0]
        for g in GRP:
            c = np.mean([r[g]["curve"] for r in rows], 0)
            ax.plot(c, color=COL[g], lw=2.2, label=f"{LAB[g]}  AUC {np.mean([r[g]['auc'] for r in rows]):.3f} · final {c[-1]:.3f}")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("move")
        ax.set_ylabel("coverage")
        ax.legend(loc="lower right", fontsize=8.5)
        ST.clean(ax, "both")
        ax.set_title(f"{nm}: mean over {len(rows)} episodes", loc="left")
        scenes = sorted({r["scene"] for r in rows}, key=lambda s: (A.SPLIT_OF.get(s, ""), s))
        for j, s in enumerate(scenes[:6]):
            ax = axs[i, 1 + j]
            for g in GRP:
                ax.plot(np.mean([r[g]["curve"] for r in rows if r["scene"] == s], 0), color=COL[g], lw=1.6)
            ax.set_xlim(0, xmax)
            ax.set_ylim(0, 1)
            ST.clean(ax, "both")
            ax.set_title(s.replace("_L", " L") + (f" ({A.SPLIT_OF[s]})" if s in A.SPLIT_OF else ""), loc="left", fontsize=9)
        for j in range(len(scenes), 6):
            axs[i, 1 + j].axis("off")
    fig.suptitle("Task 3 / E1: frozen V2-Mix with Sparse-P1 (B0), FullDepth-P1 (B1) and dense input / sparse scoring (B2); Task 2 protocol",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "coverage_curves.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- fixed states
def fixed_states(dataset):
    rows = [json.loads(x) for p in sorted(os.listdir(os.path.join(W3, "geometry", dataset)))
            for x in open(os.path.join(W3, "geometry", dataset, p))]
    seen, uniq = set(), []
    for r in rows:
        if r["sid"] not in seen:
            seen.add(r["sid"])
            uniq.append(r)
    man = {s["sid"]: s for s in A.load_manifest(dataset)["states"]}
    rk = []
    for r in uniq:
        st = man[r["sid"]]
        o = np.load(os.path.join(A.state_dir(dataset, r["sid"]), "oracle.npz"))
        rec = {k: np.asarray(x) for k, x in np.load(os.path.join(A.REPO, next(e["svc"] for e in st["sources"] if e["source"] == "V2M"))).items()
               if k in ("legal_idx", "executed_pose_idx_history")}
        C = np.asarray(st["cand"])
        g = o["gain"][C] * 100
        row = {"sid": r["sid"], "scene": r["scene"], "start": r["start"], "src": "V2M", "t": r["t"]}
        for b in ("B0", "B1", "B2", "B3"):
            s = np.asarray(r["scores"][b])
            pk = AL.pick_of(dataset, st, s, rec, "V2M")[0]
            loc = {int(c): i for i, c in enumerate(C)}
            gp = g[loc[pk]]
            row[f"{b}_top1"] = float(gp >= g.max() - TAU)
            row[f"{b}_regret"] = float(g.max() - gp)
            row[f"{b}_spearman"] = T13.spearman(s[C], g)
        for m, nm in (("s", "sparse"), ("f", "full")):
            for k in ("t_known_valid", "pseudo_front", "match", "n_seg_mean", "lu_mean", "med_abs_err"):
                row[f"{nm}_cur_{k}"] = r[m]["cur"][k]
                row[f"{nm}_cand_{k}"] = r[m]["cand"][k]
            for k in ("band", "empty", "unknown", "band_over_carved", "coarse_band_mean", "coarse_band_cells", "coarse_empty_mean"):
                row[f"{nm}_{k}"] = r[m][k]
            row[f"{nm}_fusion_step_s"] = r["timing_fusion"][m]["step_s"]
            row[f"{nm}_rebuild_s"] = r["timing_fusion"][m]["rebuild_s"]
            row[f"{nm}_score_s"] = r["score_timing"][m]["total_s"]
        row["repro_b0"] = r["repro_b0"]
        rk.append(row)
    keys = [k for k in rk[0] if k not in ("sid", "scene", "start", "src", "t", "repro_b0")]
    S = {"n_states": len(rk), "repro_b0_all": all(x["repro_b0"] for x in rk)}
    for k in keys:
        hm = AL.hier_mean([dict(x, src="V2M") for x in rk], k)
        S[k] = AL.boot(hm) if hm else None
    for b in ("B1", "B2", "B3"):
        for met in ("top1", "regret", "spearman"):
            a = AL.hier_mean(rk, f"{b}_{met}")
            z = AL.hier_mean(rk, f"B0_{met}")
            S[f"paired_{b}_minus_B0_{met}"] = AL.boot({s: a[s] - z[s] for s in a if s in z})
    return S, rk


def fmt(b, nd=3):
    if b is None or b.get("mean") is None:
        return "—"
    if b.get("n", b.get("scenes", 0)) < 4:
        return f"{b['mean']:.{nd}f}"
    return f"{b['mean']:.{nd}f} [{b['ci95'][0]:.{nd}f}, {b['ci95'][1]:.{nd}f}]"


def main():
    os.makedirs(FIG, exist_ok=True)
    ST.setup()
    rows_in, rows_out = indoor_episodes(), outdoor_episodes()
    S = {"indoor": summarize(rows_in, "indoor"), "outdoor": summarize(rows_out, "outdoor")}
    t_ratio = max(S[d]["means"]["time_per_move"]["B1"]["mean"] / S[d]["means"]["time_per_move"]["B0"]["mean"] for d in S)
    m_ratio = max(S[d]["means"]["peak_gb"]["B1"]["mean"] / max(1e-9, S[d]["means"]["peak_gb"]["B0"]["mean"]) for d in S)
    S["cost"] = {"time_ratio_max": t_ratio, "vram_ratio_max": m_ratio, "ok": bool(t_ratio <= 2.0 and m_ratio <= 1.2)}
    S["decision"] = decision(S, S["cost"]["ok"])
    S["pairs"] = {}
    for hi, lo in PAIRS:
        P = {"indoor": summarize(rows_in, "indoor", hi, lo), "outdoor": summarize(rows_out, "outdoor", hi, lo)}
        tr = max(P[d]["means"]["time_per_move"][hi]["mean"] / P[d]["means"]["time_per_move"][lo]["mean"] for d in ("indoor", "outdoor"))
        mr = max(P[d]["means"]["peak_gb"][hi]["mean"] / max(1e-9, P[d]["means"]["peak_gb"][lo]["mean"]) for d in ("indoor", "outdoor"))
        P["cost"] = {"time_ratio_max": tr, "vram_ratio_max": mr, "ok": bool(tr <= 2.0 and mr <= 1.2)}
        P["decision"] = decision(P, P["cost"]["ok"])
        S["pairs"][f"{hi}-{lo}"] = {"indoor": {k: P["indoor"][k] for k in ("paired", "criteria")},
                                    "outdoor": {k: P["outdoor"][k] for k in ("paired", "criteria")},
                                    "cost": P["cost"], "decision": P["decision"]}
    for d in ("indoor", "outdoor"):
        try:
            S[f"fixed_{d}"], rk = fixed_states(d)
            json.dump(rk, open(os.path.join(OUT, f"fixed_states_{d}.json"), "w"))
        except (FileNotFoundError, StopIteration, KeyError) as e:
            S[f"fixed_{d}"] = {"error": repr(e)}
    json.dump({"summary": S, "indoor": rows_in, "outdoor": rows_out}, open(os.path.join(OUT, "t34_closed_loop.json"), "w"), default=float)
    curves(rows_in, rows_out)
    # tables
    L = ["### Table 1 — 闭环（场景均值的平均，方括号为场景 bootstrap 95% CI）", "",
         "| 数据集 | 方法 | Mean AUC ↑ | Mean Final Coverage ↑ | Mean Time/Move (s) ↓ | Peak VRAM (GB) ↓ |", "|---|---|---|---|---|---|"]
    for d, nm in (("indoor", "MP3D（AUC@150）"), ("outdoor", "Macarons++（AUC@100，单调覆盖）")):
        m = S[d]["means"]
        for g in GRP:
            L.append(f"| {nm} | {LAB[g]} | {fmt(m['auc'][g])} | {fmt(m['final'][g])} | {m['time_per_move'][g]['mean']:.2f} | {m['peak_gb'][g]['mean']:.1f} |")
        for pair in ("B2-B0", "B2-B1"):
            p = S["pairs"][pair][d]["paired"]
            L.append(f"| {nm} | **{pair.replace('-', ' − ')}** | {fmt(p['auc'])}（相对 {100 * p['auc']['rel']:+.1f}%；胜 {p['auc']['wins']} 负 {p['auc']['losses']}） | "
                     f"{fmt(p['final'])}（相对 {100 * p['final']['rel']:+.1f}%；胜 {p['final']['wins']} 负 {p['final']['losses']}） | "
                     f"{p['time_per_move']['mean']:+.2f} | {p['peak_gb']['mean']:+.1f} |")
        p = S["paired"] if False else S[d]["paired"]
        L.append(f"| {nm} | **B1 − B0** | {fmt(p['auc'])}（相对 {100 * p['auc']['rel']:+.1f}%；胜 {p['auc']['wins']} 负 {p['auc']['losses']}） | "
                 f"{fmt(p['final'])}（相对 {100 * p['final']['rel']:+.1f}%；胜 {p['final']['wins']} 负 {p['final']['losses']}） | "
                 f"{p['time_per_move']['mean']:+.2f} | {p['peak_gb']['mean']:+.1f} |")
    L += ["", "### Table 2 — 逐 episode", "", "| 数据集 | 场景 | 起点 | B0 AUC | B1 AUC | B2 AUC | ΔAUC B1−B0 | ΔAUC B2−B0 | B0 Final | B1 Final | B2 Final | ΔFinal B1−B0 | ΔFinal B2−B0 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rows, d in ((rows_in, "MP3D"), (rows_out, "Macarons++")):
        for r in rows:
            L.append(f"| {d} | {r['scene']}{' (' + r['split'] + ')' if d != 'MP3D' else ''} | {r['start']} | {r['B0']['auc']:.3f} | {r['B1']['auc']:.3f} | {r['B2']['auc']:.3f} | "
                     f"{r['B1']['auc'] - r['B0']['auc']:+.3f} | {r['B2']['auc'] - r['B0']['auc']:+.3f} | {r['B0']['final']:.3f} | {r['B1']['final']:.3f} | {r['B2']['final']:.3f} | "
                     f"{r['B1']['final'] - r['B0']['final']:+.3f} | {r['B2']['final'] - r['B0']['final']:+.3f} |")
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps({d: {"auc": S[d]["paired"]["auc"], "final": S[d]["paired"]["final"], "criteria": S[d]["criteria"],
                          "B0_equals_task2": S[d]["B0_equals_task2"]} for d in ("indoor", "outdoor")}, indent=1, default=float))
    print("cost", S["cost"], "decision", S["decision"])
    for k, v in S["pairs"].items():
        print(k, {d: (round(v[d]["paired"]["auc"]["mean"], 4), [round(c, 4) for c in v[d]["paired"]["auc"]["ci95"]], v[d]["paired"]["auc"]["wins"],
                      round(v[d]["paired"]["final"]["mean"], 4), [round(c, 4) for c in v[d]["paired"]["final"]["ci95"]], v[d]["paired"]["final"]["wins"])
                  for d in ("indoor", "outdoor")}, v["cost"], v["decision"])


if __name__ == "__main__":
    main()
