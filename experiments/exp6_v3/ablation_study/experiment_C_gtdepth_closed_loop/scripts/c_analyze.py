"""EXP6 V3 Ablation Experiment C -- closed-loop analysis (seen2scene env, CPU; protocol §4).

Groups: MAG-G, MAG-B (Task 2 shared-protocol closed loops), B0, B2, E2D (Task 3 / E2), T0, T1 (EXP6 V3) -- historical
G3 results read in place -- and C1, C2 (this experiment). Readers / paired scene bootstrap / "meaningful - degraded"
criteria = task3_full_depth_p1/scripts/t34_analyze.py (as analyze_g3.py); MP3D stall = e2_g3_analyze.stall.
Adds: per-step diagnostics of C1 / C2 (one-step oracle regret inside their own closed loops, candidate-space diagnostics
in stall windows), execution cost, automatically picked failure cases.
Output results/{c_summary.json, c_episodes.csv, c_steps_{indoor,outdoor}.csv}, figures/*.png and the report fragments
report/{C_GTDEPTH_G3.md, C_ORACLE_GAIN_G3.md, C_FAILURE_CASES.md} (C_FINAL_REPORT.md is written by hand from these).
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_common as CC  # noqa: E402

sys.path.insert(0, os.path.join(CC.EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(CC.EXP5, "task3_full_depth_p1", "e2_pilot", "scripts"))
import t34_analyze as T  # noqa: E402
import e2_g3_analyze as G3  # noqa: E402

GRP = ("MAG-G", "MAG-B", "B0", "B2", "E2D", "T0", "T1", "C1", "C2")
T.GRP = GRP
HIST = ("MAG-G", "MAG-B", "B0", "B2", "E2D", "T0", "T1")
LAB = {"MAG-G": "MAGICIAN Greedy (Task 2)", "MAG-B": "MAGICIAN Beam (Task 2)", "B0": "B0 frozen V2-Mix, sparse", "B2": "B2 frozen V2-Mix, D100",
       "E2D": "E2-D100 fine-tuned", "T0": "V3-VAE (T0)", "T1": "V3-VAE-XAttn (T1)", "C1": "C1 GT depth + DistP1", "C2": "C2 GT gain greedy"}
COL = {"MAG-G": "#bdbdbd", "MAG-B": "#7f7f7f", "B0": "#86b6ef", "B2": "#1c5cab", "E2D": "#eb6834", "T0": "#1baf7a", "T1": "#8e44ad", "C1": "#d62728", "C2": "#111111"}
PAIRS = (("C1", "B2"), ("C1", "T1"), ("C1", "E2D"), ("C2", "C1"), ("C2", "B2"), ("C2", "T1"), ("C1", "MAG-B"), ("C2", "MAG-B"), ("C1", "T0"), ("C1", "B0"), ("C2", "E2D"))
OUT, FIG, REP = (os.path.join(CC.EXPC, k) for k in ("results", "figures", "report"))
EPS_HIST = {g: (os.path.join(CC.W6E, "closed_loop"), os.path.join(CC.W6E, "metrics", "outdoor"), "V2M") for g in ("B0", "B2", "E2D", "T0", "T1")}
EPS_HIST.update({g: (os.path.join(CC.T2E, "{d}", "closed_loop"), os.path.join(CC.T2E, "outdoor", "metrics"), g) for g in ("MAG-G", "MAG-B")})
plt = T.plt


def indoor_dir(g, tag):
    if g in CC.GROUPS:
        return CC.indoor_dir(g, tag)
    root = EPS_HIST[g][0].replace("{d}", "indoor")
    return os.path.join(root, g, tag) if g.startswith("MAG") else os.path.join(root, "indoor", g, "V2M", tag)


def outdoor_paths(g, s, k):
    if g in CC.GROUPS:
        tg = CC.tag_out(g, s, k)
        return CC.outdoor_root(g), tg, os.path.join(CC.outdoor_metrics_dir(g), f"{tg}.json")
    root, mroot, pre = EPS_HIST[g]
    root = root.replace("{d}", "outdoor")
    base = os.path.join(root, g) if g.startswith("MAG") else os.path.join(root, "outdoor", g, "V2M")
    tg = f"{pre}_{s}_s{k}_r0"
    return base, tg, os.path.join(mroot, g, "V2M" if not g.startswith("MAG") else "", f"{tg}.json").replace("//", "/")


def _row_indoor(d, g):
    e = json.load(open(os.path.join(d, "episode.json")))
    plp = os.path.join(d, "planner.jsonl")
    pl = [json.loads(x) for x in open(plp)] if os.path.exists(plp) else []
    c = np.asarray(e["coverage"])
    ex = [tuple(p) for p in e["executed"]]
    pos = [p[:3] for p in ex]
    r = {"auc": float(np.trapz(c, dx=1) / e["moves"]), "final": float(c[-1]), "curve": c.tolist(),
         **{f"cov{x}": (float(c[x]) if x < len(c) else float("nan")) for x in (10, 20, 50, 100)},
         "revisit_pose": float(np.mean([ex[i] in set(ex[:i]) for i in range(1, len(ex))])),
         "revisit_position": float(np.mean([pos[i] in set(pos[:i]) for i in range(1, len(pos))])),
         "translations": int(sum(pos[i] != pos[i - 1] for i in range(1, len(pos)))),
         "n_cand": float(np.mean([s["n_legal"] for s in e["steps"]])),
         "time_per_move": float(np.mean([x["timing"]["decision_s"] for x in pl])) if pl else float(np.mean([s.get("decision_s", 0) for s in e["steps"]])),
         "peak_gb": float(max(x.get("peak_mem_gb", 0) for x in pl)) if pl else 0.0, "executed": [list(p) for p in ex],
         "fallbacks": int(sum(bool(s.get("fallback")) for s in e["steps"])), "wall_s": float(e.get("wall_s", 0))}
    if g in CC.GROUPS:
        st = e["steps"]
        r["diag"] = [dict(s["diag"], t=s["t"], n_legal=s["n_legal"], pick=s["pick"], fallback=s["fallback"], cov=s["cov"]) for s in st if "diag" in s]
        r["gt_query_s"] = float(np.mean([s["gt_query_s"] for s in st]))
        r["diag_s"] = float(np.mean([s["diag"]["diag_s"] for s in st if "diag" in s])) if r["diag"] else 0.0
        r["checks"] = e.get("checks", {})
        r["cq_reads"] = pl[-1].get("cq_reads") if pl else None
    return r


def _row_outdoor(g, s, k):
    base, tg, mp = outdoor_paths(g, s, k)
    e = json.load(open(os.path.join(base, "episodes", f"ep_{tg}.json")))
    m = json.load(open(mp))
    plp = os.path.join(base, "planner_logs", f"pl_{tg}.json")
    pl = json.load(open(plp)) if os.path.exists(plp) else None
    c = np.asarray(m["cov_mono"])
    keys = e["executed_keys"]
    r = {"auc": float(m["auc_mono"]), "final": float(m["cov_mono_final"]), "curve": c.tolist(),
         "auc_official": float(m["auc_official"]), "final_official": float(m["cov_official_final"]),
         **{f"cov{x}": (float(c[x]) if x < len(c) else float("nan")) for x in (10, 20, 50)},
         "revisit_pose": float(np.mean([keys[i] in set(keys[:i]) for i in range(1, len(keys))])),
         "hover": int(e.get("n_hover", 0)), "n_cand": float(np.mean([st["n_legal"] for st in e["steps"]])),
         "time_per_move": float(np.mean([st["decision_s"] for st in pl["steps"]])) if pl else float(np.mean([st.get("decision_s", 0) for st in e["steps"]])),
         "peak_gb": float(pl.get("peak_mem_gb_this_process", 0)) if pl else 0.0, "keys": keys, "cov_official": e["coverage"],
         "official_max_drop": float(m.get("official_max_drop", 0)), "wall_s": float(e.get("seconds_this_process", 0)),
         "fallbacks": int(sum(1 for st in e["steps"][:-1] if (st.get("planner") or {}).get("fallback"))) if pl else None}
    if g in CC.GROUPS:
        st = e["steps"][:-1]
        r["diag"] = [dict(x["planner"]["c_diag"], t=x["t"], pick=x["pick"], fallback=x["planner"].get("fallback"), cov=x["cov"]) for x in st]
        r["gt_query_s"] = float(np.mean([x["planner"]["c_diag"]["gt_query_s"] for x in st]))
        r["diag_s"] = float(np.mean([x["planner"]["c_diag"].get("diag_s", 0) for x in st]))
        r["checks"] = (e.get("planner_bye") or {}).get("checks", {})
        r["cq_reads"] = (e.get("planner_bye") or {}).get("cq_reads")
    return r


def indoor_episodes(groups=GRP):
    rows = []
    for ep in T.A.episodes("indoor"):
        r = {"dataset": "indoor", "scene": ep["scene"], "start": ep["start"], "split": "mp3d_val", "tag": ep["tag"], "B0_equals_task2": True}
        for g in groups:
            r[g] = _row_indoor(indoor_dir(g, ep["tag"]), g)
        rows.append(r)
    return rows


def outdoor_episodes(groups=GRP):
    rows = []
    for ep in T.A.episodes("outdoor"):
        r = {"dataset": "outdoor", "scene": ep["scene"], "start": ep["start"], "split": ep["split"], "tag": ep["tag"], "B0_equals_task2": True}
        for g in groups:
            r[g] = _row_outdoor(g, ep["scene"], ep["start"])
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------- diagnostics
def regret_curve(rows, g, n):
    """Mean one-step regret (pp) of the executed pick over episodes at every step (C groups only)."""
    M = np.full((len(rows), n), np.nan)
    for i, r in enumerate(rows):
        for d in r[g]["diag"]:
            if d.get("regret") is not None:
                M[i, d["t"]] = 100 * d["regret"]
    return np.nanmean(M, 0), M


def diag_summary(rows, g, dataset, window=None):
    """Aggregate of the per-step diagnostics over all steps (or the steps inside the MP3D stall windows)."""
    vals = {k: [] for k in ("n_legal", "gstar", "regret", "top1", "remaining", "share", "gstar_blocked", "fallback", "gap_blocked")}
    for r in rows:
        steps = r[g]["diag"]
        if window is not None:
            lo, hi = window[r["tag"]]
            steps = [d for d in steps if lo <= d["t"] <= hi]
        for d in steps:
            if d.get("gstar") is None:
                continue
            vals["n_legal"].append(d["n_legal"] if "n_legal" in d else d["n_cand"])
            vals["gstar"].append(100 * d["gstar"])
            vals["regret"].append(100 * d["regret"] if d.get("regret") is not None else np.nan)
            vals["top1"].append(float(d["top1"]) if d.get("top1") is not None else np.nan)
            vals["remaining"].append(100 * d["remaining"] if d.get("remaining") is not None else np.nan)
            vals["share"].append(d.get("remaining_in_region_share", d.get("remaining_reachable_share")))
            vals["fallback"].append(float(bool(d.get("fallback"))))
            if dataset == "indoor":
                gb = d.get("gstar_blocked")
                vals["gstar_blocked"].append(100 * gb if gb is not None else np.nan)
                vals["gap_blocked"].append(float(gb is not None and gb > d["gstar"] + CC.TIE_TOL))
    out = {"n_steps": len(vals["gstar"])}
    for k, v in vals.items():
        v = np.asarray([x for x in v if x is not None], float)
        if len(v):
            out[k] = {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v))}
    return out


def stall_windows(rows, g, lo=30, hi=120, eps=0.01):
    """Longest stall window [s, e] of every episode (e2_g3_analyze.stall rule) -> {tag: (s, e, length)}."""
    out = {}
    for r in rows:
        c = np.asarray(r[g]["curve"])
        best, bs, be = 0, lo, lo
        for s in range(lo, hi):
            e = s
            while e + 1 <= hi and c[e + 1] - c[s] < eps:
                e += 1
            if e - s > best:
                best, bs, be = e - s, s, e
        out[r["tag"]] = (bs, be, best)
    return out


# ------------------------------------------------------------------------------------------------- tables
def f(b, nd=3):
    return T.fmt(b, nd)


def main_table(res, d, nm):
    M = res["means"][d]
    cov_keys = ("cov10", "cov20", "cov50", "cov100") if d == "indoor" else ("cov10", "cov20", "cov50")
    L = ["| 组 | AUC | Final | " + " | ".join(k.replace("cov", "Cov@") for k in cov_keys) + " | 重访率 | 每步时间 s* | 峰值显存 GB |", "|---|---|---|" + "---|" * len(cov_keys) + "---|---|---|"]
    for g in GRP:
        L.append(f"| {LAB[g]} | {f(M['auc'][g])} | {f(M['final'][g])} | " + " | ".join(f(M[k][g]) for k in cov_keys) +
                 f" | {M['revisit_pose'][g]['mean']:.3f} | {M['time_per_move'][g]['mean']:.2f} | {M['peak_gb'][g]['mean']:.1f} |")
    return "\n".join(L)


def pair_table(res, d):
    L = ["| 配对 | ΔAUC [CI] | 胜/负 | ΔFinal [CI] | 胜/负 | ΔCov@20 [CI] | 有意义 | 退化 | 逐场景 ΔAUC |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["pairs"].items():
        p, c = v[d]["paired"], v[d]["criteria"]
        L.append(f"| {k} | {f(p['auc'])} | {p['auc']['wins']}/{p['auc']['losses']} | {f(p['final'])} | {p['final']['wins']}/{p['final']['losses']} | {f(p['cov20'])} | "
                 f"{'✅' if c['meaningful'] else '—'} | {'❌' if c['degraded'] else '—'} | " + "，".join(f"{s} {x:+.3f}" for s, x in p["auc"]["per_scene"].items()) + " |")
    return "\n".join(L)


def per_scene_table(rows, d):
    scenes = sorted({r["scene"] for r in rows}, key=lambda s: (T.A.SPLIT_OF.get(s, ""), s))
    L = ["| 场景 | " + " | ".join(GRP) + " |", "|---|" + "---|" * len(GRP)]
    for s in scenes:
        sub = [r for r in rows if r["scene"] == s]
        L.append(f"| {s}{' (' + T.A.SPLIT_OF[s] + ')' if s in T.A.SPLIT_OF else ''} | " + " | ".join(f"{np.mean([r[g]['auc'] for r in sub]):.3f} / {np.mean([r[g]['final'] for r in sub]):.3f}" for g in GRP) + " |")
    return "\n".join(L)


def episode_table(rows, d):
    L = ["| 场景 | 起点 | " + " | ".join(f"{g} AUC" for g in GRP) + " | " + " | ".join(f"{g} Final" for g in ("B2", "T1", "C1", "C2")) + " |", "|---|---|" + "---|" * (len(GRP) + 4)]
    for r in rows:
        L.append(f"| {r['scene']} | {r['start']} | " + " | ".join(f"{r[g]['auc']:.3f}" for g in GRP) + " | " + " | ".join(f"{r[g]['final']:.3f}" for g in ("B2", "T1", "C1", "C2")) + " |")
    return "\n".join(L)


def curves(rows_in, rows_out, groups, name, title):
    fig, axs = plt.subplots(2, 7, figsize=(24, 8), gridspec_kw={"width_ratios": [2.2, 1, 1, 1, 1, 1, 1]})
    for i, (rows, nm, xmax) in enumerate(((rows_in, "MP3D (C_obs)", 150), (rows_out, "Macarons++ (monotone coverage)", 100))):
        ax = axs[i, 0]
        for g in groups:
            c = np.mean([r[g]["curve"] for r in rows], 0)
            ax.plot(c, color=COL[g], lw=2.4 if g in CC.GROUPS else 1.6, ls=":" if g.startswith("MAG") else ("--" if g in ("B0", "B2", "E2D") else "-"),
                    label=f"{LAB[g]}  AUC {np.mean([r[g]['auc'] for r in rows]):.3f} · final {c[-1]:.3f}")
        ax.set_xlim(0, xmax); ax.set_xlabel("move"); ax.set_ylabel("coverage"); ax.legend(loc="lower right", fontsize=7.5)
        T.ST.clean(ax, "both"); ax.set_title(f"{nm}: mean over {len(rows)} episodes", loc="left")
        scenes = sorted({r["scene"] for r in rows}, key=lambda s: (T.A.SPLIT_OF.get(s, ""), s))
        for j, s in enumerate(scenes[:6]):
            ax = axs[i, 1 + j]
            for g in groups:
                ax.plot(np.mean([r[g]["curve"] for r in rows if r["scene"] == s], 0), color=COL[g], lw=1.8 if g in CC.GROUPS else 1.1,
                        ls=":" if g.startswith("MAG") else ("--" if g in ("B0", "B2", "E2D") else "-"))
            ax.set_xlim(0, xmax); ax.set_ylim(0, 1); T.ST.clean(ax, "both")
            ax.set_title(s.replace("_L", " L") + (f" ({T.A.SPLIT_OF[s]})" if s in T.A.SPLIT_OF else ""), loc="left", fontsize=9)
        for j in range(len(scenes), 6):
            axs[i, 1 + j].axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=130)
    plt.close(fig)


def regret_fig(rows_in, rows_out):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, rows, n, nm in ((axs[0], rows_in, 150, "MP3D"), (axs[1], rows_out, 100, "Macarons++")):
        for g in CC.GROUPS:
            m, _ = regret_curve(rows, g, n)
            ax.plot(np.convolve(np.nan_to_num(m), np.ones(5) / 5, mode="same"), color=COL[g], lw=1.8, label=f"{LAB[g]}  mean {np.nanmean(m):.3f} pp")
        ax.set_xlabel("move"); ax.set_ylabel("one-step oracle regret of the executed pick (pp, 5-step mean)"); ax.set_title(nm + ": regret inside the closed loop", loc="left")
        ax.legend(fontsize=8); T.ST.clean(ax, "both")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "c_regret_curves.png"), dpi=130); plt.close(fig)


def failure_cases(rows_in, rows_out, res):
    L = []
    for d, rows, nm in (("indoor", rows_in, "MP3D"), ("outdoor", rows_out, "Macarons++")):
        L += [f"## {nm}", ""]
        worst = sorted(rows, key=lambda r: r["C1"]["auc"] - r["B2"]["auc"])[:3]
        L += ["### C1 相对 B2 最差的 3 条 episode（ΔAUC 最小）", "", "| 场景 | 起点 | B2 AUC / Final | C1 AUC / Final | C2 AUC / Final | C1 重访 | C1 fallback | C1 平均 regret pp | C1 Top-1 率 |", "|---|---|---|---|---|---|---|---|---|"]
        for r in worst:
            dg = [x for x in r["C1"]["diag"] if x.get("regret") is not None]
            L.append(f"| {r['scene']} | {r['start']} | {r['B2']['auc']:.3f} / {r['B2']['final']:.3f} | {r['C1']['auc']:.3f} / {r['C1']['final']:.3f} | {r['C2']['auc']:.3f} / {r['C2']['final']:.3f} | "
                     f"{r['C1']['revisit_pose']:.2f} | {r['C1']['fallbacks']} | {100 * np.mean([x['regret'] for x in dg]):.3f} | {np.mean([x['top1'] for x in dg]):.2f} |")
        L += ["", "### C1 单步 regret 最大的 5 个状态（闭环内）", "", "| 场景 | 起点 | t | 候选数 | G\\* pp | 被执行增益 pp | regret pp | 覆盖 | 剩余 pp |", "|---|---|---|---|---|---|---|---|---|"]
        steps = [(r, x) for r in rows for x in r["C1"]["diag"] if x.get("regret") is not None]
        for r, x in sorted(steps, key=lambda z: -z[1]["regret"])[:5]:
            L.append(f"| {r['scene']} | {r['start']} | {x['t']} | {x.get('n_legal', x.get('n_cand'))} | {100 * x['gstar']:.3f} | {100 * x['gain_pick']:.3f} | {100 * x['regret']:.3f} | {x['cov']:.3f} | {100 * x['remaining']:.1f} |")
        if d == "indoor":
            L += ["", "### 停滞最长的 3 条 episode（C1 / C2 各自；e2_g3 规则）", "", "| 组 | 场景 | 起点 | 窗口 | 长度 | 窗口内候选数 | 窗口内 G\\* pp（中位） | 窗口内剩余 pp | 剩余在区域内比例 | 阻挡邻居 G\\* > 合法 G\\* 的步数比例 |", "|---|---|---|---|---|---|---|---|---|---|"]
            for g in CC.GROUPS:
                sw = stall_windows(rows, g)
                for r in sorted(rows, key=lambda r: -sw[r["tag"]][2])[:3]:
                    s0, e0, ln = sw[r["tag"]]
                    ds = diag_summary([r], g, d, {r["tag"]: (s0, e0)})
                    if ln == 0 or ds["n_steps"] == 0:
                        continue
                    L.append(f"| {g} | {r['scene']} | {r['start']} | [{s0}, {e0}] | {ln} | {ds['n_legal']['mean']:.1f} | {ds['gstar']['median']:.3f} | {ds['remaining']['mean']:.1f} | "
                             f"{ds['share']['mean']:.2f} | {ds['gap_blocked']['mean']:.2f} |")
        L.append("")
    return "\n".join(L)


def stage_c1():
    """Interim C1-only summary (C2 not finished): paired tables vs the historical groups -> results/stage_c1_summary.json + stdout."""
    global GRP
    GRP = HIST + ("C1",)
    T.GRP = GRP
    rows_in, rows_out = indoor_episodes(GRP), outdoor_episodes(GRP)
    pairs = [(a, b) for a, b in PAIRS if "C2" not in (a, b)]
    P = {f"{hi}-{lo}": {"indoor": T.summarize(rows_in, "indoor", hi, lo), "outdoor": T.summarize(rows_out, "outdoor", hi, lo)} for hi, lo in pairs}
    res = {"pairs": {k: {d: {"paired": v[d]["paired"], "criteria": v[d]["criteria"]} for d in ("indoor", "outdoor")} for k, v in P.items()},
           "means": {d: P["C1-B2"][d]["means"] for d in ("indoor", "outdoor")}, "per_scene": {d: P["C1-B2"][d]["per_scene"] for d in ("indoor", "outdoor")}}
    for r in rows_in:
        for g in GRP:
            r[g]["stall"] = G3.stall(r[g]["curve"])
    res["stall_mp3d"] = {g: int(sum(r[g]["stall"] >= 30 for r in rows_in)) for g in GRP}
    res["diag"] = {d: diag_summary(rows, "C1", d) for d, rows in (("indoor", rows_in), ("outdoor", rows_out))}
    res["test_dev"] = {g: {sp: {"auc": float(np.mean([r[g]["auc"] for r in rows_out if r["split"] == sp])), "final": float(np.mean([r[g]["final"] for r in rows_out if r["split"] == sp]))} for sp in ("test", "dev")} for g in GRP}
    res["cost"] = {d: {"wall_s_mean": float(np.mean([r["C1"]["wall_s"] for r in rows])), "wall_s_total": float(np.sum([r["C1"]["wall_s"] for r in rows])), "decision_s": float(np.mean([r["C1"]["time_per_move"] for r in rows])),
                       "gt_query_s": float(np.mean([r["C1"]["gt_query_s"] for r in rows])), "diag_s": float(np.mean([r["C1"]["diag_s"] for r in rows])), "peak_gb": float(max(r["C1"]["peak_gb"] for r in rows)),
                       "fallbacks": int(sum(r["C1"]["fallbacks"] or 0 for r in rows)), "checks": {k: int(sum(r["C1"]["checks"].get(k, 0) for r in rows)) for k in rows[0]["C1"]["checks"]}} for d, rows in (("indoor", rows_in), ("outdoor", rows_out))}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "stage_c1_summary.json"), "w"), indent=1, default=float)
    for d, nm in (("indoor", "MP3D"), ("outdoor", "Macarons++")):
        print(f"== {nm}"); print(main_table(res, d, nm)); print(pair_table(res, d)); print(per_scene_table(rows_in if d == "indoor" else rows_out, d))
    print("stalls", res["stall_mp3d"]); print("diag", json.dumps(res["diag"], default=float)); print("test/dev", json.dumps(res["test_dev"]["C1"]), json.dumps(res["test_dev"]["B2"]), json.dumps(res["test_dev"]["T1"]))
    print("cost", json.dumps(res["cost"], default=float))


def main():
    if "--stage-c1" in sys.argv:
        T.ST.setup()
        return stage_c1()
    for p in (OUT, FIG, REP):
        os.makedirs(p, exist_ok=True)
    T.ST.setup()
    rows_in, rows_out = indoor_episodes(), outdoor_episodes()
    P = {f"{hi}-{lo}": {"indoor": T.summarize(rows_in, "indoor", hi, lo), "outdoor": T.summarize(rows_out, "outdoor", hi, lo)} for hi, lo in PAIRS}
    res = {"groups": list(GRP), "pairs": {k: {d: {"paired": v[d]["paired"], "criteria": v[d]["criteria"]} for d in ("indoor", "outdoor")} for k, v in P.items()},
           "means": {d: P["C1-B2"][d]["means"] for d in ("indoor", "outdoor")}, "per_scene": {d: P["C1-B2"][d]["per_scene"] for d in ("indoor", "outdoor")}}
    for r in rows_in:
        for g in GRP:
            r[g]["stall"] = G3.stall(r[g]["curve"])
    res["stall_mp3d"] = {g: {"episodes_with_stall": int(sum(r[g]["stall"] >= 30 for r in rows_in)), "mean_longest_window": float(np.mean([r[g]["stall"] for r in rows_in]))} for g in GRP}
    res["diag"] = {d: {g: diag_summary(rows, g, d) for g in CC.GROUPS} for d, rows in (("indoor", rows_in), ("outdoor", rows_out))}
    res["diag_stall"] = {g: diag_summary(rows_in, g, "indoor", {t: (w[0], w[1]) for t, w in stall_windows(rows_in, g).items()}) for g in CC.GROUPS}
    res["regret_curve"] = {d: {g: regret_curve(rows, g, 150 if d == "indoor" else 100)[0].tolist() for g in CC.GROUPS} for d, rows in (("indoor", rows_in), ("outdoor", rows_out))}
    res["test_dev"] = {g: {sp: {"auc": float(np.mean([r[g]["auc"] for r in rows_out if r["split"] == sp])), "final": float(np.mean([r[g]["final"] for r in rows_out if r["split"] == sp])),
                                "cov20": float(np.mean([r[g]["cov20"] for r in rows_out if r["split"] == sp]))} for sp in ("test", "dev")} for g in GRP}
    res["checks"] = {d: {g: {k: int(sum(r[g]["checks"].get(k, 0) for r in rows)) for k in ("n", "state_hash_equal", "executed_sample_equal", "pick_in_candidates", "executed_official_equal", "executed_mono_equal", "executed_gain_equal")}
                         for g in CC.GROUPS} for d, rows in (("indoor", rows_in), ("outdoor", rows_out))}
    res["cost"] = {d: {g: {"wall_s_mean": float(np.mean([r[g]["wall_s"] for r in rows])), "wall_s_total": float(np.sum([r[g]["wall_s"] for r in rows])),
                           "decision_s": float(np.mean([r[g]["time_per_move"] for r in rows])), "gt_query_s": float(np.mean([r[g]["gt_query_s"] for r in rows])),
                           "diag_s": float(np.mean([r[g]["diag_s"] for r in rows])), "peak_gb": float(max(r[g]["peak_gb"] for r in rows)),
                           "fallbacks": int(sum(r[g]["fallbacks"] or 0 for r in rows))} for g in CC.GROUPS} for d, rows in (("indoor", rows_in), ("outdoor", rows_out))}
    json.dump(res, open(os.path.join(OUT, "c_summary.json"), "w"), indent=1, default=float)
    with open(os.path.join(OUT, "c_episodes.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "scene", "start", "split"] + [f"{g}_{k}" for g in GRP for k in ("auc", "final", "cov20")])
        for rows in (rows_in, rows_out):
            for r in rows:
                w.writerow([r["dataset"], r["scene"], r["start"], r["split"]] + [r[g][k] for g in GRP for k in ("auc", "final", "cov20")])
    for d, rows in (("indoor", rows_in), ("outdoor", rows_out)):
        with open(os.path.join(OUT, f"c_steps_{d}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["group", "scene", "start", "t", "n_legal", "pick", "fallback", "cov", "gstar", "gain_pick", "regret", "top1", "remaining", "share", "gstar_blocked"])
            for g in CC.GROUPS:
                for r in rows:
                    for x in r[g]["diag"]:
                        w.writerow([g, r["scene"], r["start"], x["t"], x.get("n_legal", x.get("n_cand")), x["pick"], x.get("fallback"), x["cov"], x.get("gstar"), x.get("gain_pick"),
                                    x.get("regret"), x.get("top1"), x.get("remaining"), x.get("remaining_in_region_share", x.get("remaining_reachable_share")), x.get("gstar_blocked")])
    # ---- figures
    curves(rows_in, rows_out, GRP, "c_coverage_curves_all.png", "Experiment C: GT-depth (C1) and GT-gain (C2) closed loops vs the historical G3 groups (same protocol); measured")
    curves(rows_in, rows_out, ("B2", "T1", "E2D", "C1", "C2"), "c_coverage_curves_main.png", "Experiment C: C1 / C2 vs B2 / T1 / E2-D100; measured")
    regret_fig(rows_in, rows_out)
    # ---- report fragments
    hdr = ["所有均值 = 场景（MP3D 楼层）均值的平均；[CI] = 场景级 bootstrap 95%（10,000 次，seed 0）；配对差按 (scene, start) 配对后按场景聚合；判定 = Task 3 §4（有意义 ΔAUC ≥ +0.010 且 ΔFinal ≥ +0.010 且胜 ≥ 4 场景且无场景 < −0.030；退化 ΔAUC ≤ −0.010 或 ΔFinal ≤ −0.010）。\\* 每步时间为各自运行时的共享负载记录，不可直接比较。历史组 = 正式 G3 结果（不重跑）。Test-3 非盲。", ""]
    L = ["# C1 — GT Depth + DistP1 的完整闭环（自动生成 `scripts/c_analyze.py`）", ""] + hdr
    for d, nm in (("indoor", "MP3D（150 步，C_obs）"), ("outdoor", "Macarons++（100 步，单调覆盖）")):
        L += [f"## {nm}", "", main_table(res, d, nm), "", "### 配对差", "", pair_table(res, d), "", "### 逐场景 AUC / Final", "", per_scene_table(rows_in if d == "indoor" else rows_out, d), "",
              "### 逐 episode", "", episode_table(rows_in if d == "indoor" else rows_out, d), ""]
    L += ["## Macarons++ Test-3 / Dev-3（各 3 场景，不给 CI）", "", "| 组 | Test-3 AUC / Final / Cov@20 | Dev-3 AUC / Final / Cov@20 |", "|---|---|---|"]
    for g in GRP:
        t, v = res["test_dev"][g]["test"], res["test_dev"][g]["dev"]
        L.append(f"| {LAB[g]} | {t['auc']:.3f} / {t['final']:.3f} / {t['cov20']:.3f} | {v['auc']:.3f} / {v['final']:.3f} / {v['cov20']:.3f} |")
    L += ["", "## MP3D 中期停滞（t ∈ [30, 120] 内 ≥ 30 步覆盖增长 < 0.01 的 episode 数 / 10；平均最长窗口）", "", "| 组 | 停滞 episode | 平均最长窗口（步） |", "|---|---|---|"]
    for g in GRP:
        L.append(f"| {LAB[g]} | {res['stall_mp3d'][g]['episodes_with_stall']} | {res['stall_mp3d'][g]['mean_longest_window']:.1f} |")
    L += ["", "## 闭环内单步 oracle regret（评估侧诊断；与 Experiment A 的固定状态 regret 分表，不换算）", "",
          "| 数据集 | 组 | 步数 | 平均候选数 | G\\* pp（均值 / 中位） | regret pp（均值 / 中位） | Top-1 率 | 剩余 pp（均值） | 区域内 / 可达 剩余比例 | fallback 率 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d in ("indoor", "outdoor"):
        for g in CC.GROUPS:
            x = res["diag"][d][g]
            L.append(f"| {d} | {g} | {x['n_steps']} | {x['n_legal']['mean']:.1f} | {x['gstar']['mean']:.3f} / {x['gstar']['median']:.3f} | {x['regret']['mean']:.3f} / {x['regret']['median']:.3f} | "
                     f"{x['top1']['mean']:.3f} | {x['remaining']['mean']:.1f} | {x['share']['mean']:.2f} | {x['fallback']['mean']:.3f} |")
    L += ["", "MP3D 停滞窗口内（各组自己的最长窗口）：", "", "| 组 | 窗口内步数 | 候选数 | G\\* pp（中位） | regret pp | 剩余 pp | 剩余在区域内比例 | 阻挡邻居 G\\* pp（中位） | 阻挡邻居优于合法集的步数比例 |", "|---|---|---|---|---|---|---|---|---|"]
    for g in CC.GROUPS:
        x = res["diag_stall"][g]
        if x["n_steps"]:
            L.append(f"| {g} | {x['n_steps']} | {x['n_legal']['mean']:.1f} | {x['gstar']['median']:.3f} | {x['regret']['mean']:.3f} | {x['remaining']['mean']:.1f} | {x['share']['mean']:.2f} | "
                     f"{x['gstar_blocked']['median']:.3f} | {x['gap_blocked']['mean']:.2f} |")
    L += ["", "## 执行成本与完整性检查", "", "| 数据集 | 组 | episode 均墙钟 s | 合计 s | 子进程决策 s/步 | GT 查询 s/步 | 诊断 s/步 | 子进程峰值显存 GB | fallback 次数 | 完整性检查（全部步数 = 通过数） |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d in ("indoor", "outdoor"):
        for g in CC.GROUPS:
            c, k = res["cost"][d][g], res["checks"][d][g]
            L.append(f"| {d} | {g} | {c['wall_s_mean']:.0f} | {c['wall_s_total']:.0f} | {c['decision_s']:.2f} | {c['gt_query_s']:.2f} | {c['diag_s']:.2f} | {c['peak_gb']:.1f} | {c['fallbacks']} | {k} |")
    L += ["", "图：`figures/c_coverage_curves_all.png`、`c_coverage_curves_main.png`、`c_regret_curves.png`。"]
    open(os.path.join(REP, "C_GTDEPTH_G3.md"), "w").write("\n".join(L) + "\n")
    # C2 fragment: the same tables restricted to the oracle-gain comparison
    L2 = ["# C2 — GT Gain Greedy 的完整闭环（自动生成 `scripts/c_analyze.py`）", "", "C2 = 候选集单步 oracle greedy（真实单步增益 argmax，并列最小索引），与 C1 共用模拟器 / 候选 / 选择规则，只有分数不同。**不是**轨迹最优，不保证 Final ≥ C1。", ""] + hdr
    for d, nm in (("indoor", "MP3D"), ("outdoor", "Macarons++")):
        sub = {k: v for k, v in res["pairs"].items() if k.startswith("C2")}
        L2 += [f"## {nm}：C2 相对 C1 / B2 / T1 / E2D / MAG-B", "", "| 配对 | ΔAUC [CI] | 胜/负 | ΔFinal [CI] | 胜/负 | ΔCov@20 [CI] | 有意义 | 退化 |", "|---|---|---|---|---|---|---|---|"]
        for k, v in sub.items():
            p, c = v[d]["paired"], v[d]["criteria"]
            L2.append(f"| {k} | {f(p['auc'])} | {p['auc']['wins']}/{p['auc']['losses']} | {f(p['final'])} | {p['final']['wins']}/{p['final']['losses']} | {f(p['cov20'])} | {'✅' if c['meaningful'] else '—'} | {'❌' if c['degraded'] else '—'} |")
        L2.append("")
    L2 += ["## C1 与 C2 的逐 episode 对照（AUC / Final）", ""]
    for rows, nm in ((rows_in, "MP3D"), (rows_out, "Macarons++")):
        L2 += [f"### {nm}", "", "| 场景 | 起点 | C1 AUC | C2 AUC | ΔAUC C2−C1 | C1 Final | C2 Final | ΔFinal | C1 平均 regret pp | C2 平均 regret pp |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            r1 = [x["regret"] for x in r["C1"]["diag"] if x.get("regret") is not None]
            r2 = [x["regret"] for x in r["C2"]["diag"] if x.get("regret") is not None]
            L2.append(f"| {r['scene']} | {r['start']} | {r['C1']['auc']:.3f} | {r['C2']['auc']:.3f} | {r['C2']['auc'] - r['C1']['auc']:+.3f} | {r['C1']['final']:.3f} | {r['C2']['final']:.3f} | "
                      f"{r['C2']['final'] - r['C1']['final']:+.3f} | {100 * np.mean(r1):.3f} | {100 * np.mean(r2):.3f} |")
        L2.append("")
    open(os.path.join(REP, "C_ORACLE_GAIN_G3.md"), "w").write("\n".join(L2) + "\n")
    open(os.path.join(REP, "C_FAILURE_CASES.md"), "w").write("# Experiment C — 失败案例（自动按预先规则挑选 `scripts/c_analyze.py`）\n\n" + failure_cases(rows_in, rows_out, res) + "\n")
    print(json.dumps({k: {d: (round(v[d]["paired"]["auc"]["mean"], 4), v[d]["paired"]["auc"]["wins"], round(v[d]["paired"]["final"]["mean"], 4), v[d]["criteria"]["meaningful"], v[d]["criteria"]["degraded"])
                          for d in ("indoor", "outdoor")} for k, v in res["pairs"].items()}, indent=0))
    print("means", {d: {g: (round(res["means"][d]["auc"][g]["mean"], 3), round(res["means"][d]["final"][g]["mean"], 3)) for g in GRP} for d in ("indoor", "outdoor")})


if __name__ == "__main__":
    main()
