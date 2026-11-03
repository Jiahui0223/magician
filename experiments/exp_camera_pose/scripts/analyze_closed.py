"""Camera Pose Phase 1 -- closed-loop analysis (seen2scene env, CPU): Stage A regression, Stage C pilot gate, Stage D G3.

Episodes are read from the job list (cl_in / cl_out lines). C1 runs and new C0 runs live under
work_dirs/exp_camera_pose/closed_loop/<arm>_<cond>/; historical C0 (B2 / E2-D100, Task 3 + E2 pilot G3) under
work_dirs/exp5_future_view/task3_full_depth_p1/{closed_loop,metrics}/.../{B2,E2D}. Metrics (as in the historical reports):
MP3D: episode.json auc150 / cov_final / cov20 (C_obs); Macarons++: metrics json auc_mono / cov_mono_final / cov_mono[20]
(official monotone coverage). Per episode: delta C1 - C0; per (arm, domain): means, episode wins, the pre-registered gate
(protocol.md §3: mean dAUC >= +0.005, no episode dAUC < -0.03, mean dFinal >= -0.01); path statistics (translations,
rotations, hovers); C1 step-log diagnostics (share of steps whose argmax moved off the DistP1 argmax, all-zero DistP1
steps, F statistics, prior time). For --phase full: scene-level bootstrap CI and Dev-3 / Test-3 split.
Output reports/<phase>.{json,md} and reports/figures/<phase>_coverage.png.
Usage: python analyze_closed.py --phase pilot|full --jobs JOBFILE [--c0 historical|new]"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(E))
W = os.path.join(R, "work_dirs", "exp_camera_pose", "closed_loop")
H = os.path.join(R, "work_dirs", "exp5_future_view", "task3_full_depth_p1")
DEV3, TEST3 = ("barts", "pantheon", "colosseum"), ("alhambra", "bridge", "neuschwanstein")
GATE = {"mean_dAUC_min": 0.005, "episode_dAUC_min": -0.03, "mean_dFinal_min": -0.01}


def jl(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def indoor_paths(arm, cond, scan, lvl, st, hist):
    tag = f"{scan}_L{lvl}_S{st}"
    if hist:
        d = os.path.join(H, "closed_loop", "indoor", arm, "V2M", tag)
    else:
        g = f"{arm}_{cond}"
        d = os.path.join(W, g, "closed_loop", "indoor", g, "V2M", tag)
    return os.path.join(d, "episode.json"), os.path.join(d, "c1_steps.jsonl")


def outdoor_paths(arm, cond, scene, k, hist):
    tg = f"V2M_{scene}_s{k}_r0"
    if hist:
        return (os.path.join(H, "closed_loop", "outdoor", arm, "V2M", "episodes", f"ep_{tg}.json"), os.path.join(H, "metrics", "outdoor", arm, "V2M", f"{tg}.json"), None)
    g = f"{arm}_{cond}"
    base = os.path.join(W, g, "closed_loop", "outdoor", g, "V2M")
    return os.path.join(base, "episodes", f"ep_{tg}.json"), os.path.join(W, g, "metrics", "outdoor", g, "V2M", f"{tg}.json"), os.path.join(base, "c1_steps", f"{tg}.jsonl")


def path_stats(poses):
    tr = sum(1 for a, b in zip(poses, poses[1:]) if list(a[:3]) != list(b[:3]))
    rot = sum(1 for a, b in zip(poses, poses[1:]) if list(a[:3]) == list(b[:3]) and list(a[3:]) != list(b[3:]))
    stay = sum(1 for a, b in zip(poses, poses[1:]) if list(a) == list(b))
    return {"translations": tr, "rotations": rot, "stays": stay, "distinct_positions": len({tuple(p[:3]) for p in poses})}


def steplog_stats(p):
    S = jl(p)
    if not S:
        return None
    ch = [s["argmax_score"] != s["argmax_D"] for s in S if s["n"] > 1]
    return {"calls": len(S), "argmax_moved_share": float(np.mean(ch)) if ch else 0.0, "all_zero_D_steps": int(sum(max(s["D"]) == 0 for s in S if s["D"])),
            "F_mean": float(np.mean([np.mean(s["F"]) for s in S if s["F"]])), "F_max_mean": float(np.mean([max(s["F"]) for s in S if s["F"]])),
            "frontier_s_mean": float(np.mean([s["frontier_s"] for s in S]))}


def episode(dom, arm, cond, ep, hist):
    if dom == "indoor":
        pe, ps = indoor_paths(arm, cond, *ep, hist)
        if not os.path.exists(pe):
            return None
        e = json.load(open(pe))
        c = np.asarray(e["coverage"], float)
        r = {"auc": float(e["auc150"]), "final": float(e["cov_final"]), "cov20": float(e["cov20"]), "curve": c.tolist()}
        r.update(path_stats([s["pose"] for s in e["steps"]] + [e["executed"][-1]] if e.get("executed") else [s["pose"] for s in e["steps"]]))
        r["picks"] = [s["pick"] for s in e["steps"]]
        r["fallbacks"] = int(sum(bool(s.get("fallback")) for s in e["steps"]))
    else:
        pe, pm, ps = outdoor_paths(arm, cond, *ep, hist)
        if not (os.path.exists(pe) and os.path.exists(pm)):
            return None
        e, m = json.load(open(pe)), json.load(open(pm))
        c = np.asarray(m["cov_mono"], float)
        r = {"auc": float(m["auc_mono"]), "final": float(m["cov_mono_final"]), "cov20": float(c[20]), "curve": c.tolist()}
        poses = [json.loads(s["pose_key"]) for s in e["steps"]]
        r.update(path_stats(poses))
        r["hovers"] = int(e.get("n_hover", 0))
        r["picks"] = [s.get("action_key") for s in e["steps"]]          # the last record (t = moves) has no action
        r["fallbacks"] = int(sum(bool((s.get("planner") or {}).get("fallback")) for s in e["steps"]))
    r["steplog"] = steplog_stats(ps) if ps else None
    return r


def boot_ci(x, groups, n=10000, seed=0):
    """Scene-level bootstrap of the mean paired difference (episodes averaged within scene first)."""
    rng = np.random.default_rng(seed)
    G = sorted(set(groups))
    per = np.asarray([np.mean([v for v, g in zip(x, groups) if g == s]) for s in G])
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], {s: float(v) for s, v in zip(G, per)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "full"], required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--c0-outdoor", choices=["historical", "new"], default="historical")
    ap.add_argument("--c0-indoor", choices=["historical", "new"], default="new")
    a = ap.parse_args()
    lam = json.load(open(os.path.join(E, "configs", "c1_lambda.json")))
    eps = {}
    for line in open(a.jobs):
        f = line.strip().split(":")
        if not f[0] or f[0] == "cov_out" or f[2] != "C1":
            continue
        dom = "indoor" if f[0] == "cl_in" else "outdoor"
        ep = (f[3], int(f[4]), int(f[5])) if dom == "indoor" else (f[3], int(f[4]))
        eps.setdefault((f[1], dom), []).append(ep)
    res = {"phase": a.phase, "lambda": {k: lam[k] for k in ("B2", "E2D")}, "gate": GATE, "groups": {}}
    # Stage A regression (B2 C0 barts s0 new vs historical)
    if a.phase == "pilot":
        n0, h0 = episode("outdoor", "B2", "C0", ("barts", 0), False), episode("outdoor", "B2", "C0", ("barts", 0), True)
        if n0 and h0:
            res["regression_B2_C0_barts_s0"] = {"picks_equal": n0["picks"] == h0["picks"], "curve_equal": n0["curve"] == h0["curve"],
                                                "auc_new": n0["auc"], "auc_hist": h0["auc"], "final_new": n0["final"], "final_hist": h0["final"]}
    for (arm, dom), lst in sorted(eps.items()):
        c0hist = (a.c0_outdoor if dom == "outdoor" else a.c0_indoor) == "historical"
        rows = []
        for ep in lst:
            c1, c0 = episode(dom, arm, "C1", ep, False), episode(dom, arm, "C0", ep, c0hist)
            if c1 is None or c0 is None:
                rows.append({"episode": ep, "missing": [n for n, x in (("C1", c1), ("C0", c0)) if x is None]})
                continue
            rows.append({"episode": ep, "scene": ep[0], "C0": c0, "C1": c1, "dAUC": c1["auc"] - c0["auc"], "dFinal": c1["final"] - c0["final"], "dCov20": c1["cov20"] - c0["cov20"],
                         "first_divergent_step": next((i for i, (p, q) in enumerate(zip(c0["picks"], c1["picks"])) if p != q), None)})
        ok = [r for r in rows if "dAUC" in r]
        g = {"n_episodes": len(lst), "n_complete": len(ok), "c0_source": "historical" if c0hist else "new (same entry, lambda 0)", "episodes": rows}
        if ok:
            d = np.asarray([r["dAUC"] for r in ok]); f = np.asarray([r["dFinal"] for r in ok])
            g.update({"C0_auc": float(np.mean([r["C0"]["auc"] for r in ok])), "C1_auc": float(np.mean([r["C1"]["auc"] for r in ok])),
                      "C0_final": float(np.mean([r["C0"]["final"] for r in ok])), "C1_final": float(np.mean([r["C1"]["final"] for r in ok])),
                      "C0_cov20": float(np.mean([r["C0"]["cov20"] for r in ok])), "C1_cov20": float(np.mean([r["C1"]["cov20"] for r in ok])),
                      "mean_dAUC": float(d.mean()), "mean_dFinal": float(f.mean()), "min_dAUC": float(d.min()), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                      "identical_episodes": int(sum(r["C0"]["picks"] == r["C1"]["picks"] for r in ok)),
                      "path": {c: {k: float(np.mean([r[c][k] for r in ok])) for k in ("translations", "rotations", "stays", "distinct_positions")} for c in ("C0", "C1")},
                      "c1_steplog": {k: float(np.mean([r["C1"]["steplog"][k] for r in ok if r["C1"].get("steplog")])) for k in ("argmax_moved_share", "all_zero_D_steps", "F_mean", "frontier_s_mean")} if any(r["C1"].get("steplog") for r in ok) else None})
            if a.phase == "pilot":
                g["gate_pass"] = bool(len(ok) == len(lst) and d.mean() >= GATE["mean_dAUC_min"] and d.min() >= GATE["episode_dAUC_min"] and f.mean() >= GATE["mean_dFinal_min"])
            else:
                ci, per = boot_ci(list(d), [r["scene"] for r in ok])
                cif, _ = boot_ci(list(f), [r["scene"] for r in ok])
                g.update({"dAUC_ci95": ci, "dFinal_ci95": cif, "per_scene_dAUC": per, "scene_wins": int(sum(v > 0 for v in per.values())), "n_scenes": len(per)})
                if dom == "outdoor":
                    for nm, S in (("dev3", DEV3), ("test3", TEST3)):
                        sub = [r for r in ok if r["scene"] in S]
                        if sub:
                            g[f"{nm}_mean_dAUC"] = float(np.mean([r["dAUC"] for r in sub])); g[f"{nm}_mean_dFinal"] = float(np.mean([r["dFinal"] for r in sub]))
        res["groups"][f"{arm}|{dom}"] = g
    out = os.path.join(E, "reports")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    slim = json.loads(json.dumps(res, default=float))
    for g in slim["groups"].values():
        for r in g["episodes"]:
            for c in ("C0", "C1"):
                if c in r:
                    r[c].pop("curve", None); r[c].pop("picks", None)
    json.dump(slim, open(os.path.join(out, f"{a.phase}.json"), "w"), indent=1)
    plot(res, os.path.join(out, "figures", f"{a.phase}_coverage.png"))
    write_md(res, os.path.join(out, f"{a.phase}.md"))
    print(json.dumps({k: {kk: g.get(kk) for kk in ("n_complete", "mean_dAUC", "mean_dFinal", "min_dAUC", "wins", "losses", "gate_pass", "dAUC_ci95")} for k, g in res["groups"].items()}, indent=1, default=float))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G = [(k, g) for k, g in res["groups"].items() if g.get("n_complete")]
    if not G:
        return
    fig, axes = plt.subplots(1, len(G), figsize=(4.6 * len(G), 3.8), squeeze=False)
    for ax, (k, g) in zip(axes[0], G):
        ok = [r for r in g["episodes"] if "dAUC" in r]
        for c, col in (("C0", "#868e96"), ("C1", "#1971c2")):
            cs = [np.asarray(r[c]["curve"]) for r in ok]
            n = min(len(x) for x in cs)
            for x in cs:
                ax.plot(x[:n], color=col, lw=0.6, alpha=0.35)
            ax.plot(np.mean([x[:n] for x in cs], 0), color=col, lw=2.2, label=f"{c}  AUC {g[f'{c}_auc']:.3f} · Final {g[f'{c}_final']:.3f}")
        ax.set_title(f"{k.replace('|', ' · ')}  ({len(ok)} ep, ΔAUC {g['mean_dAUC']:+.3f})", fontsize=10)
        ax.set_xlabel("move"); ax.set_ylabel("coverage"); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Camera Pose {res['phase']}: C0 (original DistP1) vs C1 (DistP1 + λ·FrontierPrior, λ = {res['lambda']})", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def write_md(res, path):
    L = [f"# Camera Pose —— {res['phase']}（C0 vs C1；λ = {res['lambda']}；评估场景非盲）", ""]
    if "regression_B2_C0_barts_s0" in res:
        x = res["regression_B2_C0_barts_s0"]
        L += [f"**Stage A 回归**（新入口 λ 0 的 B2 Outdoor barts s0 完整 100 步 vs 历史 C0）：选择序列相同 {x['picks_equal']}，覆盖曲线相同 {x['curve_equal']}（AUC {x['auc_new']:.4f} vs {x['auc_hist']:.4f}）。", ""]
    L += ["| 模型 · 域 | episodes | C0 AUC / Final / Cov@20 | C1 AUC / Final / Cov@20 | mean ΔAUC | mean ΔFinal | 最差 ΔAUC | 胜 / 负 / 完全相同 | " + ("门槛" if res["phase"] == "pilot" else "ΔAUC 95% CI（场景 bootstrap）") + " |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, g in res["groups"].items():
        if not g.get("n_complete"):
            L.append(f"| {k} | 0/{g['n_episodes']} | — | — | — | — | — | — | — |")
            continue
        last = ("✅ 通过" if g["gate_pass"] else "❌ 未通过") if res["phase"] == "pilot" else f"[{g['dAUC_ci95'][0]:+.3f}, {g['dAUC_ci95'][1]:+.3f}]，场景胜 {g['scene_wins']}/{g['n_scenes']}"
        L.append(f"| {k.replace('|', ' · ')} | {g['n_complete']}/{g['n_episodes']} | {g['C0_auc']:.3f} / {g['C0_final']:.3f} / {g['C0_cov20']:.3f} | {g['C1_auc']:.3f} / {g['C1_final']:.3f} / {g['C1_cov20']:.3f} | "
                 f"{g['mean_dAUC']:+.4f} | {g['mean_dFinal']:+.4f} | {g['min_dAUC']:+.4f} | {g['wins']} / {g['losses']} / {g['identical_episodes']} | {last} |")
    L += ["", "逐 episode：", "", "| 模型 · 域 | episode | C0 AUC / Final | C1 AUC / Final | ΔAUC | ΔFinal | 首个不同选择的步 | C0 / C1 平移 · 旋转 | C1 选择偏离 DistP1 argmax 的步比例 |", "|---|---|---|---|---|---|---|---|---|"]
    for k, g in res["groups"].items():
        for r in g["episodes"]:
            if "dAUC" not in r:
                L.append(f"| {k} | {r['episode']} | 缺失 {r['missing']} | | | | | | |")
                continue
            sl = r["C1"].get("steplog") or {}
            L.append(f"| {k.replace('|', ' · ')} | {'_'.join(str(x) for x in r['episode'])} | {r['C0']['auc']:.3f} / {r['C0']['final']:.3f} | {r['C1']['auc']:.3f} / {r['C1']['final']:.3f} | {r['dAUC']:+.4f} | {r['dFinal']:+.4f} | "
                     f"{r['first_divergent_step']} | {r['C0']['translations']}·{r['C0']['rotations']} / {r['C1']['translations']}·{r['C1']['rotations']} | {sl.get('argmax_moved_share', float('nan')):.3f} |")
    L += ["", f"门槛（protocol.md §3）：mean ΔAUC ≥ {GATE['mean_dAUC_min']}，单 episode ΔAUC ≥ {GATE['episode_dAUC_min']}，mean ΔFinal ≥ {GATE['mean_dFinal_min']}。曲线：`figures/{res['phase']}_coverage.png`。"]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
