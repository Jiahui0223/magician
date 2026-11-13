"""Experiment U Phase E -- collection + statistics of the closed-loop episodes (CPU; seen2scene env). Nothing is re-run.
Sources: D1 = archived Table 1 row V3-T0-D1, C1 = archived Table 3 row V3-T0-K4D1 (both via cvpr_lib.load_manifest: replay-verified
canonical records, protocol table1-protocol-v1.0); U1 / U2 / U3 (and an optional C1 re-run 'c1') = this experiment's episodes
work_dirs/exp6_v3/experiment_U/closed_loop/<policy>/<scene>_s<k>/ (episode file + official replay metrics; a record counts only if the
replay equals the stored curve bitwise, 101 values, 100 moves, V3-T0 checkpoint sha in the planner log).
Metrics (frozen, = cvpr_lib): C[0..100] official coverage; AUC@T = trapezoid(C[0..T]) / T; Cov@T = C[T]; Final = C[100]; scene value = mean
of the 5 starts; group value = scene-equal mean over COMPLETE scenes. Groups All-14 / Train-6 / Dev-5 / Test-3 / Dev+Test-8 (never
called "unseen" as a whole). Paired scene bootstrap (10,000 draws, seed 0) vs D1 and vs C1, per-scene win / loss, full curves saved.
Outputs results/closed_loop/*.csv|json|npz, figures/U1_coverage_curves.png, results/closed_loop/CLOSED_LOOP_STATUS.md.
Usage: python u_collect.py [--policies u1,u2,u3]"""
import argparse
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

WU = os.path.join(R, "work_dirs", "exp6_v3", "experiment_U", "closed_loop")
OUT = os.path.join(UD, "results", "closed_loop")
NAMES = ["AUC@20", "Cov@20", "AUC@50", "AUC@100", "Final"]
EXTRA = ["AUC@10", "Cov@10", "Cov@50", "AUC@30"]
DISP = {"D1": "D1 Coverage (Two-Step) [archived V3-T0-D1]", "C1": "C1 Coverage (One-Step) [archived V3-T0-K4D1]", "c1": "C1 re-run (this experiment)",
        "u1": "U1 Hit Disagreement", "u2": "U2 Depth Disagreement", "u3": "U3 Geometric Disagreement"}


def load_u(policy):
    """{(scene, start): record} of finished, replay-verified episodes; plus a list of problems."""
    E, bad = {}, []
    for d in sorted(glob.glob(os.path.join(WU, policy, "*_s[0-9]"))):
        s, k = os.path.basename(d).rsplit("_s", 1)
        tag = f"V2M_{s}_s{k}_r0"
        mp, ep = os.path.join(d, "metrics", "V2M", f"{tag}.json"), os.path.join(d, "episodes", f"ep_{tag}.json")
        if not (os.path.exists(mp) and os.path.exists(ep)):
            bad.append(f"{policy} {s} s{k}: not finished")
            continue
        m, e = json.load(open(mp)), json.load(open(ep))
        c = m["cov_official"]
        ok = m.get("official_equal_stored") and e["coverage"] == c and len(c) == 101 and e.get("moves") == 100 and m.get("moves") == 100
        plog = os.path.join(d, "planner_logs", f"pl_{tag}.json")
        sha = json.load(open(plog)).get("ckpt", {}) if os.path.exists(plog) else {}
        sha = sha.get("sha256") if isinstance(sha, dict) else None
        if not ok:
            bad.append(f"{policy} {s} s{k}: replay / length check failed")
            continue
        E[(s, int(k))] = {"scene": s, "start": int(k), "coverage_official": c, "executed_keys": e["executed_keys"], "episode_file": os.path.relpath(ep, R),
                          "metrics_file": os.path.relpath(mp, R), "ckpt_sha256": sha, "split": L.SPLIT[s]}
    return E, bad


def planner_stats(policy):
    """decision time, fallback count and per-step chosen-candidate scores from the planner step logs."""
    dt, fb, n = [], 0, 0
    for f in glob.glob(os.path.join(WU, policy, "*", "planner_logs", "*.steps.jsonl")):
        for line in open(f):
            s = json.loads(line)
            n += 1
            dt.append(s.get("decision_s") or 0.0)
            fb += int(bool((s.get("choice") or {}).get("fallback")))
    return {"steps": n, "decision_s_mean": float(np.mean(dt)) if dt else None, "decision_s_median": float(np.median(dt)) if dt else None, "fallback_steps": fb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="u1,u2,u3")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    M = {"D1": L.load_manifest("V3-T0-D1"), "C1": L.load_manifest("V3-T0-K4D1")}
    problems = []
    for p in a.policies.split(","):
        M[p], b = load_u(p)
        problems += b
    for k, E in M.items():
        sh = {r["ckpt_sha256"] for r in E.values()}
        if sh and sh != {L.T0_SHA}:
            problems.append(f"{k}: checkpoint sha {sh} != V3-T0")
    names = NAMES + EXTRA
    tabs = {k: L.scene_table(E, names) for k, E in M.items()}
    rows, prow, scn = [], [], []
    for g in ("All-14", "Train-6", "Dev-5", "Test-3", "Dev+Test-8"):
        for k in M:
            r = {"group": g, "method": k, "display": DISP.get(k, k), "n_episodes": sum(1 for (s, _) in M[k] if s in L.GROUPS[g])}
            for n in names:
                r[n], r["n_scenes"] = L.group_mean(tabs[k], L.GROUPS[g], n)
            rows.append(r)
            for ref in ("D1", "C1"):
                if k == ref:
                    continue
                for n in names:
                    m_, lo, hi, ns, w, l_ = L.paired(tabs[k], tabs[ref], L.GROUPS[g], n)
                    prow.append({"group": g, "method": k, "ref": ref, "metric": n, "diff": m_, "lo": lo, "hi": hi, "n_scenes": ns, "wins": w, "losses": l_})
    for k in M:
        for s in L.SCENE_ORDER:
            if s in tabs[k]:
                scn.append({"method": k, "scene": s, "split": L.SPLIT[s], **{n: tabs[k][s][n] for n in names}})
    L.write_csv(os.path.join(OUT, "table_U1_groups.csv"), list(rows[0].keys()), [list(x.values()) for x in rows])
    if prow:
        L.write_csv(os.path.join(OUT, "paired_vs_D1_C1.csv"), list(prow[0].keys()), [list(x.values()) for x in prow])
    if scn:
        L.write_csv(os.path.join(OUT, "per_scene.csv"), list(scn[0].keys()), [list(x.values()) for x in scn])
    # full per-step curves (every episode, every method) + scene-equal mean curves
    cur = {}
    for k, E in M.items():
        for (s, st), r in E.items():
            cur[f"{k}|{s}|s{st}"] = np.asarray(r["coverage_official"], np.float64)
        mc, _ = L.mean_curve(E, L.GROUPS["All-14"])
        if mc is not None:
            cur[f"{k}|ALL14_scene_equal_mean"] = mc
    np.savez_compressed(os.path.join(OUT, "coverage_curves.npz"), **cur)
    stats = {k: planner_stats(k) for k in M if k not in ("D1", "C1")}
    status = {k: {"episodes": len(E), "complete_scenes": len(tabs[k])} for k, E in M.items()}
    json.dump({"status": status, "problems": problems, "planner": stats}, open(os.path.join(OUT, "collect_status.json"), "w"), indent=1)
    figure(M)
    with open(os.path.join(OUT, "CLOSED_LOOP_STATUS.md"), "w") as f:
        f.write("# Experiment U — Phase E 汇总状态（u_collect.py 自动生成）\n\n")
        f.write("| 方法 | episode | 完整场景 |\n|---|---|---|\n" + "".join(f"| {DISP.get(k, k)} | {v['episodes']} / 70 | {v['complete_scenes']} / 14 |\n" for k, v in status.items()))
        f.write("\n| 组 | 方法 | " + " | ".join(NAMES) + " | 场景 |\n|---|---|" + "---|" * (len(NAMES) + 1) + "\n")
        for r in rows:
            f.write(f"| {r['group']} | {r['method']} | " + " | ".join(L.fmt(r[n]) for n in NAMES) + f" | {r['n_scenes']} |\n")
        f.write("\n问题：" + ("无" if not problems else "；".join(problems[:30])) + "\n")
    print(json.dumps(status), "problems", len(problems))


def figure(M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"D1": "#2a78d6", "C1": "#1baf7a", "c1": "#008300", "u1": "#eb6834", "u2": "#6250d6", "u3": "#e34948"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, hi in zip(axs, (100, 20)):
        for k, E in M.items():
            mc, per = L.mean_curve(E, L.GROUPS["All-14"])
            if mc is None:
                continue
            lab = f"{k} ({len(per)}/14 scenes)" if len(per) < 14 else k
            ax.plot(np.arange(hi + 1), mc[:hi + 1], lw=2, color=col.get(k, "#555555"), label=lab)
        ax.set_xlabel("step"); ax.set_ylabel("official coverage (scene-equal, complete scenes)")
        ax.set_title(f"All-14, steps 0-{hi}")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(color="#e6e6e6", lw=0.6)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(UD, "figures"), exist_ok=True)
    fig.savefig(os.path.join(UD, "figures", "U1_coverage_curves.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
