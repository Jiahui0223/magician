"""EXP6 V3 -- G3 closed-loop analysis (seen2scene env, CPU): the Task 3 readers / paired scene bootstrap
(task3_full_depth_p1/scripts/t34_analyze.py) over the groups B0, B2, E2D (historical, symlinked into
work_dirs/exp6_v3/eval) and T0, T1 (V3). Pairs: T1-T0 (primary), T0-B2, T1-B2, T0-E2D, T1-E2D, T0-B0, T1-B0; the frozen
Task 3 §4 "meaningful / degraded" criteria; Cov@k, behaviour, cost, MP3D mid-episode stall, curves.
Output exp6_v3/reports/g3.{json,md}, reports/figures/g3_coverage_curves.png
"""
import json
import os
import sys

import numpy as np

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts"))
import t34_analyze as T  # noqa: E402
import e2_g3_analyze as G3  # noqa: E402

T.W3 = os.path.join(REPO, "work_dirs", "exp6_v3", "eval")
T2E = os.path.join(REPO, "work_dirs", "exp5_future_view", "task2", "eval")
T.GRP = ("MAG-G", "MAG-B", "B0", "B2", "E2D", "T0", "T1")
# group -> (closed-loop root per dataset, outdoor metrics root, tag prefix); V3 / Task 3 groups use the exp6 eval root
# (reference groups symlinked), the MAGICIAN references are read from the Task 2 evaluation (shared action protocol)
ROOTS = {g: (os.path.join(T.W3, "closed_loop"), os.path.join(T.W3, "metrics", "outdoor"), "V2M") for g in ("B0", "B2", "E2D", "T0", "T1")}
ROOTS.update({g: (os.path.join(T2E, "{d}", "closed_loop"), os.path.join(T2E, "outdoor", "metrics"), g) for g in ("MAG-G", "MAG-B")})
COL = {"MAG-G": "#bdbdbd", "MAG-B": "#7f7f7f", "B0": "#86b6ef", "B2": "#1c5cab", "E2D": "#eb6834", "T0": "#1baf7a", "T1": "#8e44ad"}
LAB = {"MAG-G": "MAGICIAN Greedy (Task 2)", "MAG-B": "MAGICIAN Beam (Task 2)", "B0": "B0 frozen V2-Mix, sparse", "B2": "B2 frozen V2-Mix, D100",
       "E2D": "E2-D100 fine-tuned", "T0": "V3-VAE (T0)", "T1": "V3-VAE-XAttn (T1)"}
PAIRS = (("T1", "T0"), ("T0", "B2"), ("T1", "B2"), ("T0", "E2D"), ("T1", "E2D"), ("T0", "B0"), ("T1", "B0"), ("T0", "MAG-B"), ("T1", "MAG-B"))


def _indoor_dir(g, tag):
    root = ROOTS[g][0].replace("{d}", "indoor")
    return os.path.join(root, g, tag) if g.startswith("MAG") else os.path.join(root, "indoor", g, "V2M", tag)


def _outdoor(g, s, k):
    root, mroot, pre = ROOTS[g]
    root = root.replace("{d}", "outdoor")
    base = os.path.join(root, g) if g.startswith("MAG") else os.path.join(root, "outdoor", g, "V2M")
    tg = f"{pre}_{s}_s{k}_r0"
    return base, tg, os.path.join(mroot, g, "V2M" if not g.startswith("MAG") else "", f"{tg}.json").replace("//", "/")


def indoor_episodes():
    """t34_analyze.indoor_episodes with per-group roots / tags (MAG groups from Task 2)."""
    rows = []
    for ep in T.A.episodes("indoor"):
        r = {"dataset": "indoor", "scene": ep["scene"], "start": ep["start"], "split": "mp3d_val"}
        for g in T.GRP:
            d = _indoor_dir(g, ep["tag"])
            e = json.load(open(os.path.join(d, "episode.json")))
            pl = [json.loads(x) for x in open(os.path.join(d, "planner.jsonl"))] if os.path.exists(os.path.join(d, "planner.jsonl")) else []
            c = np.asarray(e["coverage"])
            ex = [tuple(p) for p in e["executed"]]
            pos = [p[:3] for p in ex]
            r[g] = {"auc": float(np.trapz(c, dx=1) / e["moves"]), "final": float(c[-1]), "curve": c.tolist(),
                    **{f"cov{x}": float(c[x]) for x in (10, 20, 50, 100)},
                    "revisit_pose": float(np.mean([ex[i] in set(ex[:i]) for i in range(1, len(ex))])),
                    "revisit_position": float(np.mean([pos[i] in set(pos[:i]) for i in range(1, len(pos))])),
                    "translations": int(sum(pos[i] != pos[i - 1] for i in range(1, len(pos)))),
                    "n_cand": float(np.mean([s["n_legal"] for s in e["steps"]])),
                    "time_per_move": float(np.mean([x["timing"]["decision_s"] for x in pl])) if pl else float(np.mean([s.get("decision_s", 0) for s in e["steps"]])),
                    "peak_gb": float(max(x.get("peak_mem_gb", 0) for x in pl)) if pl else 0.0, "executed": [list(p) for p in ex]}
        ref = json.load(open(os.path.join(T.T2E, "indoor", "closed_loop", "V2M", ep["tag"], "episode.json")))
        r["B0_equals_task2"] = bool(ref["coverage"] == r["B0"]["curve"] and ref["executed"] == r["B0"]["executed"])
        rows.append(r)
    return rows


def outdoor_episodes():
    rows = []
    for ep in T.A.episodes("outdoor"):
        s, k = ep["scene"], ep["start"]
        r = {"dataset": "outdoor", "scene": s, "start": k, "split": ep["split"]}
        for g in T.GRP:
            base, tg, mp = _outdoor(g, s, k)
            e = json.load(open(os.path.join(base, "episodes", f"ep_{tg}.json")))
            m = json.load(open(mp))
            plp = os.path.join(base, "planner_logs", f"pl_{tg}.json")
            pl = json.load(open(plp)) if os.path.exists(plp) else None
            c = np.asarray(m["cov_mono"])
            keys = e["executed_keys"]
            r[g] = {"auc": float(m["auc_mono"]), "final": float(m["cov_mono_final"]), "curve": c.tolist(),
                    "auc_official": float(m["auc_official"]), "final_official": float(m["cov_official_final"]),
                    **{f"cov{x}": float(c[x]) for x in (10, 20, 50)},
                    "revisit_pose": float(np.mean([keys[i] in set(keys[:i]) for i in range(1, len(keys))])),
                    "hover": int(e.get("n_hover", 0)), "n_cand": float(np.mean([st["n_legal"] for st in e["steps"]])),
                    "time_per_move": float(np.mean([st["decision_s"] for st in pl["steps"]])) if pl else float(np.mean([st.get("decision_s", 0) for st in e["steps"]])),
                    "peak_gb": float(pl.get("peak_mem_gb_this_process", 0)) if pl else 0.0, "keys": keys, "cov_official": e["coverage"]}
        ref = json.load(open(os.path.join(T.T2E, "outdoor", "closed_loop", "V2M", "episodes", f"ep_V2M_{s}_s{k}_r0.json")))
        r["B0_equals_task2"] = bool(ref["coverage"] == r["B0"]["cov_official"] and ref["executed_keys"] == r["B0"]["keys"])
        rows.append(r)
    return rows
E6 = os.path.join(REPO, "experiments", "exp6_v3")
OUT, FIG = os.path.join(E6, "reports"), os.path.join(E6, "reports", "figures")
plt = T.plt


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    T.ST.setup()
    rows_in, rows_out = indoor_episodes(), outdoor_episodes()
    P = {f"{hi}-{lo}": {"indoor": T.summarize(rows_in, "indoor", hi, lo), "outdoor": T.summarize(rows_out, "outdoor", hi, lo)} for hi, lo in PAIRS}
    for r in rows_in:
        for g in T.GRP:
            r[g]["stall"] = G3.stall(r[g]["curve"])
    st = {g: {"episodes_with_stall": int(sum(r[g]["stall"] >= 30 for r in rows_in)), "mean_longest_window": float(np.mean([r[g]["stall"] for r in rows_in]))} for g in T.GRP}
    dec = {}
    for hi, lo in PAIRS:
        k = f"{hi}-{lo}"
        c = {d: P[k][d]["criteria"] for d in ("indoor", "outdoor")}
        dec[k] = {"indoor": c["indoor"], "outdoor": c["outdoor"],
                  "verdict": "meaningful on both" if c["indoor"]["meaningful"] and c["outdoor"]["meaningful"] else
                  ("degraded" if c["indoor"]["degraded"] or c["outdoor"]["degraded"] else
                   ("meaningful on one" if c["indoor"]["meaningful"] or c["outdoor"]["meaningful"] else "neither"))}
    res = {"groups": list(T.GRP), "B0_equals_task2": {"indoor": sum(r["B0_equals_task2"] for r in rows_in), "outdoor": sum(r["B0_equals_task2"] for r in rows_out)},
           "pairs": {k: {d: {"paired": v[d]["paired"], "criteria": v[d]["criteria"]} for d in ("indoor", "outdoor")} for k, v in P.items()},
           "means": {d: P["T1-T0"][d]["means"] for d in ("indoor", "outdoor")}, "stall_mp3d": st, "decisions": dec,
           "per_scene": {d: P["T1-T0"][d]["per_scene"] for d in ("indoor", "outdoor")}}
    json.dump(res, open(os.path.join(OUT, "g3.json"), "w"), indent=1, default=float)
    f = T.fmt
    L = ["# V3 G3 闭环（自动生成，`shared/analyze_g3.py`）", "", "| 数据集 | 组 | AUC | Final | Cov@10 | Cov@20 | Cov@50 | Cov@100 | 每步时间 s* | 峰值显存 GB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d, nm in (("indoor", "MP3D（150 步）"), ("outdoor", "Macarons++（100 步）")):
        M = res["means"][d]
        for g in T.GRP:
            L.append(f"| {nm} | {LAB[g]} | {f(M['auc'][g])} | {f(M['final'][g])} | {f(M['cov10'][g])} | {f(M['cov20'][g])} | {f(M['cov50'][g])} | "
                     f"{f(M['cov100'][g]) if 'cov100' in M else '—'} | {M['time_per_move'][g]['mean']:.2f} | {M['peak_gb'][g]['mean']:.1f} |")
    L += ["", "\\* 每步时间为各自运行时的共享负载记录，不可直接比较。", "", "| 配对差 | 数据集 | ΔAUC | 胜/负 | ΔFinal | 胜/负 | ΔCov@20 | 有意义 | 退化 |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["pairs"].items():
        for d in ("indoor", "outdoor"):
            p, c = v[d]["paired"], v[d]["criteria"]
            L.append(f"| {k} | {'MP3D' if d == 'indoor' else 'Macarons++'} | {f(p['auc'])} | {p['auc']['wins']}/{p['auc']['losses']} | {f(p['final'])} | "
                     f"{p['final']['wins']}/{p['final']['losses']} | {f(p['cov20'])} | {c['meaningful']} | {c['degraded']} |")
    L += ["", "判定（Task 3 §4 标准）：" + "；".join(f"{k}: {v['verdict']}" for k, v in dec.items()), "", "逐场景 AUC：", ""]
    for d in ("indoor", "outdoor"):
        for s, v in res["per_scene"][d].items():
            L.append(f"- {s}: " + "，".join(f"{g} {v[g]['auc']:.3f}" for g in T.GRP))
    L += ["", "MP3D 中期停滞（≥ 30 步覆盖增长 < 0.01 的 episode 数 / 10）：" + "，".join(f"{g} {st[g]['episodes_with_stall']}" for g in T.GRP)]
    open(os.path.join(OUT, "g3.md"), "w").write("\n".join(L) + "\n")
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, rows, nm, xmax in ((axs[0], rows_in, "MP3D (C_obs), 10 episodes", 150), (axs[1], rows_out, "Macarons++ (monotone coverage), 30 episodes", 100)):
        for g in T.GRP:
            c = np.mean([r[g]["curve"] for r in rows], 0)
            ax.plot(c, color=COL[g], lw=2.0, ls=":" if g.startswith("MAG") else ("--" if g in ("B0", "B2", "E2D") else "-"), label=f"{LAB[g]}  AUC {np.mean([r[g]['auc'] for r in rows]):.3f} · final {c[-1]:.3f}")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("move")
        ax.set_ylabel("coverage")
        ax.set_title(nm, loc="left")
        ax.legend(fontsize=8, loc="lower right")
        T.ST.clean(ax, "both")
    fig.suptitle("EXP6 V3 G3: historical B0 / B2 / E2-D100 (dashed) vs scratch V3-VAE (T0) and V3-VAE-XAttn (T1); measured", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "g3_coverage_curves.png"))
    plt.close(fig)
    print(json.dumps({k: v["verdict"] for k, v in dec.items()}))


if __name__ == "__main__":
    main()
