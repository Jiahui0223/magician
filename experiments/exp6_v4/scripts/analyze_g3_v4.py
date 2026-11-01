"""EXP6 V4 / V3.1 -- G3 closed-loop analysis (seen2scene env, CPU): exp6_v3/shared/analyze_g3.py readers / paired scene
bootstrap (Task 3 t34_analyze) over the historical groups MAG-G, MAG-B, V2O (V2 Original, Task 2 sparse planner), B0,
B2, E2D, T0, T1 and the Outdoor-only arms V31, V4 (work_dirs/exp6_v4/eval). Pairs: V4-V31 (cross-architecture, same
data / budget), each new arm vs V2O / B2 / T0 or T1 / E2D / MAG-G / MAG-B. Output exp6_v4/reports/g3_v4.{json,md},
reports/figures/g3_v4_coverage_curves.png. Usage: python analyze_g3_v4.py"""
import json
import os
import sys

import numpy as np

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v3", "shared"))
import analyze_g3 as AG  # noqa: E402

T, G3 = AG.T, AG.G3
W4 = os.path.join(REPO, "work_dirs", "exp6_v4", "eval")
E4 = os.path.join(REPO, "experiments", "exp6_v4")
OUT, FIG = os.path.join(E4, "reports"), os.path.join(E4, "reports", "figures")
TASK2 = ("MAG-G", "MAG-B", "V2O")
GRP = ("MAG-G", "MAG-B", "V2O", "B0", "B2", "E2D", "T0", "T1", "V31", "V4")
PAIRS = (("V4", "V31"), ("V4", "V2O"), ("V31", "V2O"), ("V4", "B2"), ("V31", "B2"), ("V4", "T1"), ("V31", "T0"), ("V4", "E2D"), ("V31", "E2D"),
         ("V4", "MAG-G"), ("V31", "MAG-G"), ("V4", "MAG-B"), ("V31", "MAG-B"), ("V4", "B0"), ("V31", "B0"), ("V31", "T1"))
LAB = {"MAG-G": "MAGICIAN Greedy", "MAG-B": "MAGICIAN Beam", "V2O": "V2 Original (Outdoor-only scratch 60k, sparse P1)", "B0": "B0 V2-Mix sparse",
       "B2": "B2 V2-Mix D100", "E2D": "E2-D100 Mixed FT", "T0": "V3-T0 Mixed scratch", "T1": "V3-T1 Mixed scratch (X-Attn)",
       "V31": "V3.1 Outdoor-only scratch", "V4": "V4 Outdoor-only scratch (Ray X-Attn)"}
SRC = {"MAG-G": "official weights", "MAG-B": "official weights", "V2O": "Outdoor-only scratch 60k, sparse / sparse", "B0": "Mixed FT 20k, sparse / sparse",
       "B2": "Mixed FT 20k, D100 / sparse", "E2D": "Mixed FT 20k + 5k, D100 / sparse", "T0": "Mixed scratch 40k, D100 / sparse", "T1": "Mixed scratch 40k, D100 / sparse",
       "V31": "Outdoor-only scratch 40k, D100 / sparse", "V4": "Outdoor-only scratch 40k, D100 / sparse"}
COL = dict(AG.COL, **{"V2O": "#c9a227", "V31": "#0b7285", "V4": "#d6336c"})
AG.ROOTS.update({g: (os.path.join(AG.T2E, "{d}", "closed_loop"), os.path.join(AG.T2E, "outdoor", "metrics"), g) for g in TASK2})
AG.ROOTS.update({g: (os.path.join(W4, "closed_loop"), os.path.join(W4, "metrics", "outdoor"), "V2M") for g in ("V31", "V4")})


def _indoor_dir(g, tag):
    root = AG.ROOTS[g][0].replace("{d}", "indoor")
    return os.path.join(root, g, tag) if g in TASK2 else os.path.join(root, "indoor", g, "V2M", tag)


def _outdoor(g, s, k):
    root, mroot, pre = AG.ROOTS[g]
    root = root.replace("{d}", "outdoor")
    base = os.path.join(root, g) if g in TASK2 else os.path.join(root, "outdoor", g, "V2M")
    tg = f"{pre}_{s}_s{k}_r0"
    return base, tg, os.path.join(mroot, g, "" if g in TASK2 else "V2M", f"{tg}.json").replace("//", "/")


AG._indoor_dir, AG._outdoor = _indoor_dir, _outdoor
T.GRP = GRP


def main():
    global GRP, PAIRS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=None, help="interim: subset of groups (e.g. without V31); pairs restricted accordingly")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    if a.groups:
        GRP = tuple(g for g in GRP if g in a.groups)
        PAIRS = tuple((x, y) for x, y in PAIRS if x in GRP and y in GRP)
        T.GRP = GRP
    primary = f"{PAIRS[0][0]}-{PAIRS[0][1]}"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    T.ST.setup()
    rows_in, rows_out = AG.indoor_episodes(), AG.outdoor_episodes()
    P = {f"{hi}-{lo}": {"indoor": T.summarize(rows_in, "indoor", hi, lo), "outdoor": T.summarize(rows_out, "outdoor", hi, lo)} for hi, lo in PAIRS}
    for r in rows_in:
        for g in GRP:
            r[g]["stall"] = G3.stall(r[g]["curve"])
    st = {g: {"episodes_with_stall": int(sum(r[g]["stall"] >= 30 for r in rows_in)), "mean_longest_window": float(np.mean([r[g]["stall"] for r in rows_in]))} for g in GRP}
    dec = {}
    for hi, lo in PAIRS:
        k = f"{hi}-{lo}"
        c = {d: P[k][d]["criteria"] for d in ("indoor", "outdoor")}
        dec[k] = {"indoor": c["indoor"], "outdoor": c["outdoor"],
                  "verdict": "meaningful on both" if c["indoor"]["meaningful"] and c["outdoor"]["meaningful"] else
                  ("degraded" if c["indoor"]["degraded"] or c["outdoor"]["degraded"] else
                   ("meaningful on one" if c["indoor"]["meaningful"] or c["outdoor"]["meaningful"] else "neither"))}
    per_ep = {"indoor": [{"scene": r["scene"], "start": r["start"], **{g: {k: r[g][k] for k in ("auc", "final", "cov20", "stall", "translations", "time_per_move", "peak_gb")} for g in GRP}} for r in rows_in],
              "outdoor": [{"scene": r["scene"], "start": r["start"], "split": r["split"], **{g: {k: r[g][k] for k in ("auc", "final", "cov20", "hover", "time_per_move", "peak_gb")} for g in GRP}} for r in rows_out]}
    res = {"groups": list(GRP), "sources": SRC, "B0_equals_task2": {"indoor": sum(r["B0_equals_task2"] for r in rows_in), "outdoor": sum(r["B0_equals_task2"] for r in rows_out)},
           "pairs": {k: {d: {"paired": v[d]["paired"], "criteria": v[d]["criteria"]} for d in ("indoor", "outdoor")} for k, v in P.items()},
           "means": {d: P[primary][d]["means"] for d in ("indoor", "outdoor")}, "stall_mp3d": st, "decisions": dec,
           "per_scene": {d: P[primary][d]["per_scene"] for d in ("indoor", "outdoor")}, "per_episode": per_ep}
    json.dump(res, open(os.path.join(OUT, f"g3_v4{a.suffix}.json"), "w"), indent=1, default=float)
    f = T.fmt
    L = ["# V4 / V3.1 G3 闭环（自动生成，`exp6_v4/scripts/analyze_g3_v4.py`；均值 = 场景 / 楼层均值的平均，[CI] = 场景级 bootstrap 95 %）", "",
         "| 数据集 | 组 | 训练来源 / P1 口径 | AUC | Final | Cov@20 | Cov@50 | 每步时间 s* | 峰值显存 GB |", "|---|---|---|---|---|---|---|---|---|"]
    for d, nm in (("indoor", "MP3D（10 ep × 150 步）"), ("outdoor", "Macarons++（30 ep × 100 步）")):
        M = res["means"][d]
        for g in GRP:
            L.append(f"| {nm} | {LAB[g]} | {SRC[g]} | {f(M['auc'][g])} | {f(M['final'][g])} | {f(M['cov20'][g])} | {f(M['cov50'][g])} | "
                     f"{M['time_per_move'][g]['mean']:.2f} | {M['peak_gb'][g]['mean']:.1f} |")
    L += ["", "\\* 每步时间为各自运行时的共享负载记录，不可直接比较。", "", "| 配对差 | 数据集 | ΔAUC [CI] | 胜/负 | ΔFinal [CI] | 胜/负 | ΔCov@20 | 有意义 | 退化 |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["pairs"].items():
        for d in ("indoor", "outdoor"):
            p, c = v[d]["paired"], v[d]["criteria"]
            L.append(f"| {k} | {'MP3D' if d == 'indoor' else 'Macarons++'} | {f(p['auc'])} | {p['auc']['wins']}/{p['auc']['losses']} | {f(p['final'])} | "
                     f"{p['final']['wins']}/{p['final']['losses']} | {f(p['cov20'])} | {c['meaningful']} | {c['degraded']} |")
    L += ["", "判定（Task 3 §4 标准）：" + "；".join(f"{k}: {v['verdict']}" for k, v in dec.items()), "", "逐场景 / 楼层 AUC · Final · Cov@20（后两项若分析器未提供则为 nan；完整逐 episode 值在 json 的 per_episode）：", ""]
    for d in ("indoor", "outdoor"):
        for s, v in res["per_scene"][d].items():
            L.append(f"- {s}: " + "，".join(f"{g} {v[g]['auc']:.3f} / {v[g].get('final', float('nan')):.3f} / {v[g].get('cov20', float('nan')):.3f}" for g in GRP))
    L += ["", "MP3D 中期停滞 episode 数（/ 10）：" + "，".join(f"{g} {st[g]['episodes_with_stall']}" for g in GRP)]
    open(os.path.join(OUT, f"g3_v4{a.suffix}.md"), "w").write("\n".join(L) + "\n")
    plt = AG.plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, rows, nm, xmax in ((axs[0], rows_in, "MP3D zero-shot (C_obs), 10 episodes", 150), (axs[1], rows_out, "Macarons++ (monotone coverage), 30 episodes", 100)):
        for g in GRP:
            c = np.mean([r[g]["curve"] for r in rows], 0)
            ax.plot(c, color=COL[g], lw=2.2 if g in ("V31", "V4") else 1.4, ls=":" if g in TASK2 else ("--" if g in ("B0", "B2", "E2D") else "-"),
                    label=f"{g}  AUC {np.mean([r[g]['auc'] for r in rows]):.3f} · final {c[-1]:.3f}")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("move")
        ax.set_ylabel("coverage")
        ax.set_title(nm, loc="left")
        ax.legend(fontsize=7.5, loc="lower right")
        T.ST.clean(ax, "both")
    fig.suptitle("EXP6 V4 / V3.1 G3: Outdoor-only scratch arms (solid, thick) vs Mixed V3 (solid), Mixed-FT V2 (dashed), Task 2 references (dotted); measured", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"g3_v4{a.suffix}_coverage_curves.png"))
    plt.close(fig)
    print(json.dumps({k: v["verdict"] for k, v in dec.items()}))


if __name__ == "__main__":
    main()
