"""EXP6 V3 Task 2 / Experiment 2 -- MP3D failure cases (CPU; brief §8): the longest stall windows of the C1 (GT depth +
DistP1) and B2 closed loops, with a top-down view of the historical observations, the Task 2 planning region, the
candidates with predicted / true gains, the frontier map (records only), the executed move and the subsequent coverage.
C1 episodes carry per-step true gains of all candidates (Experiment C diagnostics); for B2 the fixed states of Pool CL
(Experiment A cache: B2 K=4 scores; oracle.npz: true gains) inside the window are used.
Output figures/e2_failure_<case>.png, report/E2_FAILURE_CASES.md
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
T2 = os.path.dirname(EXP)
E6 = os.path.dirname(T2)
REPO = os.path.dirname(os.path.dirname(E6))
sys.path.insert(0, os.path.join(T2, "common"))
sys.path.insert(0, os.path.join(E6, "ablation_study", "shared"))
sys.path.insert(0, os.path.join(E6, "ablation_study", "experiment_C_gtdepth_closed_loop", "scripts"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "ablation_study", "common"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts"))
import t2_frontier as TF  # noqa: E402
import ab_common as A  # noqa: E402
import t2_common as C2  # noqa: E402
import m_common as MC  # noqa: E402
import mp3d_frame as F  # noqa: E402
import e2_g3_analyze as G3  # noqa: E402

WC = os.path.join(REPO, "work_dirs", "exp6_v3", "ablation_study", "experiment_C", "eval", "closed_loop", "indoor")
WA = os.path.join(REPO, "work_dirs", "exp6_v3", "ablation_study", "cache", "cl", "indoor")
W3 = os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "closed_loop", "indoor", "B2", "V2M")
FIG, REP, OUT = (os.path.join(EXP, k) for k in ("figures", "report", "results"))


def stall_window(curve, lo=30, hi=120, eps=0.01):
    c = np.asarray(curve); best, bs, be = 0, lo, lo
    for s in range(lo, hi):
        e = s
        while e + 1 <= hi and c[e + 1] - c[s] < eps:
            e += 1
        if e - s > best:
            best, bs, be = e - s, s, e
    return bs, be, best


def topdown(ax, recs_xy, cam_xy, traj_xy, cand_xy, pred, true, pick, best, region, fm, title):
    """recs_xy [N,2] observed sample points (m), cam, traj, candidates; frontier cells from fm (converted to metres)."""
    ax.scatter(recs_xy[:, 0], recs_xy[:, 1], s=0.3, c="#9aa5b1", alpha=0.5, rasterized=True, label="observed samples (records 0..t)")
    free, occ, unk = fm.states(); fr = fm.frontier(free, unk)
    for arr, col, lab in ((fr, "#2ca02c", "frontier cell"), (occ, "#111111", "occupied cell")):
        ls, hs = np.nonzero(arr)
        if len(ls):
            p = fm.x_min[None] + (np.stack([ls + 0.5, np.full(len(ls), 0.5), hs + 0.5], 1)) * fm.cell[None]
            m = C2.rt_to_zup_m(p)
            ax.scatter(m[:, 0], m[:, 1], s=28, marker="s", c=col, alpha=0.45, label=lab)
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "-", color="#1c5cab", lw=0.8, alpha=0.8, label="executed trajectory")
    ax.plot(cam_xy[0], cam_xy[1], "*", color="#1c5cab", ms=14, label="current camera")
    (x0, y0), (x1, y1) = region
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#d62728", lw=1.2, ls="--", label="planning region n_win = 5"))
    for j, (cx, cy) in enumerate(cand_xy):
        col = "#ff7f0e" if j == pick else ("#2ca02c" if j == best else "#555555")
        ax.plot(cx, cy, "o", color=col, ms=6 + 40 * (true[j] / max(1e-9, max(true))), alpha=0.7)
        ax.annotate(f"{j}: p{pred[j]:.0f}/g{100 * true[j]:.2f}", (cx, cy), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_aspect("equal"); ax.set_title(title, loc="left", fontsize=9); ax.legend(fontsize=6, loc="upper right")


def main():
    for p in (FIG, REP, OUT):
        os.makedirs(p, exist_ok=True)
    cs = json.load(open(os.path.join(E6, "ablation_study", "experiment_C_gtdepth_closed_loop", "results", "c_summary.json")))
    man = {s["sid"]: s for s in A.load_manifest("indoor")["states"]}
    eps_in = [{"tag": f"{s}_L{lv}_S{k}", "scene": f"{s}_L{lv}", "start": k} for s, lv, k in [tuple(x) for x in C2.EVAL_LEVELS]]
    # C1 stall windows (3 longest) and B2 stall windows (3 longest)
    cases = []
    # second group: the Task 2 V2M (= B0) trajectories, whose fixed states carry B2's cached K = 4 scores and the oracle
    # gains (the Task 3 B2 trajectories themselves have no cached oracle); label "B2@B0traj"
    for grp, root in (("C1", lambda tag: os.path.join(WC, "C1", tag)), ("B2@B0traj", lambda tag: os.path.join(A.CL_IN, "V2M", tag))):
        wins = []
        for ep in eps_in:
            e = json.load(open(os.path.join(root(ep["tag"]), "episode.json")))
            s0, e0, ln = stall_window(e["coverage"])
            wins.append((ln, ep, s0, e0, e))
        for ln, ep, s0, e0, e in sorted(wins, key=lambda x: -x[0])[:3]:
            cases.append((grp, ep, s0, e0, ln, e, root(ep["tag"])))
    L = ["# E2 — MP3D 失败案例（自动生成 `scripts/e2_failure_cases.py`；C1 闭环与 B0 轨迹（B2 在其固定状态上的分数）各取停滞最长的 3 条 episode；GT 只在评估侧）", "",
         "每个案例：(1) 覆盖率 (2) 相机位置 (3) 历史观测俯视图 (4) 规划区域 (5) 候选分布 (6) 预测增益 (7) 真实增益（离线）(8) frontier 与可达性 (9) 规划器选择 (10) 后续覆盖曲线与轨迹；类型：A 有好候选但排错 / B 区域外有收益 / C 候选皆低但有前进方向 / D 合法动作进不了未探索区 / E 候选够但 greedy 长期不理想。", ""]
    rows = []
    for ci, (grp, ep, s0, e0, ln, e, d) in enumerate(cases):
        tag, scene = ep["tag"], ep["scene"]
        card = F.load_card(scene)
        t = s0 if grp == "C1" else None
        if grp != "C1":                                        # a Pool CL state of this B0 episode inside the window
            sts = [s for s in man.values() if s["tag"] == tag and any(src["source"] == "V2M" for src in s["sources"]) and s0 <= s["t"] <= e0]
            if not sts:
                sts = [s for s in man.values() if s["tag"] == tag and any(src["source"] == "V2M" for src in s["sources"]) and s["t"] >= s0]
            if not sts:
                continue
            st = min(sts, key=lambda s: abs(s["t"] - s0)); t = int(st["t"])
        recs = [MC.load_service_record(os.path.join(d, "svc", f"svc_t{q}.npz")) for q in range(t + 1)]
        fm = TF.FrontierMap(card)
        for r in recs:
            fm.add_record(r)
        rec = recs[-1]
        pts = np.concatenate([r["sample_points"] for r in recs])
        recs_xy = C2.rt_to_zup_m(pts)[:, :2]
        cam_xy = C2.rt_to_zup_m(rec["cam_center"])[:2]
        traj_xy = C2.rt_to_zup_m(np.stack([r["cam_center"] for r in recs]))[:, :2]
        cand_xy = C2.rt_to_zup_m(rec["legal_center"])[:, :2]
        tr = 10 * (t // 10)
        region = C2.region_box_m(card, recs[tr]["cam_center"], C2.N_WIN)
        if grp == "C1":
            step = e["steps"][t]; pl = [json.loads(x) for x in open(os.path.join(d, "planner.jsonl"))][t]
            pred, true, pick, cand = np.asarray(pl["scores"], float), np.asarray(step["diag"]["gain"], float), step["pick"], step["diag"]["cand"]
            share, rem = step["diag"]["remaining_in_region_share"], step["diag"]["remaining"]
        else:
            cj = json.load(open(os.path.join(WA, st["sid"] + ".json")))
            pred, true, cand = np.asarray(cj["models"]["B2"]["distp1"], float), np.asarray(cj["oracle_gain"], float), [int(x) for x in st["cand"]]
            import ab_data as D
            pick = int(D.pick_indoor(pred, cand, "B2", tag, t)[0])        # B2's own choice on this state (the trajectory is B0's)
            share, rem = None, None
        best = int(cand[int(np.argmax(true[cand]))])
        fp, finfo = fm.progress(rec, rec["legal_R"], rec["legal_T"])
        gstar = float(true[cand].max() * 100)
        kind = ("A 排错" if (gstar > 0.1 and pick != best and true[pick] < true[best] - 1e-4) else ("C/D 候选皆低" if gstar < 0.05 else "E greedy"))
        if share is not None and share < 0.3 and gstar < 0.1:
            kind = "B 收益在区域外"
        fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.6, 1]})
        topdown(axs[0], recs_xy, cam_xy, traj_xy, cand_xy, pred, true, pick, best, (region[0, :2], region[1, :2]), fm,
                f"{grp} {tag} t={t}: cov {e['coverage'][t]:.3f}; stall [{s0},{e0}] ({ln} moves); G* {gstar:.3f} pp; pick {pick} (gain {100 * true[pick]:.3f} pp); frontier dist {finfo['dist_to_frontier']}, reachable {finfo['n_reachable_frontier']}")
        ax = axs[1]
        ax.plot(e["coverage"], color="#1c5cab", lw=1.6, label=f"{grp} coverage")
        ax.axvspan(s0, e0, color="#d62728", alpha=0.15, label="stall window"); ax.axvline(t, color="k", lw=0.8, ls=":")
        for other, root2, col in (("B2", W3, "#7f7f7f"), ("C2", os.path.join(WC, "C2"), "#111111"), ("B0", os.path.join(A.CL_IN, "V2M"), "#86b6ef")):
            p2 = os.path.join(root2, tag, "episode.json")
            if os.path.exists(p2) and other != grp.split("@")[0] and not (grp == "B2@B0traj" and other == "B0"):
                ax.plot(json.load(open(p2))["coverage"], color=col, lw=1.0, ls="--", label=f"{other} coverage")
        ax.set_xlabel("move"); ax.set_ylabel("coverage (C_obs)"); ax.legend(fontsize=8); ax.set_title("subsequent coverage", loc="left", fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout(); name = f"e2_failure_{grp}_{tag}_t{t}.png"; fig.savefig(os.path.join(FIG, name), dpi=120); plt.close(fig)
        row = {"group": grp, "tag": tag, "t": t, "window": [s0, e0, ln], "coverage": float(e["coverage"][t]), "cam_xy_m": cam_xy.tolist(), "region_m": region[:, :2].tolist(), "n_cand": len(cand),
               "pred": pred.tolist(), "true_pp": (100 * true).tolist(), "pick": pick, "best": best, "gstar_pp": gstar, "frontier": finfo, "fp": fp, "remaining": rem, "remaining_in_region_share": share, "kind": kind, "figure": name}
        rows.append(row)
        L += [f"## 案例 {ci + 1}：{grp} · {tag} · t = {t}（停滞窗口 [{s0}, {e0}]，{ln} 步；类型 **{kind}**）", "", f"![]({os.path.join('..', 'figures', name)})", "",
              "| 项 | 值 |", "|---|---|",
              f"| 1 当前覆盖 | {e['coverage'][t]:.3f}（窗口末 {e['coverage'][e0]:.3f}；Final {e['coverage'][-1]:.3f}）|",
              f"| 2 相机位置（m，z-up xy） | ({cam_xy[0]:.1f}, {cam_xy[1]:.1f})，位姿 {list(map(int, rec['pose_idx']))} |",
              f"| 3 历史 P1 | 图左：记录 0..{t} 的 10% 采样点（灰）；{len(pts):,} 点 |",
              f"| 4 评分区域 | n_win = 5 框（红虚线）x [{region[0, 0]:.1f}, {region[1, 0]:.1f}] y [{region[0, 1]:.1f}, {region[1, 1]:.1f}] m" + (f"；剩余未覆盖表面在区域内 {share:.2f}，剩余 {100 * rem:.1f}%" if share is not None else "") + " |",
              f"| 5 候选 | {len(rec['legal_idx'])} 合法，候选集 {cand} |",
              f"| 6 预测增益（{grp} 分数） | " + "，".join(f"{j}:{pred[j]:.0f}" for j in range(len(pred))) + " |",
              f"| 7 真实增益 pp（离线） | " + "，".join(f"{j}:{100 * true[j]:.3f}" for j in range(len(true))) + f"；G\\* {gstar:.3f} |",
              f"| 8 frontier | 可达 frontier {finfo['n_reachable_frontier']} / {finfo['n_frontier']}，最近距离 {finfo['dist_to_frontier']} 格；候选 FP {[round(x, 1) for x in fp]} |",
              f"| 9 规划器选择 | 选 {pick}（真实 {100 * true[pick]:.3f} pp）；oracle 最优 {best}（{gstar:.3f} pp） |",
              f"| 10 后续 | 图右：覆盖曲线与停滞窗口；轨迹见图左 |", ""]
    json.dump(rows, open(os.path.join(OUT, "e2_failure_cases.json"), "w"), indent=1, default=float)
    L += ["## 类型汇总", "", "| 类型 | 案例数 |", "|---|---|"] + [f"| {k} | {sum(r['kind'] == k for r in rows)} |" for k in sorted({r["kind"] for r in rows})]
    open(os.path.join(REP, "E2_FAILURE_CASES.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps([(r["group"], r["tag"], r["t"], r["kind"], round(r["gstar_pp"], 3), r["frontier"]["dist_to_frontier"]) for r in rows]))


if __name__ == "__main__":
    main()
