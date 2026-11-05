"""EXP6 V3 / ablation E, Task 2 -- colosseum low-coverage plateau analysis (CPU; read-only inputs; generated tables + figures).
Inputs: CVPR Table 1 canonical records (official coverage, executed keys) of the colosseum episodes; planner step logs (H0 =
historical T1 root, H3 / D1 / YBEST-H3 = Table 1 g3 roots); the H pose table of colosseum (settings-grid index -> camera
centre / rotation, built with MAGICIAN's camera code); E-B oracle outputs (oracle/<model>/colosseum_s<k>_r0.{json,npz});
E-C branch and E-D replicate episodes (runs/branch/*, runs/rep/*).
Classification (PROTOCOL E-v1): low plateau = Final < 0.70; branch / replicate escaped = Final >= 0.75.
Outputs: results/colosseum_episodes.csv, results/oracle_steps.csv, results/plateau_summary.json, results/PLATEAU_TABLES.md,
FIGURES/colosseum_*.{png,pdf}."""
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(EX))))
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_E_stability_plateau")
G3 = os.path.join(R, "work_dirs", "cvpr_results", "table1", "g3", "closed_loop", "outdoor")
H0R = os.path.join(R, "work_dirs", "exp6_v3", "eval", "closed_loop", "outdoor", "T1", "V2M")
PT = np.load(os.path.join(R, "work_dirs", "exp6_v3", "H", "pose_tables", "colosseum.npz"))
LUT = {tuple(int(x) for x in r): j for j, r in enumerate(PT["idx"])}
S = "colosseum"
LOW, ESC = 0.70, 0.75
MAIN = {"H0": "V3", "H3": "V3-H3", "D1": "V3-D1-H3-NORISK", "YBEST-H3": "V3-YBEST-H3"}
EXTRA = {"H1": "V3-H1", "H2": "V3-H2", "YD": "V3-YD-COMBINED-Y50K-D1", "D2": "V3-D2-DEPTH3-B3", "D3": "V3-D3-DEPTH3-B5", "D4": "V3-D4-DEPTH4-B3",
         "Y50K+H0": "V3-Y50K-H0", "B0": "B0", "B2": "B2", "MAG-B": "MAG-B"}
COL = {"H0": "#6b6b6b", "H3": "#e67e22", "D1": "#2f6db5", "YBEST-H3": "#a61e4d", "D4": "#2a9d8f", "YD": "#0b7285"}
LOGROOT = {"H0": H0R, "H3": os.path.join(G3, "V3-H3", "V2M"), "D1": os.path.join(G3, "V3-D1-H3-NORISK", "V2M"), "YBEST-H3": os.path.join(G3, "V3-YBEST-H3", "V2M")}


def keyt(s):
    return tuple(int(x) for x in json.loads(s))


def centre(k):
    return PT["C"][LUT[keyt(k)]]


def forward(k):
    return PT["R"][LUT[keyt(k)]][:, 2]


def load_ep(mid, k):
    p = os.path.join(EP, mid, f"{S}_s{k}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def onset(c):
    c = np.asarray(c)
    return next((t for t in range(101) if c[min(t + 20, 100)] - c[t] < 0.02), None)


def move_stats(keys, c):
    P = np.asarray([keyt(x) for x in keys])
    pos = [tuple(r[:3]) for r in P]
    d = np.diff(np.asarray(c))
    revisit = sum(1 for t in range(1, len(pos)) if pos[t] in pos[:t]) / (len(pos) - 1)
    back = sum(1 for t in range(2, len(pos)) if pos[t] == pos[t - 2] and pos[t] != pos[t - 1]) / (len(pos) - 2)
    rot_only = sum(1 for t in range(1, len(pos)) if pos[t] == pos[t - 1]) / (len(pos) - 1)
    C = np.asarray([centre(x) for x in keys])
    out = {"distinct_positions": len(set(pos)), "distinct_poses": len(set(map(tuple, P))), "revisit_rate": revisit, "back_and_forth_rate": back,
           "rotation_only_rate": rot_only, "y_max_idx": int(P[:, 1].max()), "frac_steps_y_ge3": float((P[:, 1] >= 3).mean()),
           "x_extent": int(P[:, 0].max() - P[:, 0].min()), "z_extent": int(P[:, 2].max() - P[:, 2].min()),
           "path_len": float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())}
    for a, b in ((0, 20), (20, 50), (50, 100)):
        out[f"zero_gain_{a}_{b}"] = float((d[a:b] < 0.001).mean())
        out[f"y_ge3_{a}_{b}"] = float((P[a:b, 1] >= 3).mean())
        out[f"distinct_pos_{a}_{b}"] = len(set(pos[a:b]))
    return out


def first_div(a, b):
    return next((t for t, (x, y) in enumerate(zip(a, b)) if x != y), None)


def planner_log(name, k):
    p = os.path.join(LOGROOT[name], "planner_logs", f"pl_V2M_{S}_s{k}_r0.steps.jsonl")
    return {json.loads(l)["t"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}


def oracle(name, k):
    p = os.path.join(W, "oracle", MAIN[name], f"{S}_s{k}_r0.json")
    if not os.path.exists(p):
        return None, None
    return json.load(open(p)), np.load(p.replace(".json", ".npz"))


def unpack(npz, t, n):
    return np.unpackbits(npz[f"covered_t{t}"])[:n].astype(bool)


def runs(kind):
    out = []
    for f in sorted(glob.glob(os.path.join(W, "runs", kind, "*", "episodes", "ep_*.json"))):
        e = json.load(open(f))
        d = os.path.basename(os.path.dirname(os.path.dirname(f)))
        br = os.path.join(os.path.dirname(os.path.dirname(f)), "branch.json")
        out.append({"dir": d, "start": int(e["start"]), "rep": int(e["rep"]), "cov": e["coverage"], "keys": e["executed_keys"],
                    "branch": json.load(open(br)) if os.path.exists(br) else None})
    return out


def main():
    os.makedirs(os.path.join(EX, "results"), exist_ok=True)
    FIG = os.path.join(EX, "FIGURES")
    os.makedirs(FIG, exist_ok=True)
    allm = {**MAIN, **EXTRA}
    EPS = {(n, k): load_ep(m, k) for n, m in allm.items() for k in range(5)}
    SUM = {"episodes": [], "oracle": {}, "region": {}, "branch": [], "rep": [], "divergence": []}
    # ---------------------------------------------------------------- A: per-episode table
    rows = []
    for (n, k), e in EPS.items():
        if e is None:
            continue
        c = e["coverage_official"]
        r = {"policy": n, "start": k, "cov20": c[20], "cov50": c[50], "final": c[100], "low_plateau": c[100] < LOW, "onset": onset(c),
             "div_vs_H0": first_div(e["executed_keys"], EPS[("H0", k)]["executed_keys"]),
             "div_vs_H3": first_div(e["executed_keys"], EPS[("H3", k)]["executed_keys"])}
        if n not in ("MAG-B",):
            r.update(move_stats(e["executed_keys"], c))
        rows.append(r)
    SUM["episodes"] = rows
    keys_csv = sorted({kk for r in rows for kk in r})
    with open(os.path.join(EX, "results", "colosseum_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "start"] + [x for x in keys_csv if x not in ("policy", "start")])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # ---------------------------------------------------------------- B: oracle per-step
    orows = []
    for n in MAIN:
        for k in range(5):
            o, z = oracle(n, k)
            if o is None:
                continue
            L = planner_log(n, k)
            e = EPS[(n, k)]
            low = e["coverage_official"][100] < LOW
            on = onset(e["coverage_official"])
            for st in o["steps"]:
                t = st["t"]
                ca = st["cands"]
                lg = L.get(t, {})
                a = lg.get("all_scores") or {}
                dc = np.asarray([c["dcov_pp"] for c in ca])
                gv = np.asarray([c["gt_value"] for c in ca])
                gvr = np.asarray([c.get("gt_value_rel", np.nan) for c in ca])
                pol = np.asarray(lg.get("scores") or [np.nan] * len(ca), float)
                hm = np.asarray(a.get("h_mean") or [np.nan] * len(ca), float)
                hs = np.asarray(a.get("h_std") or [np.nan] * len(ca), float)
                dp = np.asarray(a.get("h0_distp1") or a.get("distp1") or [np.nan] * len(ca), float)
                ex = st["executed"]
                best = float(dc.max())
                r = {"policy": n, "start": k, "t": t, "low_episode": low, "phase": "pre" if on is None or t < on else "plateau", "onset": on, "n": len(ca),
                     "cov": st["cov"], "max_dcov": best, "chosen_dcov": float(dc[ex]), "top1": bool(dc[ex] >= best - 1e-9),
                     "gain_frac": float(dc[ex] / best) if best > 1e-6 else np.nan, "missed_pp": best - float(dc[ex]),
                     "n_cands_ge05": int((dc >= 0.5).sum()), "chosen_gt_value": float(gv[ex]), "chosen_gt_value_rel": float(gvr[ex]),
                     "chosen_gt_value_irr": float(ca[ex].get("gt_value_irr", np.nan)), "chosen_pred_h_mean": float(hm[ex]), "chosen_pred_h_std": float(hs[ex]),
                     "chosen_policy_score": float(pol[ex]),
                     "gtvalue_argmax_top1": bool(dc[int(np.argmax(gv))] >= best - 1e-9),
                     "gtvalue_rel_argmax_top1": bool(dc[int(np.nanargmax(gvr))] >= best - 1e-9) if np.isfinite(gvr).any() else None,
                     "hmean_argmax_top1": bool(dc[int(np.nanargmax(hm))] >= best - 1e-9) if np.isfinite(hm).any() else None,
                     "distp1_argmax_top1": bool(dc[int(np.nanargmax(dp))] >= best - 1e-9) if np.isfinite(dp).any() else None}
                if len(ca) >= 3:
                    for nm, v in (("rho_policy_dcov", pol), ("rho_hmean_dcov", hm), ("rho_distp1_dcov", dp), ("rho_gtvalue_dcov", gv), ("rho_hmean_gtvalue", hm)):
                        ref = gv if nm == "rho_hmean_gtvalue" else dc
                        ok = np.isfinite(v)
                        r[nm] = float(spearmanr(v[ok], ref[ok])[0]) if ok.sum() >= 3 and np.ptp(v[ok]) > 0 and np.ptp(ref[ok]) > 0 else np.nan
                if "two_union_value" in ca[ex]:
                    r.update({"c2_legal": ca[ex].get("c2_legal"), "chosen_two_dcov": ca[ex].get("two_dcov_pp"), "chosen_two_union_gt": ca[ex].get("two_union_value"),
                              "chosen_pred_q": float(pol[ex])})
                    leg = [c.get("c2_legal") for c in ca if "c2_legal" in c]
                    r["c2_legal_frac_all"] = float(np.mean(leg)) if leg else np.nan
                orows.append(r)
    if orows:
        ks = sorted({kk for r in orows for kk in r})
        with open(os.path.join(EX, "results", "oracle_steps.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["policy", "start", "t"] + [x for x in ks if x not in ("policy", "start", "t")])
            w.writeheader()
            for r in orows:
                w.writerow(r)

    def agg(sel):
        if not sel:
            return None
        g = lambda q: float(np.nanmean([r[q] for r in sel if r.get(q) is not None])) if any(r.get(q) is not None for r in sel) else None  # noqa: E731
        sp, sg = [r["chosen_pred_h_mean"] for r in sel], [r["chosen_gt_value"] for r in sel]
        return {"steps": len(sel), "max_dcov": g("max_dcov"), "chosen_dcov": g("chosen_dcov"), "missed_pp": g("missed_pp"), "top1": g("top1"),
                "frac_steps_best_ge05": float(np.mean([r["max_dcov"] >= 0.5 for r in sel])), "frac_steps_best_ge1": float(np.mean([r["max_dcov"] >= 1.0 for r in sel])),
                "pred_over_gt_value": float(np.nansum(sp) / max(1e-9, np.nansum(sg))) if np.isfinite(sp).any() else None,
                "pred_over_gt_value_rel": float(np.nansum(sp) / max(1e-9, np.nansum([r["chosen_gt_value_rel"] for r in sel]))) if np.isfinite(sp).any() else None,
                "irr_share_of_chosen_gt_value": float(np.nansum([r["chosen_gt_value_irr"] for r in sel]) / max(1e-9, np.nansum(sg))),
                "rho_policy_dcov": g("rho_policy_dcov"), "rho_hmean_dcov": g("rho_hmean_dcov"), "rho_gtvalue_dcov": g("rho_gtvalue_dcov"),
                "rho_hmean_gtvalue": g("rho_hmean_gtvalue"), "rho_distp1_dcov": g("rho_distp1_dcov"),
                "gtvalue_argmax_top1": g("gtvalue_argmax_top1"), "gtvalue_rel_argmax_top1": g("gtvalue_rel_argmax_top1"), "hmean_argmax_top1": g("hmean_argmax_top1"),
                "c2_legal": g("c2_legal"), "chosen_two_dcov": g("chosen_two_dcov"), "chosen_pred_q": g("chosen_pred_q"), "chosen_two_union_gt": g("chosen_two_union_gt"),
                "c2_legal_frac_all": g("c2_legal_frac_all")}
    for n in MAIN:
        for grp, f_ in (("low_plateau_phase", lambda r: r["low_episode"] and r["phase"] == "plateau"), ("low_pre_phase", lambda r: r["low_episode"] and r["phase"] == "pre"),
                        ("ok_steps_0_50", lambda r: (not r["low_episode"]) and r["t"] < 50), ("ok_steps_50_100", lambda r: (not r["low_episode"]) and r["t"] >= 50),
                        ("all", lambda r: True)):
            SUM["oracle"][f"{n} | {grp}"] = agg([r for r in orows if r["policy"] == n and f_(r)])
    # ---------------------------------------------------------------- C: region (per start; reference = H0 at t = 100)
    for k in range(5):
        o0, z0 = oracle("H0", k)
        if o0 is None:
            continue
        gp = z0["gt_points"]
        N = len(gp)
        ref = unpack(z0, 100, N)
        reg = {"n_gt": N, "H0_final_mask_mean": float(ref.mean())}
        for n in ("H3", "D1", "YBEST-H3"):
            o, z = oracle(n, k)
            if o is None:
                continue
            m = unpack(z, 100, N)
            assert np.allclose(z["gt_points"], gp), "GT points differ between episodes of the same start"
            miss = ref & ~m
            extra = m & ~ref
            reg[n] = {"final": EPS[(n, k)]["coverage_official"][100], "missing_vs_H0_frac": float(miss.mean()), "extra_vs_H0_frac": float(extra.mean())}
            if miss.sum() > 0:
                P = gp[miss]
                reg[n].update({"missing_centroid": P.mean(0).tolist(), "missing_y_q": np.quantile(P[:, 1], [0.1, 0.5, 0.9]).tolist(),
                               "missing_r_norm_q": np.quantile(np.sqrt((P[:, 0] / 52.0) ** 2 + (P[:, 2] / 43.0) ** 2), [0.1, 0.5, 0.9]).tolist()})
            # when did the H0 episode cover the region that this policy misses?
            reg[n]["H0_cover_of_missing_by_t"] = {t: float(unpack(z0, t, N)[miss].mean()) if miss.any() else None for t in range(10, 101, 10)}
        SUM["region"][k] = reg
    # ---------------------------------------------------------------- D / E: branches and replicates
    for r in runs("branch"):
        b = r["branch"]
        c = r["cov"]
        src_final = None
        if b:
            src = json.load(open(b["source_episode"]))
            src_final = src["coverage"][100]
            assert src["executed_keys"][:b["T"]] == r["keys"][:b["T"]], ("branch prefix differs", r["dir"])
        SUM["branch"].append({"dir": r["dir"], "start": r["start"], "T": b["T"] if b else None, "source_final": src_final, "cov50": c[50], "final": c[100],
                              "escaped": c[100] >= ESC, "low": c[100] < LOW, "div_from_source_at": first_div(r["keys"], src["executed_keys"]) if b else None})
    for r in runs("rep"):
        c = r["cov"]
        SUM["rep"].append({"policy": r["dir"], "start": r["start"], "rep": r["rep"], "cov50": c[50], "final": c[100], "low": c[100] < LOW, "onset": onset(c)})
    # ---------------------------------------------------------------- F: prediction audit (e_predict.py outputs, any scene)
    prow = []
    for f in sorted(x for x in glob.glob(os.path.join(W, "oracle", "*", "*_r0_predict.jsonl")) if "/smoke/" not in x):
        mid = os.path.basename(os.path.dirname(f))
        nm = next((n for n, m in MAIN.items() if m == mid), mid)
        base = os.path.basename(f).replace("_r0_predict.jsonl", "")
        scene, k = base.rsplit("_s", 1)
        k = int(k)
        rec = load_ep(mid, k) if scene == S else json.load(open(os.path.join(EP, mid, f"{scene}_s{k}.json")))
        c = rec["coverage_official"]
        low = c[100] < LOW if scene == S else None
        on = onset(c)
        for line in open(f):
            r = json.loads(line)
            ca = r["cands"]
            if not ca or "v_inbox" not in ca[0]:
                continue
            dc = np.asarray([x["true_dcov_pp"] for x in ca])
            best = dc.max()
            hm = np.asarray([x["pred_value"] for x in ca])
            vb = np.asarray([x["v_inbox"] for x in ca])
            ex = r["executed"]
            row = {"policy": nm, "scene": scene, "start": k, "t": r["t"], "low_episode": low, "phase": "pre" if on is None or r["t"] < on else "plateau",
                   "pred": ca[ex]["pred_value"], "hit": ca[ex]["v_hit_tol"], "seen": ca[ex]["v_seen_tol"], "empty": ca[ex]["v_empty_tol"], "inbox": ca[ex]["v_inbox"],
                   "prec": ca[ex]["prec_tol"], "recall": ca[ex]["recall_tol"], "true": ca[ex]["true_value"], "true_rel_share": ca[ex]["true_rel_share"],
                   "all_pred": float(hm.sum()), "all_inbox": float(vb.sum()),
                   "top1_hmean": bool(dc[int(np.argmax(hm))] >= best - 1e-9) if hm.max() > 0 else None,
                   "top1_inbox": bool(dc[int(np.argmax(vb))] >= best - 1e-9) if vb.max() > 0 else None,
                   "gain_hmean": float(dc[int(np.argmax(hm))]), "gain_inbox": float(dc[int(np.argmax(vb))]), "best": float(best)}
            if len(ca) >= 3:
                for nm2, v in (("rho_hmean", hm), ("rho_inbox", vb)):
                    row[nm2] = float(spearmanr(v, dc)[0]) if np.ptp(v) > 0 and np.ptp(dc) > 0 else np.nan
            prow.append(row)
    SUM["prediction"] = {}

    def pagg(sel):
        if not sel:
            return None
        P_ = lambda q: float(np.nansum([r[q] for r in sel]))  # noqa: E731
        M_ = lambda q: float(np.nanmean([np.nan if r.get(q) is None else float(r[q]) for r in sel]))  # noqa: E731
        return {"steps": len(sel), "hit_share": P_("hit") / max(1e-9, P_("pred")), "seen_share": P_("seen") / max(1e-9, P_("pred")), "empty_share": P_("empty") / max(1e-9, P_("pred")),
                "inbox_share_chosen": P_("inbox") / max(1e-9, P_("pred")), "inbox_share_all_cands": P_("all_inbox") / max(1e-9, P_("all_pred")),
                "pred_over_true": P_("pred") / max(1e-9, P_("true")), "prec": M_("prec"), "recall": M_("recall"), "true_rel_share": M_("true_rel_share"),
                "top1_hmean": M_("top1_hmean"), "top1_inbox": M_("top1_inbox"), "gain_hmean": M_("gain_hmean"), "gain_inbox": M_("gain_inbox"), "best": M_("best"),
                "rho_hmean": M_("rho_hmean"), "rho_inbox": M_("rho_inbox")}
    groups = [("colosseum 低平台 episode 的平台阶段", lambda r: r["scene"] == S and r["low_episode"] and r["phase"] == "plateau"),
              ("colosseum 低平台 episode 的平台前", lambda r: r["scene"] == S and r["low_episode"] and r["phase"] == "pre"),
              ("colosseum 成功 episode t < 50", lambda r: r["scene"] == S and r["low_episode"] is False and r["t"] < 50),
              ("colosseum 成功 episode t ≥ 50", lambda r: r["scene"] == S and r["low_episode"] is False and r["t"] >= 50),
              ("pisa（全部步）", lambda r: r["scene"] == "pisa"), ("alhambra（全部步）", lambda r: r["scene"] == "alhambra"),
              ("全部（所有场景、所有步）", lambda r: True), ("全部 t < 20", lambda r: r["t"] < 20)]
    for gname, f_ in groups:
        SUM["prediction"][gname] = pagg([r for r in prow if f_(r)])
        for n in MAIN:
            SUM["prediction"][f"{gname} | {n}"] = pagg([r for r in prow if f_(r) and r["policy"] == n])
    json.dump(SUM, open(os.path.join(EX, "results", "plateau_summary.json"), "w"), indent=1, default=float)

    # ---------------------------------------------------------------- tables
    L = ["# PLATEAU_TABLES — colosseum 诊断自动生成表（`scripts/plateau_analysis.py`）", "",
         f"- 官方覆盖；低平台 = Final < {LOW}；分支或重复续跑“脱困” = Final ≥ {ESC}（PROTOCOL E-v1）。",
         "- 平台开始步 = 第一个满足 C[t+20] − C[t] < 0.02 的 t。", "", "## A. 逐 episode", "",
         "| 策略 | 起点 | Cov@20 | Cov@50 | Final | 低平台 | 平台开始步 | 与 H0 首次分叉 | 不同位置数 | 重复访问率 | 往返率 | 原地转向率 | 最高 y 层 | y≥3 步比例（0–20 / 20–50 / 50–100） | x / z 范围 | 零增益步比例（20–50 / 50–100） |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (list(allm).index(r["policy"]), r["start"])):
        if "distinct_positions" not in r:
            L.append(f"| {r['policy']} | {r['start']} | {r['cov20']:.3f} | {r['cov50']:.3f} | {r['final']:.3f} | {'是' if r['low_plateau'] else ''} | {r['onset']} | {r['div_vs_H0']} | | | | | | | | |")
            continue
        L.append(f"| {r['policy']} | {r['start']} | {r['cov20']:.3f} | {r['cov50']:.3f} | {r['final']:.3f} | {'是' if r['low_plateau'] else ''} | {r['onset']} | {r['div_vs_H0']} | "
                 f"{r['distinct_positions']} | {r['revisit_rate']:.2f} | {r['back_and_forth_rate']:.2f} | {r['rotation_only_rate']:.2f} | {r['y_max_idx']} | "
                 f"{r['y_ge3_0_20']:.2f} / {r['y_ge3_20_50']:.2f} / {r['y_ge3_50_100']:.2f} | {r['x_extent']} / {r['z_extent']} | {r['zero_gain_20_50']:.2f} / {r['zero_gain_50_100']:.2f} |")
    lowc = {n: sum(1 for r in rows if r["policy"] == n and r["low_plateau"]) for n in allm}
    L += ["", "低平台 episode 数（共 5）：" + "，".join(f"{n} {lowc[n]}" for n in allm), ""]
    L += ["## B. GT 反事实（每步所有合法候选）", "",
          "- 字段：",
          "  - max_dcov = 当步最佳候选的真实官方覆盖增益（pp）；Top-1 = 选中动作就是最佳候选；漏掉增益 = 最佳 − 选中（pp）；",
          "  - pred/GT（相关）= 选中动作的预测 h_mean 之和 / 其真实视图 value 中落在 GT 评估点 1.0 以内部分之和。规划器的预测只在规划域（≈ 场景盒）内计分，所以与之可比的是“相关”部分；",
          "  - 无关占比 = 真实视图 value 中不在 GT 评估点附近的部分（几乎全在场景盒外，规划器不会计入）；",
          "  - ρ = 当步候选间 Spearman 秩相关（均值）；“GT value 选最大的 Top-1” = 用真实未来视图的 value 选动作时的 Top-1（完美预测下的评分规则上限）。", "",
          "| 策略 | 步集合 | 步数 | 最佳一步增益 pp | 选中增益 pp | 漏掉增益 pp | Top-1 | 最佳 ≥ 0.5 pp 的步比例 | pred/GT（相关） | 无关占比 | ρ(策略分, dcov) | ρ(h_mean, dcov) | ρ(GT value, dcov) | ρ(h_mean, GT value) | GT value（不限域）选最大的 Top-1 | GT 相关 value 选最大的 Top-1（完美预测上限） | c2* 合法比例（选中 / 全部） | 选中的两步真实增益 pp |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda x: "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"  # noqa: E731
    for key, a in SUM["oracle"].items():
        if a is None:
            continue
        n, grp = key.split(" | ")
        L.append(f"| {n} | {grp} | {a['steps']} | {f2(a['max_dcov'])} | {f2(a['chosen_dcov'])} | {f2(a['missed_pp'])} | {f2(a['top1'])} | {f2(a['frac_steps_best_ge05'])} | "
                 f"{f2(a['pred_over_gt_value_rel'])} | {f2(a['irr_share_of_chosen_gt_value'])} | {f2(a['rho_policy_dcov'])} | {f2(a['rho_hmean_dcov'])} | {f2(a['rho_gtvalue_dcov'])} | "
                 f"{f2(a['rho_hmean_gtvalue'])} | {f2(a['gtvalue_argmax_top1'])} | {f2(a['gtvalue_rel_argmax_top1'])} | {f2(a['c2_legal'])} / {f2(a['c2_legal_frac_all'])} | {f2(a['chosen_two_dcov'])} |")
    L += ["", "## C. 缺失区域（相对同起点 H0 在第 100 步已覆盖、本策略未覆盖的 GT 点）", "",
          "| 起点 | 策略 | Final | 缺失占 GT 比例 | 多出占比 | 缺失点 y 分位（10/50/90） | 缺失点归一化半径分位（10/50/90） | H0 覆盖该缺失区域的进度（t = 10 / 20 / 30 / 50） |", "|---|---|---|---|---|---|---|---|"]
    for k, reg in SUM["region"].items():
        for n in ("H3", "D1", "YBEST-H3"):
            if n not in reg:
                continue
            x = reg[n]
            prog = x["H0_cover_of_missing_by_t"]
            L.append(f"| {k} | {n} | {x['final']:.3f} | {x['missing_vs_H0_frac']:.3f} | {x['extra_vs_H0_frac']:.3f} | "
                     + (" / ".join(f"{v:.1f}" for v in x["missing_y_q"]) if "missing_y_q" in x else "—") + " | "
                     + (" / ".join(f"{v:.2f}" for v in x["missing_r_norm_q"]) if "missing_r_norm_q" in x else "—") + " | "
                     + " / ".join(f2(prog[t]) for t in (10, 20, 30, 50)) + " |")
    L += ["", "## D. 分支续跑（state swap）", "", "| 分支 | 起点 | T | 源 Final | 续跑 Cov@50 | 续跑 Final | 脱困（≥ 0.75） | 与源首次不同的步 |", "|---|---|---|---|---|---|---|---|"]
    for b in sorted(SUM["branch"], key=lambda b: b["dir"]):
        L.append(f"| {b['dir']} | {b['start']} | {b['T']} | {f2(b['source_final'])} | {b['cov50']:.3f} | {b['final']:.3f} | {'是' if b['escaped'] else ''} | {b['div_from_source_at']} |")
    L += ["", "## E. 噪声重复（rep 1 / 2；rep 0 = Table 1）", "", "| 策略 | 起点 | rep 0 Final | rep 1 Final | rep 2 Final |", "|---|---|---|---|---|"]
    repname = {"H0": "H0", "H3": "H3", "D1": "D1", "YBEST-H3": "YBEST-H3"}
    for n in repname:
        for k in range(5):
            v = {r["rep"]: r["final"] for r in SUM["rep"] if r["policy"] == n and r["start"] == k}
            v[0] = EPS[(n, k)]["coverage_official"][100]
            L.append(f"| {n} | {k} | " + " | ".join(f2(v.get(rr)) for rr in (0, 1, 2)) + " |")
        done = [r for r in SUM["rep"] if r["policy"] == n]
        L.append(f"| {n} | 低平台次数 | {lowc[n]}/5 | {sum(1 for r in done if r['rep'] == 1 and r['low'])}/{sum(1 for r in done if r['rep'] == 1)} | {sum(1 for r in done if r['rep'] == 2 and r['low'])}/{sum(1 for r in done if r['rep'] == 2)} |")
    L += ["", "## F. 预测审计（离线重建规划器预测，e_predict.py；选中动作的预测值分解，容差 2.0）", "",
          "- 命中 = 预测新增 cell 在真实视图新增 cell 2.0 以内；已见 = 在 GT 评估点附近但真实视图不新增；空 = 2.0 内无 GT 评估点（地面、盒外地形或幻影表面）。",
          "- 盒内占比 = 预测值中 cell 落在 MAGICIAN settings.json 场景盒内的比例（选中动作 / 所有候选）。",
          "- 离线一步重打分：只按预测 h_mean 选 vs 只按盒内预测值选，与 GT 一步增益比较（Top-1、所选动作的真实增益 pp、ρ）。这只是同一预测的不同计分，不是闭环结果。", "",
          "| 步集合 | 策略 | 步数 | 命中 | 已见 | 空 | 盒内（选中 / 全部） | 预测/真实 value | 真实视图相关占比 | Top-1：h_mean → 盒内 | 所选真实增益 pp：h_mean → 盒内（最佳） | ρ：h_mean → 盒内 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, a in SUM["prediction"].items():
        if a is None:
            continue
        g_, n_ = (key.split(" | ") + ["全部策略"])[:2]
        L.append(f"| {g_} | {n_} | {a['steps']} | {a['hit_share']:.2f} | {a['seen_share']:.2f} | {a['empty_share']:.2f} | {a['inbox_share_chosen']:.2f} / {a['inbox_share_all_cands']:.2f} | "
                 f"{a['pred_over_true']:.2f} | {f2(a['true_rel_share'])} | {f2(a['top1_hmean'])} → {f2(a['top1_inbox'])} | {f2(a['gain_hmean'])} → {f2(a['gain_inbox'])}（{f2(a['best'])}） | {f2(a['rho_hmean'])} → {f2(a['rho_inbox'])} |")
    open(os.path.join(EX, "results", "PLATEAU_TABLES.md"), "w").write("\n".join(L) + "\n")

    # ---------------------------------------------------------------- figures
    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for k in range(5):
        for n in ("D4", "YD"):
            ax[k].plot(EPS[(n, k)]["coverage_official"], color=COL[n], lw=0.8, ls=":", label=n if k == 0 else None)
        for n in MAIN:
            ax[k].plot(EPS[(n, k)]["coverage_official"], color=COL[n], lw=1.6, label=n if k == 0 else None)
        ax[k].axvspan(20, 50, color="#f2e6c9", alpha=0.5)
        ax[k].axhline(LOW, color="#999", lw=0.6, ls="--")
        ax[k].set_title(f"colosseum start {k}")
        ax[k].set_xlabel("step")
        ax[k].grid(alpha=0.3)
    ax[0].set_ylabel("official coverage")
    ax[0].legend(fontsize=7, loc="lower right")
    save(fig, "colosseum_coverage_5starts")
    fig, ax = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for k in range(5):
        for n in MAIN:
            ax[k].plot(range(20, 51), EPS[(n, k)]["coverage_official"][20:51], color=COL[n], lw=1.6, label=n if k == 0 else None, marker=".", ms=3)
        ax[k].set_title(f"start {k}: steps 20-50")
        ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    save(fig, "colosseum_coverage_20_50")
    # trajectories (top-down x-z) with GT points of the start (from the H0 oracle) and the region missed by H3
    fig, ax = plt.subplots(5, 4, figsize=(16, 20))
    for k in range(5):
        o0, z0 = oracle("H0", k)
        gp = z0["gt_points"] if z0 is not None else None
        miss = None
        if z0 is not None:
            o3, z3 = oracle("H3", k)
            if z3 is not None:
                miss = unpack(z0, 100, len(gp)) & ~unpack(z3, 100, len(gp))
        for j, n in enumerate(MAIN):
            a = ax[k, j]
            if gp is not None:
                a.scatter(gp[::7, 0], gp[::7, 2], s=0.3, color="#cfcfcf")
                if miss is not None and miss.any():
                    a.scatter(gp[miss][::3, 0], gp[miss][::3, 2], s=0.5, color="#f4a3a3")
            e = EPS[(n, k)]
            C = np.asarray([centre(x) for x in e["executed_keys"]])
            P = np.asarray([keyt(x) for x in e["executed_keys"]])
            sc = a.scatter(C[:, 0], C[:, 2], c=np.arange(len(C)), cmap="viridis", s=8 + 10 * P[:, 1], zorder=3)
            a.plot(C[:, 0], C[:, 2], color="#555", lw=0.4, zorder=2)
            for t in (20, 50):
                a.scatter(C[t, 0], C[t, 2], marker="x", color="k", s=40, zorder=4)
                a.annotate(str(t), (C[t, 0], C[t, 2]), fontsize=7)
            dv = first_div(e["executed_keys"], EPS[("H0", k)]["executed_keys"])
            a.set_title(f"s{k} {n}: Final {e['coverage_official'][100]:.3f}, div vs H0 @ {dv}", fontsize=9)
            a.set_xlim(-70, 70)
            a.set_ylim(-70, 70)
            a.set_aspect("equal")
    fig.suptitle("Camera trajectories (top view x-z; colour = step, size = height level; grey = GT points; pink = GT covered by H0 but not by H3 at t=100)", fontsize=10)
    save(fig, "colosseum_trajectories")
    # height level over time
    fig, ax = plt.subplots(1, 5, figsize=(20, 3.6), sharey=True)
    for k in range(5):
        for n in MAIN:
            P = np.asarray([keyt(x) for x in EPS[(n, k)]["executed_keys"]])
            ax[k].step(range(len(P)), P[:, 1] + {"H0": -0.09, "H3": -0.03, "D1": 0.03, "YBEST-H3": 0.09}[n], color=COL[n], lw=1.2, label=n if k == 0 else None)
        ax[k].set_title(f"start {k}: camera height level (0 = 4, 4 = 36 runtime units)")
        ax[k].axvspan(20, 50, color="#f2e6c9", alpha=0.5)
        ax[k].set_xlabel("step")
    ax[0].legend(fontsize=7)
    save(fig, "colosseum_height_levels")
    # predicted vs GT gain for the executed action
    if orows:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        for n in MAIN:
            sel = [r for r in orows if r["policy"] == n]
            if not sel:
                continue
            for lowflag, mk in ((False, "o"), (True, "^")):
                s2 = [r for r in sel if r["low_episode"] == lowflag and r["phase"] == "plateau"] if lowflag else [r for r in sel if not r["low_episode"]]
                if not s2:
                    continue
                if n != "H0":
                    ax[0].scatter([r["chosen_pred_h_mean"] for r in s2], [r["chosen_gt_value_rel"] for r in s2], s=6, marker=mk, color=COL[n], alpha=0.5,
                                  label=f"{n} {'low-plateau steps' if lowflag else 'ok episodes'}")
                    ax[1].scatter([r["chosen_pred_h_mean"] for r in s2], [r["chosen_dcov"] for r in s2], s=6, marker=mk, color=COL[n], alpha=0.5)
        lim = ax[0].get_xlim()
        ax[0].plot(lim, lim, color="#999", lw=0.8)
        ax[0].set_xlabel("predicted value h_mean (chosen action)")
        ax[0].set_ylabel("value of the true view, GT-surface part (same definition)")
        ax[0].legend(fontsize=6)
        ax[1].set_xlabel("predicted value h_mean (chosen action)")
        ax[1].set_ylabel("true official coverage gain (pp)")
        for n in MAIN:
            for k in range(5):
                sel = sorted([r for r in orows if r["policy"] == n and r["start"] == k], key=lambda r: r["t"])
                if sel:
                    ax[2].plot([r["t"] for r in sel], [r["max_dcov"] for r in sel], color=COL[n], lw=0.6, alpha=0.7, ls="-" if not sel[0]["low_episode"] else "--")
        ax[2].set_yscale("symlog", linthresh=0.5)
        ax[2].set_xlabel("step")
        ax[2].set_ylabel("best one-step true gain among legal candidates (pp)")
        ax[2].set_title("dashed = low-plateau episodes")
        for a in ax:
            a.grid(alpha=0.3)
        save(fig, "colosseum_pred_vs_true_gain")
    print("ok", len(rows), "oracle steps", len(orows), "branches", len(SUM["branch"]), "reps", len(SUM["rep"]))


if __name__ == "__main__":
    main()
