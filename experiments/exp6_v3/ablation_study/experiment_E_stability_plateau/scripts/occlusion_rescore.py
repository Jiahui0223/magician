"""EXP6 V3 / ablation E -- offline re-scoring with "observed surfaces are opaque" (CPU, diagnostic only; no closed loop).
Online-computable variant of occlusion_check.py: at step t, the agent's OWN past observations (S10 sample points of frames
0..t from the service records, de-duplicated on 0.5 cells) are used as occluders. For every legal candidate (camera centre
from its service-record legal_center) every predicted novel cell (e_predict union cells) that has a past observed point
within ANG = 1.5 deg of its viewing direction and nearer by more than 2.0 is removed (the candidate cannot see through a
surface it has already observed). Re-scored value = remaining weight. Compared with the un-filtered union value and with
the planner's mean value against the true one-step coverage gain (GT used only to score): Top-1, true gain of the argmax
(pp), Spearman rho; removed share of the value. Groups as refute_analysis.py. Output results/occlusion_rescore.json + table."""
import glob
import json
import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(EX))))
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_E_stability_plateau")
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
M20 = (1 << 20) - 1
ANG = np.deg2rad(1.5)
CH = 2 * np.sin(ANG / 2)


def dec(ids):
    return np.stack([(ids >> 40) & M20, (ids >> 20) & M20, ids & M20], 1).astype(np.float64) - (1 << 19) + 0.5


def onset(c):
    return next((t for t in range(101) if c[min(t + 20, 100)] - c[t] < 0.02), None)


def visible(cells, cam, occ):
    if len(occ) == 0:
        return np.ones(len(cells), bool)
    vc = cells - cam
    dc = np.linalg.norm(vc, axis=1)
    vo = occ - cam
    do = np.linalg.norm(vo, axis=1)
    near = do < dc.max()
    if not near.any():
        return np.ones(len(cells), bool)
    uo, do = vo[near] / do[near, None], do[near]
    tree = cKDTree(uo)
    keep = np.ones(len(cells), bool)
    for i, js in enumerate(tree.query_ball_point(vc / dc[:, None], r=CH)):
        if js and do[js].min() < dc[i] - 2.0:
            keep[i] = False
    return keep


def main():
    rows = []
    for f in sorted(x for x in glob.glob(os.path.join(W, "oracle", "*", "*_r0_predict.jsonl")) if "/smoke/" not in x):
        mid = os.path.basename(os.path.dirname(f))
        scene, k = os.path.basename(f).replace("_r0_predict.jsonl", "").rsplit("_s", 1)
        k = int(k)
        rec = json.load(open(os.path.join(EP, mid, f"{scene}_s{k}.json")))
        c = rec["coverage_official"]
        ep = json.load(open(os.path.join(R, rec["episode_file"])))
        svc = os.path.join(os.path.dirname(os.path.dirname(os.path.join(R, rec["episode_file"]))), "svc", ep["tag"])
        z = np.load(f.replace(".jsonl", "_cells.npz"))
        idx = {(int(a), int(b)): (int(p), int(n)) for a, b, p, n in z["index"]}
        on = onset(c)
        low = (c[100] < 0.70) if scene == "colosseum" else None
        occ_keys = set()
        occ = []
        pr = {json.loads(l)["t"]: json.loads(l) for l in open(f)}
        for t in range(max(pr) + 1):
            s_ = np.load(os.path.join(svc, f"svc_{scene}_s{k}_t{t}.npz"))
            sp = s_["sample_points"].astype(np.float64).reshape(-1, 3)
            kk = np.floor(sp / 0.5).astype(np.int64)
            for p, key in zip(sp, map(tuple, kk)):
                if key not in occ_keys:
                    occ_keys.add(key)
                    occ.append(p)
            if t not in pr:
                continue
            r = pr[t]
            O = np.asarray(occ, np.float64).reshape(-1, 3)
            lc = s_["legal_center"].astype(np.float64).reshape(-1, 3)
            ca = r["cands"]
            if len(lc) != len(ca):
                continue
            dc = np.asarray([x["true_dcov_pp"] for x in ca])
            best = dc.max()
            vu, vv = [], []
            for j in range(len(ca)):
                off, n = idx[(t, j)]
                if n == 0:
                    vu.append(0.0); vv.append(0.0)
                    continue
                cells, w = dec(z["ids"][off:off + n]), z["w"][off:off + n]
                keep = visible(cells, lc[j], O)
                vu.append(float(w.sum())); vv.append(float(w[keep].sum()))
            vu, vv, pv = np.asarray(vu), np.asarray(vv), np.asarray([x["pred_value"] for x in ca])
            grp = (("colosseum low plateau" if (on is not None and t >= on) else "colosseum low pre") if low else
                   ("colosseum ok t<20" if t < 20 else "colosseum ok t>=20")) if scene == "colosseum" else (f"{scene} t<20" if t < 20 else f"{scene} t>=20")
            row = {"grp": grp, "early": t < 20, "best": float(best), "removed_share": float(1 - vv.sum() / vu.sum()) if vu.sum() > 0 else np.nan,
                   "removed_share_exec": float(1 - vv[r["executed"]] / vu[r["executed"]]) if vu[r["executed"]] > 0 else np.nan}
            for nm, v in (("pred", pv), ("union", vu), ("opaque", vv)):
                jj = int(np.argmax(v)) if v.max() > 0 else None
                row[f"top1_{nm}"] = (dc[jj] >= best - 1e-9) if jj is not None else None
                row[f"gain_{nm}"] = float(dc[jj]) if jj is not None else None
                row[f"rho_{nm}"] = float(spearmanr(v, dc)[0]) if len(v) >= 3 and np.ptp(v) > 0 and np.ptp(dc) > 0 else np.nan
            rows.append(row)
    out = {}
    for g in sorted({r["grp"] for r in rows}) + ["all t<20", "all t>=20"]:
        sel = [r for r in rows if r["grp"] == g or (g == "all t<20" and r["early"]) or (g == "all t>=20" and not r["early"])]
        m = lambda q: float(np.nanmean([np.nan if r.get(q) is None else float(r[q]) for r in sel]))  # noqa: E731
        out[g] = {"steps": len(sel), **{q: m(q) for q in ("best", "removed_share", "removed_share_exec", "top1_pred", "top1_union", "top1_opaque",
                                                          "gain_pred", "gain_union", "gain_opaque", "rho_pred", "rho_union", "rho_opaque")}}
    json.dump(out, open(os.path.join(EX, "results", "occlusion_rescore.json"), "w"), indent=1)
    print("| 步集合 | 步数 | 最佳一步增益 pp | 被去掉的预测值（所有候选 / 执行动作） | Top-1：原始 / 并集 / 不透明 | 所选真实增益 pp：原始 / 并集 / 不透明 | ρ：原始 / 并集 / 不透明 |")
    print("|---|---|---|---|---|---|---|")
    for g, a in out.items():
        print(f"| {g} | {a['steps']} | {a['best']:.2f} | {a['removed_share']:.2f} / {a['removed_share_exec']:.2f} | {a['top1_pred']:.2f} / {a['top1_union']:.2f} / {a['top1_opaque']:.2f} | "
              f"{a['gain_pred']:.2f} / {a['gain_union']:.2f} / {a['gain_opaque']:.2f} | {a['rho_pred']:.2f} / {a['rho_union']:.2f} / {a['rho_opaque']:.2f} |")


if __name__ == "__main__":
    main()
