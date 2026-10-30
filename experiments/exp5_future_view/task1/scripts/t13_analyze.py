"""EXP5 Task 1.3 analysis (CPU, evaluation side; descriptive): one-step ranking diagnostics of every scoring method on
S-a (73 T5-D states, true g_forward) and S-b (1,500 B0 closed-loop states, oracle_diag g), the G1 ES comparison
(only if the reproduction passed), U component / latent diagnostics and the timing table. Output W/analysis/t13.json
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402
import v2_common as LC  # noqa: E402

import numpy as np  # noqa: E402

import m_common as MC  # noqa: E402
from seen2scene.futureview import scoring as SCO  # noqa: E402

OUT = os.path.join(VF.WORK, "analysis")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return None

    def rk(x):
        r = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rk(a), rk(b))[0, 1])


def pick_of(sc, scene, start, t):
    s = np.asarray(sc, float)
    return MC.fallback_choice(scene, start, t, len(s)) if s.max() == 0 else MC.argmax_lowest(s)


def one_step(sc, g, scene, start, t, pick=None):
    g = np.asarray(g, float)
    pk = pick_of(sc, scene, start, t) if pick is None else pick
    return {"spearman": spearman(sc, g), "top1": float(g[pk] == g.max()), "regret": float(g.max() - g[pk]),
            "false_promise": float(np.asarray(sc, float)[pk] > 0 and g[pk] < 0.01), "dead": float(g.max() < 0.1),
            "fallback": float(np.asarray(sc, float).max() == 0)}


def boot_groups(vals, n=10000, seed=0):
    """vals: {group: [x, ...]} -> mean of group means and 95% bootstrap CI over groups."""
    gm = np.asarray([np.mean(v) for v in vals.values() if len(v)])
    if len(gm) == 0:
        return {"mean": None, "ci95": [None, None], "groups": 0}
    rng = np.random.default_rng(seed)
    bs = gm[rng.integers(0, len(gm), (n, len(gm)))].mean(1)
    return {"mean": float(gm.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "groups": int(len(gm))}


def methods(row, fv):
    d = fv["scoring"]["delta_vox"]
    lam = fv["scoring"]["lambda"]
    m = {"V2_DistP1_K4": row["distp1"], "V2_DistP1_K1": [x[0] for x in row["distp1_k"]],
         "V2_Uraw": row[f"u_raw@{d}"], "V2_DistU": row[f"dist_u@{d}"],
         "V2_Uraw_H": row[f"u_raw_H@{d}"], "V2_Uraw_D": row[f"u_raw_D@{d}"],
         "V2_DistU_H": row[f"dist_u_H@{d}"], "V2_DistU_D": row[f"dist_u_D@{d}"],
         "V2_Uflow": row["u_flow"], "Frontier_DistP1": row["frontier_distp1"]}
    if "u_feat" in row:
        m["V2_Ufeat"] = row["u_feat"]
    if "truedepth_distp1" in row:
        m["TrueDepth_DistP1"] = row["truedepth_distp1"]
    if max(row["distp1"]) == 0 and max(row[f"dist_u@{d}"]) == 0:
        m["V2_GU"] = [0.0] * len(row["distp1"])
    else:
        m["V2_GU"] = SCO.gu_scores(row["distp1"], row[f"dist_u@{d}"], lam)
    for dd in VF.load_cfg("scoring_v1.json")["U"]["delta_grid_vox"]:
        m[f"diag_Uraw@{dd}"] = row[f"u_raw@{dd}"]
        m[f"diag_DistU@{dd}"] = row[f"dist_u@{dd}"]
    return m


def summarize(per, group_of):
    out = {}
    for meth, rows in per.items():
        r = {}
        for met in ("spearman", "top1", "regret", "false_promise", "dead", "fallback"):
            vals = {}
            for key, x in rows:
                if x[met] is not None:
                    vals.setdefault(group_of(key), []).append(x[met])
            r[met] = boot_groups(vals)
        out[meth] = r
    return out


def sa(fv):
    import abl_common as AC
    import t3_evalside as EV
    rows = [json.loads(x) for p in sorted(glob.glob(os.path.join(VF.WORK, "fixed_state", "sa", "part*.jsonl"))) for x in open(p)]
    arch = {}
    for p in glob.glob(os.path.join(AC.ABL, "_fixed_state", "part_*.json")):
        for r in json.load(open(p)):
            arch[(r["scene"], r["start"], r["t"])] = r
    per = {}
    ts2 = sum(r["TS2_violations"] for r in rows)
    for r in rows:
        key = (r["scene"], r["start"], r["t"])
        g = EV.gains_for(r["scene"], r["start"], r["t"], r["keys"])
        ms = methods(r, fv)
        ms["B0_T1C_DistP1_archived"] = arch[key]["distp1"]
        for nm, sc in ms.items():
            per.setdefault(nm, []).append((key, one_step(sc, g, r["scene"], r["start"], r["t"])))
    out = summarize(per, lambda k: (k[0], k[1]))
    strata = {"t": stratify(per, lambda key: next(nm for nm, lo, hi in T_BINS if lo <= key[2] <= hi))}
    by_scene = {}
    for sc in sorted({r["scene"] for r in rows}):
        by_scene[sc] = {nm: float(np.nanmean([x["spearman"] if x["spearman"] is not None else np.nan for k, x in v if k[0] == sc]))
                        for nm, v in per.items() if not nm.startswith("diag")}
    return {"n_states": len(rows), "TS2_violations_total": ts2, "methods": out, "spearman_by_scene": by_scene, "strata": strata}


def sb(fv):
    per = {}
    rows = [json.loads(x) for p in sorted(glob.glob(os.path.join(VF.WORK, "fixed_state", "sb", "sb_*.jsonl"))) for x in open(p)]
    diag, plog = {}, {}
    for s in LC.SCENES:
        for k in LC.STARTS:
            tg = f"A2_{s}_s{k}_r0"
            dp = os.path.join(LC.B0_ROOT, "oracle_diag", f"diag_{tg}.json")
            if os.path.exists(dp):
                diag[(s, k)] = {x["t"]: x for x in json.load(open(dp))["steps"]}
            plog[(s, k)] = {x["t"]: x for x in json.load(open(os.path.join(LC.B0_ROOT, "planner_logs", f"pl_{tg}.json")))["steps"]}
    agree = []
    for r in rows:
        if r.get("n", 1) == 0 or "distp1" not in r:
            continue
        s, k, t = r["scene"], r["start"], r["t"]
        dg = diag.get((s, k), {}).get(t)
        if dg is None or "g" not in dg:
            continue
        g = dg["g"]
        ms = methods(r, fv)
        b0 = plog[(s, k)][t]
        if b0["scores"] is not None:
            ms["B0_own_scores"] = b0["scores"]
        for nm, sc in ms.items():
            per.setdefault(nm, []).append(((s, k, t), one_step(sc, g, s, k, t)))
        agree.append(float(pick_of(r["distp1"], s, k, t) == b0["choice"]["pick"]))
    out = summarize(per, lambda key: (key[0], key[1]))
    # protocol §4.6 strata (evaluation side): scene, t, GT coverage of the B0 episode at t, viewing distance of B0's
    # view at t (median camera -> observed sample point distance, the abl_1 definition)
    cov, vdist = {}, {}
    for s in LC.SCENES:
        for k in LC.STARTS:
            tg = f"A2_{s}_s{k}_r0"
            c = json.load(open(os.path.join(LC.B0_ROOT, "episodes", f"ep_{tg}.json")))["coverage"]
            for t in range(100):
                cov[(s, k, t)] = c[t]
                p = os.path.join(LC.B0_ROOT, "svc", tg, f"svc_{s}_s{k}_t{t}.npz")
                if os.path.exists(p):
                    z = np.load(p)
                    if len(z["sample_points"]) >= 20:
                        vdist[(s, k, t)] = float(np.median(np.linalg.norm(z["sample_points"] - z["cam_center"][None], axis=1)))
    labels = {"scene": lambda key: key[0],
              "t": lambda key: next(nm for nm, lo, hi in T_BINS if lo <= key[2] <= hi),
              "gt_coverage": lambda key: next(nm for nm, lo, hi in COV_BINS if lo <= cov[key] < hi),
              "view_dist": lambda key: ("unknown" if key not in vdist else
                                        next(nm for nm, lo, hi in DIST_BINS if lo <= vdist[key] < hi))}
    return {"n_states": len(rows), "pick_agreement_V2G_vs_B0": float(np.mean(agree)) if agree else None, "methods": out,
            "strata": {nm: stratify(per, f) for nm, f in labels.items()}}


T_BINS = [("t=0", 0, 0), ("t1-5", 1, 5), ("t6-20", 6, 20), ("t21-50", 21, 50), ("t51-99", 51, 99)]
COV_BINS = [("cov<0.20", 0, 0.20), ("cov0.20-0.50", 0.20, 0.50), ("cov0.50-0.80", 0.50, 0.80), ("cov>=0.80", 0.80, 2)]
DIST_BINS = [("d<10", 0, 10), ("d10-20", 10, 20), ("d20-40", 20, 40), ("d>=40", 40, 1e9)]
STRATA_METHODS = ("V2_DistP1_K4", "V2_DistP1_K1", "V2_Uraw", "V2_DistU", "V2_GU", "Frontier_DistP1", "TrueDepth_DistP1",
                  "B0_own_scores", "B0_T1C_DistP1_archived")


def stratify(per, label_of):
    out = {}
    for meth, rows in per.items():
        if meth not in STRATA_METHODS:
            continue
        for key, x in rows:
            d = out.setdefault(label_of(key), {}).setdefault(meth, {"spearman": [], "regret": [], "top1": []})
            for met in d:
                if x[met] is not None:
                    d[met].append(x[met])
    return {lb: {m: {**{met: (float(np.mean(v)) if v else None) for met, v in d.items()}, "states": len(d["regret"])}
                 for m, d in ms.items()} for lb, ms in sorted(out.items())}


def g1es():
    rows = [r for p in sorted(glob.glob(os.path.join(VF.WORK, "fixed_state", "g1es", "part*_v2.json"))) for r in json.load(open(p))]
    rep = [r for p in sorted(glob.glob(os.path.join(VF.WORK, "fixed_state", "g1es", "part*.json"))) if not p.endswith("_v2.json") for r in json.load(open(p))]
    src = rows or rep
    repro = all(r[f"repro_{m}"]["equal_archived"] and r[f"repro_{m}"]["frames_equal_frame_scores"] for r in src for m in ("frozen", "C2000"))
    out = {"reproduction_pass": repro, "n_states": len(src)}
    if rows and repro:
        def grp(name):
            v = {}
            for r in rows:
                x = r["v2"]["es"] if name == "v2" else r[f"repro_{name}"]["es"]
                if x is not None:
                    v.setdefault((r["scene"], r["start"]), []).append(x)
            return v
        out["es_mean_over_states"] = {m: float(np.mean([r["v2"]["es"] if m == "v2" else r[f"repro_{m}"]["es"] for r in rows
                                                        if (r["v2"]["es"] if m == "v2" else r[f"repro_{m}"]["es"]) is not None]))
                                      for m in ("frozen", "C2000", "v2")}
        for ref in ("frozen", "C2000"):
            d = {}
            for r in rows:
                a, b = r["v2"]["es"], r[f"repro_{ref}"]["es"]
                if a is not None and b is not None:
                    d.setdefault((r["scene"], r["start"]), []).append(a - b)
            out[f"d_v2_minus_{ref}"] = boot_groups(d)
    return out


def timing():
    out = {}
    for p in sorted(glob.glob(os.path.join(VF.WORK, "timing", "*.json"))):
        d = json.load(open(p))
        out[os.path.basename(p)[:-5]] = d
    return out


def main():
    fv = VF.frozen_values()
    os.makedirs(OUT, exist_ok=True)
    res = {"scoring": fv.get("scoring"), "checkpoint": fv.get("checkpoint")}
    for nm, fn in (("S_a", lambda: sa(fv)), ("S_b", lambda: sb(fv)), ("G1_ES", g1es), ("timing", timing)):
        try:
            res[nm] = fn()
        except Exception as e:                         # partial inputs while runs are in progress
            res[nm] = {"error": repr(e)}
    json.dump(res, open(os.path.join(OUT, "t13.json"), "w"), indent=1, default=float)
    for nm in ("S_a", "S_b"):
        m = res[nm].get("methods", {})
        print(nm, {k: (round(v["spearman"]["mean"], 3) if v["spearman"]["mean"] is not None else None,
                       round(v["regret"]["mean"], 3) if v["regret"]["mean"] is not None else None)
                   for k, v in m.items() if not k.startswith("diag")})
    print("G1", res["G1_ES"])


if __name__ == "__main__":
    main()
