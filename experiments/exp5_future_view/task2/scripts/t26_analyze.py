"""EXP5 Task 2 -- final tables from the saved runs (seen2scene env, CPU; protocol §5).

Reads only the recomputed metric files and the stored episodes:
  indoor : W2/eval/indoor/closed_loop/<g>/<tag>/episode.json (C_obs curve, 151 values, decision times) and
           W2/eval/indoor/metrics/episodes/<g>/<tag>.json (recomputed C_obs = stored bitwise, Comp curves)
  outdoor: W2/eval/outdoor/closed_loop/<g>/episodes/ep_<tag>.json (official curve, decision times) and
           W2/eval/outdoor/metrics/<g>/<tag>.json (monotone curve = primary, official replay = stored bitwise)
Averages: per scene over its episodes, then over scenes; 95 % CI = scene-cluster bootstrap (10,000, seed 0); paired
differences per episode with the same bootstrap; every episode value is written out.
Outputs: task2/tables/{table1_indoor.csv, table1_outdoor_test.csv, table1_outdoor_dev.csv, table2_indoor.csv,
table2_outdoor.csv, episodes_indoor.csv, episodes_outdoor.csv, paired.csv}, W2/analysis/t26.json; missing episodes are
listed as Incomplete (never averaged in).
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402

GROUPS = ("MAG-G", "MAG-B", "V2O", "V2M")
LAB = {"MAG-G": "MAGICIAN Greedy", "MAG-B": "MAGICIAN Beam", "V2O": "V2 Original", "V2M": "V2-Mix (Ours)"}
PAIRS = (("V2M", "V2O"), ("V2M", "MAG-G"), ("V2M", "MAG-B"), ("V2O", "MAG-G"), ("V2O", "MAG-B"), ("MAG-B", "MAG-G"))
CFG_OUT = json.load(open(os.path.join(C.T2, "configs", "eval_outdoor_v1.json")))


def auc(c, T):
    c = np.asarray(c, float)[:T + 1]
    return float(np.trapz(c, dx=1.0) / T)


def boot(vals, clusters, n=10000, seed=0):
    by = {}
    for v, c in zip(vals, clusters):
        if v is not None and np.isfinite(v):
            by.setdefault(c, []).append(v)
    if not by:
        return None
    cm = np.asarray([np.mean(v) for v in by.values()])
    bs = cm[np.random.default_rng(seed).integers(0, len(cm), (n, len(cm)))].mean(1)
    allv = [x for v in by.values() for x in v]
    return {"mean": float(cm.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "scene_sd": float(cm.std(ddof=1)) if len(cm) > 1 else None, "n_episodes": len(allv), "n_scenes": len(cm),
            "n_pos": int(sum(x > 0 for x in allv))}


def indoor():
    rows, missing = [], []
    for s, lv, st in C.EVAL_LEVELS:
        tag = f"{s}_L{lv}_S{st}"
        for g in GROUPS:
            pe = os.path.join(C.W2, "eval", "indoor", "closed_loop", g, tag, "episode.json")
            pm = os.path.join(C.W2, "eval", "indoor", "metrics", "episodes", g, f"{tag}.json")
            if not (os.path.exists(pe) and os.path.exists(pm)):
                missing.append(f"{g}/{tag}")
                continue
            e, m = json.load(open(pe)), json.load(open(pm))
            assert m["coverage_reproduction_max_abs_diff"] <= 1e-12
            c = e["coverage"]
            cv = {r["t"]: r for r in m["curve"]}
            rows.append({"dataset": "MP3D", "scene": f"{s}_L{lv}", "episode": tag, "group": g, "frames": len(c),
                         "auc100": auc(c, 100), "auc150": auc(c, 150), "cov100": c[100], "cov150": c[150],
                         "comp5_150": cv[150]["comp5"], "comp_cm_mean_150": cv[150]["dist_mean_cm"], "comp_cm_median_150": cv[150]["dist_median_cm"],
                         "comp5vis_150": cv[150]["comp5_vis"], "decision_s_mean": float(np.mean([x["decision_s"] for x in e["steps"]])),
                         "pose_revisit_ratio": m.get("pose_revisit_ratio"), "path_m": m.get("path_m")})
    return rows, missing


def outdoor():
    rows, missing = [], []
    for split in ("test", "dev"):
        for s in CFG_OUT["scenes"][split]:
            for k in CFG_OUT["starts"]:
                for g in GROUPS:
                    tag = f"{g}_{s}_s{k}_r0"
                    pe = os.path.join(C.W2, "eval", "outdoor", "closed_loop", g, "episodes", f"ep_{tag}.json")
                    pm = os.path.join(C.W2, "eval", "outdoor", "metrics", g, f"{tag}.json")
                    if not (os.path.exists(pe) and os.path.exists(pm)):
                        missing.append(f"{split}/{tag}")
                        continue
                    e, m = json.load(open(pe)), json.load(open(pm))
                    assert m["official_equal_stored"] and len(m["cov_mono"]) == e["moves"] + 1 == 101
                    dec = [x["planner"]["decision_s"] for x in e["steps"][:-1] if x.get("planner")]
                    rows.append({"dataset": f"Macarons++ {split}", "split": split, "scene": s, "episode": f"{s}_s{k}", "group": g,
                                 "frames": len(m["cov_mono"]), "auc100": auc(m["cov_mono"], 100), "cov100": m["cov_mono"][100],
                                 "auc100_official": auc(m["cov_official"], 100), "cov100_official": m["cov_official"][100],
                                 "official_max_drop": m["official_max_drop"], "n_hover": e["n_hover"],
                                 "decision_s_mean": float(np.mean(dec)) if dec else None})
    return rows, missing


def summarise(rows, keys):
    out = {}
    for g in GROUPS:
        r = [x for x in rows if x["group"] == g]
        out[g] = {k: boot([x[k] for x in r], [x["scene"] for x in r]) for k in keys}
    return out


def paired(rows, keys):
    out = {}
    eps = sorted({x["episode"] for x in rows})
    idx = {(x["group"], x["episode"]): x for x in rows}
    for a, b in PAIRS:
        d = [(idx[(a, e)], idx[(b, e)]) for e in eps if (a, e) in idx and (b, e) in idx]
        out[f"{a}-{b}"] = {k: boot([x[k] - y[k] for x, y in d if x[k] is not None and y[k] is not None],
                                   [x["scene"] for x, y in d if x[k] is not None and y[k] is not None]) for k in keys}
    return out


def write_csv(p, rows, cols):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])


def fmt(s, k, pct=False):
    v = s.get(k)
    if not v:
        return "N/A"
    m, (lo, hi) = v["mean"], v["ci95"]
    return f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]" if pct else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    tabd = os.path.join(C.T2, "tables")
    ri, mi = indoor()
    ro, mo = outdoor()
    ki = ("auc100", "auc150", "cov100", "cov150", "comp5_150", "comp_cm_mean_150", "comp_cm_median_150", "comp5vis_150", "decision_s_mean")
    ko = ("auc100", "cov100", "auc100_official", "cov100_official", "decision_s_mean")
    res = {"doc": __doc__, "missing_indoor": mi, "missing_outdoor": mo,
           "indoor": summarise(ri, ki), "indoor_paired": paired(ri, ki[:7]),
           "outdoor_test": summarise([x for x in ro if x["split"] == "test"], ko),
           "outdoor_test_paired": paired([x for x in ro if x["split"] == "test"], ko[:4]),
           "outdoor_dev": summarise([x for x in ro if x["split"] == "dev"], ko),
           "outdoor_dev_paired": paired([x for x in ro if x["split"] == "dev"], ko[:4])}
    os.makedirs(os.path.join(C.W2, "analysis"), exist_ok=True)
    json.dump(res, open(os.path.join(C.W2, "analysis", "t26.json"), "w"), indent=1)
    write_csv(os.path.join(tabd, "episodes_indoor.csv"), ri, ["dataset", "scene", "episode", "group", "frames"] + list(ki) + ["pose_revisit_ratio", "path_m"])
    write_csv(os.path.join(tabd, "episodes_outdoor.csv"), ro, ["dataset", "split", "scene", "episode", "group", "frames"] + list(ko) + ["official_max_drop", "n_hover"])
    t1 = [{"Method": LAB[g], "AUC@150": fmt(res["indoor"][g], "auc150"), "Final Cov.@150": fmt(res["indoor"][g], "cov150"),
           "Comp.(%)": fmt(res["indoor"][g], "comp5_150", True), "Comp.(cm)": fmt(res["indoor"][g], "comp_cm_mean_150"),
           "AUC@100": fmt(res["indoor"][g], "auc100"), "decision s/step": fmt(res["indoor"][g], "decision_s_mean")} for g in GROUPS]
    write_csv(os.path.join(tabd, "table1_indoor.csv"), t1, list(t1[0]))
    for split in ("test", "dev"):
        t = [{"Method": LAB[g], "AUC@100": fmt(res[f"outdoor_{split}"][g], "auc100"), "Final Cov.@100": fmt(res[f"outdoor_{split}"][g], "cov100"),
              "AUC@100 (official metric)": fmt(res[f"outdoor_{split}"][g], "auc100_official"),
              "Cov.@100 (official metric)": fmt(res[f"outdoor_{split}"][g], "cov100_official"),
              "decision s/step": fmt(res[f"outdoor_{split}"][g], "decision_s_mean")} for g in GROUPS]
        write_csv(os.path.join(tabd, f"table1_outdoor_{split}.csv"), t, list(t[0]))
    t2i = []
    for sc in sorted({x["scene"] for x in ri}) + ["Average"]:
        for g in GROUPS:
            r = [x for x in ri if x["group"] == g and (sc == "Average" or x["scene"] == sc)]
            if not r:
                continue
            if sc == "Average":
                v = {k: res["indoor"][g][k]["mean"] for k in ("auc150", "cov150", "comp5_150", "comp_cm_mean_150")}
            else:
                v = {k: float(np.mean([x[k] for x in r])) for k in ("auc150", "cov150", "comp5_150", "comp_cm_mean_150")}
            t2i.append({"Scene": sc, "Method": LAB[g], "episodes": len(r), "AUC@150": round(v["auc150"], 4), "Final Cov.@150": round(v["cov150"], 4),
                        "Comp.(%)": round(100 * v["comp5_150"], 2), "Comp.(cm)": round(v["comp_cm_mean_150"], 2)})
    write_csv(os.path.join(tabd, "table2_indoor.csv"), t2i, ["Scene", "Method", "episodes", "AUC@150", "Final Cov.@150", "Comp.(%)", "Comp.(cm)"])
    t2o = []
    for split in ("test", "dev"):
        for sc in CFG_OUT["scenes"][split] + ["Average"]:
            for g in GROUPS:
                r = [x for x in ro if x["group"] == g and x["split"] == split and (sc == "Average" or x["scene"] == sc)]
                if not r:
                    continue
                v = ({k: res[f"outdoor_{split}"][g][k]["mean"] for k in ("auc100", "cov100", "auc100_official", "cov100_official")} if sc == "Average"
                     else {k: float(np.mean([x[k] for x in r])) for k in ("auc100", "cov100", "auc100_official", "cov100_official")})
                t2o.append({"Split": split, "Scene": sc, "Method": LAB[g], "episodes": len(r), "AUC@100": round(v["auc100"], 4),
                            "Final Cov.@100": round(v["cov100"], 4), "AUC@100 official": round(v["auc100_official"], 4),
                            "Cov.@100 official": round(v["cov100_official"], 4)})
    write_csv(os.path.join(tabd, "table2_outdoor.csv"), t2o, ["Split", "Scene", "Method", "episodes", "AUC@100", "Final Cov.@100", "AUC@100 official", "Cov.@100 official"])
    pr = []
    for nm, d in (("MP3D", res["indoor_paired"]), ("Macarons++ test", res["outdoor_test_paired"]), ("Macarons++ dev", res["outdoor_dev_paired"])):
        for pair, kk in d.items():
            for k, v in kk.items():
                if v:
                    pr.append({"dataset": nm, "pair": pair, "metric": k, "mean": round(v["mean"], 4), "ci95_lo": round(v["ci95"][0], 4),
                               "ci95_hi": round(v["ci95"][1], 4), "n_pos": v["n_pos"], "n_episodes": v["n_episodes"], "n_scenes": v["n_scenes"]})
    write_csv(os.path.join(tabd, "paired.csv"), pr, ["dataset", "pair", "metric", "mean", "ci95_lo", "ci95_hi", "n_pos", "n_episodes", "n_scenes"])
    print("missing indoor", mi, "missing outdoor", mo)
    for r in t1:
        print(r)


if __name__ == "__main__":
    main()
