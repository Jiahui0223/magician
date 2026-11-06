"""Experiment CT / CT4 -- closed-loop analysis (CPU). Outdoor: official coverage of the CT episodes vs Table 1 V3-DV2-KSV (40K + D-v2), paired per (scene, start), scene bootstrap.
Indoor: C_obs of the CT episodes vs exp6_v3 eval T1 (40K + frozen indoor planner), paired per (floor, start), floor bootstrap. Evaluates the pre-registered All-14 gate.
Output results/CT4_closed_loop/closed_loop.json and per-episode CSV. Usage: python ct_cl_analyze.py"""
import csv
import glob
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(FD))))
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_CT_continued_training", "closed_loop")
TB = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes", "V3-DV2-KSV")
T1 = os.path.join(R, "work_dirs", "exp6_v3", "eval", "closed_loop", "indoor", "T1", "V2M")
DEV5 = ("barts", "colosseum", "eiffel", "pantheon", "sestino_museum")
reg = yaml.safe_load(open(os.path.join(R, "experiments", "cvpr_results", "table1", "SCENE_REGISTRY.yaml")))["scenes"]
items = reg.items() if isinstance(reg, dict) else [(s["local_id"], s) for s in reg]
SPLIT = {k: str(v["split"]).lower() for k, v in items}
ALL14 = sorted(k for k, v in items if (v.get("runnable") or {}).get("overall"))


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    o = {"d": float(v.mean()), "wins": int((v > 1e-12).sum()), "losses": int((v < -1e-12).sum()), "n": len(v)}
    if len(v) >= 2:
        b = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
        o["ci"] = [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    return o


def met(c, H):
    c = np.asarray(c, float)
    o = {"auc20": float(np.trapz(c[:21]) / 20), "auc50": float(np.trapz(c[:51]) / 50), f"auc{H}": float(np.trapz(c[:H + 1]) / H), "cov20": float(c[20]), "final": float(c[H])}
    return o


def outdoor(model):
    out = {}
    for f in glob.glob(os.path.join(W, "outdoor", model, "V2M", "episodes", "ep_V2M_*_r0.json")):
        e = json.load(open(f))
        s, k = e["scene"], int(e["start"])
        base = json.load(open(os.path.join(TB, f"{s}_s{k}.json")))
        out[(s, k)] = {"c": e["coverage"], "b": base["coverage_official"], "keys": e["executed_keys"], "bkeys": base["executed_keys"], "dec": float(np.mean([x["planner"]["decision_s"] for x in e["steps"][:-1] if x.get("planner")])), "seconds": e.get("seconds_this_process")}
    return out


def indoor(model):
    out = {}
    for f in glob.glob(os.path.join(W, "indoor", model, "V2M", "*", "episode.json")):
        e = json.load(open(f))
        tag = e["tag"]
        base = json.load(open(os.path.join(T1, tag, "episode.json")))
        out[(e["level_key"], int(e["start"]))] = {"c": e["coverage"], "b": base["coverage"], "keys": e["executed"], "bkeys": base["executed"], "dec": float(np.mean([x["decision_s"] for x in e["steps"]])), "seconds": e.get("wall_s")}
    return out


def summarize(E, H, group=None):
    scenes = sorted({k[0] for k in E if group is None or k[0] in group})
    o = {"episodes": sum(1 for k in E if k[0] in scenes), "scenes": len(scenes)}
    M = ("auc20", "auc50", f"auc{H}", "cov20", "final")
    per = {}
    for s in scenes:
        ks = [k for k in E if k[0] == s]
        per[s] = {"n": len(ks)}
        for m in M:
            a = [met(E[k]["c"], H)[m] for k in ks]
            b = [met(E[k]["b"], H)[m] for k in ks]
            per[s][m] = float(np.mean(a))
            per[s][m + "_base"] = float(np.mean(b))
            per[s]["d_" + m] = float(np.mean(a) - np.mean(b))
    o["per_scene"] = per
    for m in M:
        o[m] = float(np.mean([per[s][m] for s in scenes]))
        o[m + "_base"] = float(np.mean([per[s][m + "_base"] for s in scenes]))
        o["d_" + m] = boot([per[s]["d_" + m] for s in scenes])
        ep = [met(E[k]["c"], H)[m] - met(E[k]["b"], H)[m] for k in E if k[0] in scenes]
        o["ep_d_" + m] = {"mean": float(np.mean(ep)), "wins": int(sum(x > 1e-12 for x in ep)), "losses": int(sum(x < -1e-12 for x in ep)), "sd": float(np.std(ep, ddof=1)) if len(ep) > 1 else None}
    o["decision_s"] = float(np.mean([E[k]["dec"] for k in E if k[0] in scenes]))
    o["identical_trajectories"] = int(sum(E[k]["keys"] == E[k]["bkeys"] for k in E if k[0] in scenes))
    return o


def gate(od, ind):
    if not od or not ind or od.get("scenes") != 5 or ind.get("scenes", 0) < 5:
        return {"evaluable": False}
    per = od["per_scene"]
    c = {
        "dAUC20 >= -0.005": od["d_auc20"]["d"] >= -0.005, "no scene AUC20 < -0.015": all(per[s]["d_auc20"] >= -0.015 for s in per),
        "dAUC100 >= +0.005 or dFinal >= +0.005": od["d_auc100"]["d"] >= 0.005 or od["d_final"]["d"] >= 0.005,
        ">= 3 of 5 scenes >= 0 (the improving metric)": (sum(per[s]["d_auc100"] >= 0 for s in per) >= 3) if od["d_auc100"]["d"] >= 0.005 else (sum(per[s]["d_final"] >= 0 for s in per) >= 3),
        "at most one scene < -0.010 on AUC100 or Final": sum((per[s]["d_auc100"] < -0.010 or per[s]["d_final"] < -0.010) for s in per) <= 1,
        "indoor dAUC150 >= -0.005": ind["d_auc150"]["d"] >= -0.005, "indoor dFinal >= -0.010": ind["d_final"]["d"] >= -0.010}
    return {"evaluable": True, "criteria": c, "GO": all(c.values())}


def main():
    res = {"models": {}}
    rows = []
    models = sorted({os.path.basename(p) for p in glob.glob(os.path.join(W, "outdoor", "*"))} | {os.path.basename(p) for p in glob.glob(os.path.join(W, "indoor", "*"))})
    for m in models:
        O, I = outdoor(m), indoor(m)
        r = {"n_outdoor": len(O), "n_indoor": len(I)}
        if I:
            r["indoor"] = summarize(I, 150)
        if O:
            r["dev5"] = summarize(O, 100, DEV5) if any(k[0] in DEV5 for k in O) else None
            for g, ss in (("train6", [s for s in ALL14 if SPLIT[s].startswith("train")]), ("test3", [s for s in ALL14 if SPLIT[s].startswith("test")]), ("all14", ALL14)):
                if all((s, k) in O for s in ss for k in range(5)):
                    r[g] = summarize(O, 100, ss)
        if r.get("dev5") and r.get("indoor"):
            r["all14_gate"] = gate(r["dev5"], r["indoor"])
        res["models"][m] = r
        for k, e in sorted(O.items()):
            x, y = met(e["c"], 100), met(e["b"], 100)
            rows.append([m, "outdoor", k[0], k[1], x["auc20"], x["auc100"], x["cov20"], x["final"], y["auc20"], y["auc100"], y["cov20"], y["final"]])
        for k, e in sorted(I.items()):
            x, y = met(e["c"], 150), met(e["b"], 150)
            rows.append([m, "indoor", k[0], k[1], x["auc20"], x["auc150"], x["cov20"], x["final"], y["auc20"], y["auc150"], y["cov20"], y["final"]])
    d5 = {m: r["dev5"]["auc100"] for m, r in res["models"].items() if r.get("dev5") and r["dev5"]["scenes"] == 5 and r["dev5"]["episodes"] == 25 and not m.startswith("R_")}
    res["dev5_auc100"] = d5
    os.makedirs(os.path.join(FD, "results", "CT4_closed_loop"), exist_ok=True)
    json.dump(res, open(os.path.join(FD, "results", "CT4_closed_loop", "closed_loop.json"), "w"), indent=1, default=float)
    with open(os.path.join(FD, "results", "CT4_closed_loop", "episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "domain", "scene", "start", "auc20", "auc_full", "cov20", "final", "base_auc20", "base_auc_full", "base_cov20", "base_final"])
        w.writerows(rows)
    fm = lambda x: "-" if x is None else (f"{x['d']:+.3f} {[round(v, 3) for v in x.get('ci', [])]} {x['wins']}/{x['losses']}")  # noqa: E731
    for m, r in res["models"].items():
        for g, H in (("indoor", 150), ("dev5", 100), ("train6", 100), ("test3", 100), ("all14", 100)):
            x = r.get(g)
            if not x:
                continue
            print(m, g, f"eps {x['episodes']} | AUC20 {x['auc20']:.3f} (base {x['auc20_base']:.3f}) d {fm(x['d_auc20'])} | AUC{H} {x[f'auc{H}']:.3f} (base {x[f'auc{H}_base']:.3f}) d {fm(x[f'd_auc{H}'])} | Cov20 d {fm(x['d_cov20'])} | Final {x['final']:.3f} (base {x['final_base']:.3f}) d {fm(x['d_final'])} | identical {x['identical_trajectories']} dec {x['decision_s']:.2f}s")
        print(m, "gate", r.get("all14_gate"))
    print("dev5 AUC100", d5)


if __name__ == "__main__":
    main()
