"""E3 -- closed-loop comparison D-v2 vs D1 (CPU; read-only) -> results/closed_loop_{episodes.csv,summary.json}.
D-v2 episodes: work_dirs/.../E3_dv2_closed_loop/runs/dv2 (first-layer ablation: runs/dv2_first).
D1 references, strictly paired by (scene, start, rep) = same start pose, same planner noise keys:
  rep 0            CVPR Table 1 canonical D1 episodes (coverage_official; identical to the simulator coverage)
  colosseum rep 1/2 experiment E noise replicates (runs/rep/D1)
  pisa rep 1/2      E3 runs with the switch off (runs/d1_off; switch off verified bitwise == D1)
Stages / gates: configs/dv2_v2.yaml. Scene-level paired bootstrap (10k, seed 0) on scene means; episode-level bootstrap for the
single-scene stages. Also per step: planning time, D-v2 cut shares (side log), predicted value of the executed view vs realised gain,
all-zero fallbacks, trajectory statistics (first changed pick, distinct positions, azimuth sectors around the scene centre)."""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.dirname(HERE)
EE = os.path.dirname(E3)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(EE))))
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_E_stability_plateau")
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
POSE = os.path.join(R, "work_dirs", "exp6_v3", "H", "pose_tables")
SPLIT = {"train": ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer"], "dev": ["barts", "colosseum", "eiffel", "pantheon", "sestino_museum"],
         "test": ["alhambra", "bridge", "neuschwanstein"]}
ALL14 = SPLIT["train"] + SPLIT["dev"] + SPLIT["test"]
QS = ("auc20", "auc50", "auc100", "cov20", "final")
_pose = {}
CHECK = {}


def met(c):
    c = np.asarray(c, float)
    return {"auc20": float(np.trapz(c[:21]) / 20), "auc50": float(np.trapz(c[:51]) / 50), "auc100": float(np.trapz(c) / 100), "cov20": float(c[20]), "final": float(c[100])}


def plog(root, tag):
    p = os.path.join(root, "planner_logs", f"pl_{tag}.steps.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def centres(scene):
    if scene not in _pose:
        z = np.load(os.path.join(POSE, f"{scene}.npz"))
        _pose[scene] = ({tuple(int(x) for x in r): j for j, r in enumerate(z["idx"])}, z["C"])
    return _pose[scene]


def traj(scene, keys):
    lut, C = centres(scene)
    idx = [tuple(json.loads(k)) for k in keys]
    P = np.asarray([C[lut[i]] for i in idx])
    ctr = C.mean(0)
    v = P - ctr
    hor = [a for a in range(3) if a != 1] if np.ptp(C[:, 1]) < max(np.ptp(C[:, 0]), np.ptp(C[:, 2])) else [0, 2]
    az = np.degrees(np.arctan2(v[:, hor[1]], v[:, hor[0]])) % 360
    return {"positions": len({i[:3] for i in idx}), "sectors8": len(set((az // 45).astype(int).tolist())), "path_len": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()),
            "pos_set": {i[:3] for i in idx}, "late_positions": len({i[:3] for i in idx[20:]}), "late_sectors8": len(set((az[20:] // 45).astype(int).tolist()))}


def episode(root, scene, k, rep, cov_key="coverage"):
    f = os.path.join(root, "episodes", f"ep_V2M_{scene}_s{k}_r{rep}.json")
    if not os.path.exists(f):
        return None
    e = json.load(open(f))
    lg = plog(root, e["tag"])
    side = os.path.join(root, "planner_logs", f"dv2_V2M_{scene}_s{k}_r{rep}.jsonl")
    sl = [json.loads(l) for l in open(side)] if os.path.exists(side) else []
    return {"cov": e[cov_key], "keys": [s["pose_key"] for s in e["steps"]], "picks": [s.get("pick") for s in e["steps"]], "legal": [s.get("legal_keys") for s in e["steps"]],
            "log": lg, "side": sl}


def load():
    D, V, F = {}, {}, {}
    for s in ALL14:
        for k in range(5):
            p = os.path.join(EP, "V3-D1-H3-NORISK", f"{s}_s{k}.json")
            if os.path.exists(p):
                r = json.load(open(p))
                root = os.path.dirname(os.path.dirname(os.path.join(R, r["episode_file"])))
                e = episode(root, s, k, 0)
                if e:
                    e["cov"] = r["coverage_official"]
                    D[(s, k, 0)] = e
    for root, scenes in ((os.path.join(W, "runs", "rep", "D1"), ["colosseum"]), (os.path.join(W, "E3_dv2_closed_loop", "runs", "d1_off"), ALL14)):
        for s in scenes:
            for k in range(5):
                for rep in (1, 2):
                    e = episode(root, s, k, rep)
                    if e and (s, k, rep) not in D:
                        D[(s, k, rep)] = e
    for store, sub in ((V, "dv2"), (F, "dv2_first")):
        root = os.path.join(W, "E3_dv2_closed_loop", "runs", sub)
        for f in glob.glob(os.path.join(root, "episodes", "ep_V2M_*.json")):
            e = json.load(open(f))
            store[(e["scene"], int(e["start"]), int(e["rep"]))] = episode(root, e["scene"], int(e["start"]), int(e["rep"]))
    # CVPR Table 1 canonical D-v2 records (model V3-DV2-KSV, rep 0, run through the Table 1 pipeline). Episodes also run in E3 must be
    # bitwise identical (CHECK["table1_equals_e3"]); the others are added.
    CHECK.update({"table1_records": 0, "table1_overlap": 0, "table1_equals_e3": 0, "table1_mismatch": []})
    for s in ALL14:
        for k in range(5):
            p = os.path.join(EP, "V3-DV2-KSV", f"{s}_s{k}.json")
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            CHECK["table1_records"] += 1
            root = os.path.dirname(os.path.dirname(os.path.join(R, r["episode_file"])))
            e = episode(root, s, k, 0)
            e["cov"] = r["coverage_official"]
            if (s, k, 0) in V:
                CHECK["table1_overlap"] += 1
                same = V[(s, k, 0)]["cov"] == e["cov"] and V[(s, k, 0)]["keys"] == e["keys"]
                CHECK["table1_equals_e3"] += int(same)
                if not same:
                    CHECK["table1_mismatch"].append(f"{s}_s{k}")
            else:
                V[(s, k, 0)] = e
    return D, V, F


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [None, None]
    b = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def pair_rows(D, V):
    rows = []
    for key in sorted(set(D) & set(V)):
        d, v = D[key], V[key]
        md, mv = met(d["cov"]), met(v["cov"])
        div = next((i for i, (x, y) in enumerate(zip(d["keys"], v["keys"])) if x != y), None)
        td, tv = traj(key[0], d["keys"]), traj(key[0], v["keys"])
        ps = lambda L: float(np.mean([x["decision_s"] for x in L])) if L else None  # noqa: E731
        late = lambda e: [x for x in e["log"] if x["t"] >= 20]  # noqa: E731
        pv = lambda e: float(np.median([x["scores"][x["choice"]["pick"]] for x in late(e) if x["choice"]["pick"] is not None])) if late(e) else None  # noqa: E731
        zero = lambda e: sum(1 for x in e["log"] if x["choice"].get("fallback"))  # noqa: E731
        side = v["side"]
        cs = lambda L, a, b: float(sum(x["cut"][L] for x in side[a:b]) / max(1, sum(x["rays"][L] for x in side[a:b]))) if side else None  # noqa: E731
        rows.append({"scene": key[0], "start": key[1], "rep": key[2], **{f"d1_{q}": md[q] for q in QS}, **{f"dv2_{q}": mv[q] for q in QS}, **{f"d_{q}": mv[q] - md[q] for q in QS},
                     "d1_plan_s": ps(d["log"]), "dv2_plan_s": ps(v["log"]), "dv2_filter_s": float(np.mean([sum(x["dv2_s"]) for x in side])) if side else None,
                     "first_changed_t": div, "pos_jaccard": len(td["pos_set"] & tv["pos_set"]) / max(1, len(td["pos_set"] | tv["pos_set"])),
                     "d1_positions": td["positions"], "dv2_positions": tv["positions"], "d1_sectors8": td["sectors8"], "dv2_sectors8": tv["sectors8"],
                     "d1_late_positions": td["late_positions"], "dv2_late_positions": tv["late_positions"], "d1_path_len": td["path_len"], "dv2_path_len": tv["path_len"],
                     "d1_late_gain": float(d["cov"][100] - d["cov"][20]), "dv2_late_gain": float(v["cov"][100] - v["cov"][20]),
                     "d1_late_pred_value": pv(d), "dv2_late_pred_value": pv(v), "d1_allzero_steps": zero(d), "dv2_allzero_steps": zero(v),
                     "cut_L1_early": cs(0, 0, 20), "cut_L1_late": cs(0, 20, 100), "cut_L2_early": cs(1, 0, 20), "cut_L2_late": cs(1, 20, 100)})
    return rows


def scene_stage(rows, scenes, reps):
    out = {}
    for s in scenes:
        r = [x for x in rows if x["scene"] == s and x["rep"] in reps]
        if not r:
            continue
        o = {"pairs": len(r), "plateau_d1": sum(x["d1_final"] < 0.70 for x in r), "plateau_dv2": sum(x["dv2_final"] < 0.70 for x in r)}
        for q in QS:
            dd = [x[f"d_{q}"] for x in r]
            o[q] = {"d1": float(np.mean([x[f"d1_{q}"] for x in r])), "dv2": float(np.mean([x[f"dv2_{q}"] for x in r])), "d": float(np.mean(dd)), "ci": boot(dd),
                    "better": int(sum(v > 1e-9 for v in dd)), "worse": int(sum(v < -1e-9 for v in dd))}
        for a in ("plan_s", "positions", "sectors8", "late_positions", "late_gain", "late_pred_value", "allzero_steps", "path_len"):
            for m in ("d1", "dv2"):
                vals = [x[f"{m}_{a}"] for x in r if x[f"{m}_{a}"] is not None]
                o[f"{m}_{a}"] = float(np.mean(vals)) if vals else None
        for a in ("first_changed_t", "pos_jaccard", "cut_L1_early", "cut_L1_late", "cut_L2_early", "cut_L2_late", "dv2_filter_s"):
            vals = [x[a] for x in r if x[a] is not None]
            o[a] = float(np.median(vals)) if a == "first_changed_t" and vals else (float(np.mean(vals)) if vals else None)
        out[s] = o
    return out


def multi_scene(rows, scenes, n_per_scene=5):
    r0 = [x for x in rows if x["rep"] == 0 and x["scene"] in scenes]
    done = [s for s in scenes if sum(1 for x in r0 if x["scene"] == s) == n_per_scene]
    o = {"scenes_complete": done, "episodes": sum(1 for x in r0 if x["scene"] in done), "partial": len(done) < len(scenes)}
    for q in QS:
        sd = {s: float(np.mean([x[f"d_{q}"] for x in r0 if x["scene"] == s])) for s in done}
        d = list(sd.values())
        o[q] = {"d1": float(np.mean([np.mean([x[f"d1_{q}"] for x in r0 if x["scene"] == s]) for s in done])) if done else None,
                "dv2": float(np.mean([np.mean([x[f"dv2_{q}"] for x in r0 if x["scene"] == s]) for s in done])) if done else None,
                "d": float(np.mean(d)) if d else None, "ci": boot(d), "wins": int(sum(v > 1e-9 for v in d)), "losses": int(sum(v < -1e-9 for v in d)), "per_scene": sd,
                "d_without_best": float(np.mean(sorted(d)[:-1])) if len(d) > 1 else None}
    ep = [x for x in r0 if x["scene"] in done]
    o["plateau_d1"] = sum(x["d1_final"] < 0.70 for x in ep)
    o["plateau_dv2"] = sum(x["dv2_final"] < 0.70 for x in ep)
    o["plan_s_d1"] = float(np.mean([x["d1_plan_s"] for x in ep if x["d1_plan_s"]])) if ep else None
    o["plan_s_dv2"] = float(np.mean([x["dv2_plan_s"] for x in ep if x["dv2_plan_s"]])) if ep else None
    o["episodes_better_final"] = sum(x["d_final"] > 1e-9 for x in ep)
    o["episodes_worse_final"] = sum(x["d_final"] < -1e-9 for x in ep)
    return o


def main():
    D, V, F = load()
    rows = pair_rows(D, V)
    os.makedirs(os.path.join(E3, "results"), exist_ok=True)
    with open(os.path.join(E3, "results", "closed_loop_episodes.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    out = {"n_pairs": len(rows), "unpaired_dv2": sorted(f"{a}_s{b}_r{c}" for a, b, c in set(V) - set(D)), "table1_check": dict(CHECK)}
    out["stage_A"] = scene_stage(rows, ["colosseum", "pisa"], (0,))
    out["stage_B"] = scene_stage(rows, ["colosseum", "pisa"], (0, 1, 2))
    A, B = out["stage_A"], out["stage_B"]
    if len(A) == 2 and all(A[s]["pairs"] == 5 for s in A):
        c, p = A["colosseum"], A["pisa"]
        out["stage_A_positive"] = bool((c["auc100"]["d"] > 0 or c["final"]["d"] > 0) and p["final"]["d"] >= -0.005 and c["auc20"]["d"] >= -0.02 and p["auc20"]["d"] >= -0.02)
    if len(B) == 2 and all(B[s]["pairs"] == 15 for s in B):
        c, p = B["colosseum"], B["pisa"]
        out["stage_B_proceed"] = bool(c["plateau_dv2"] <= c["plateau_d1"] and c["auc100"]["d"] >= 0 and p["final"]["d"] >= -0.005 and p["auc100"]["d"] >= -0.005
                                      and c["auc20"]["d"] >= -0.01 and p["auc20"]["d"] >= -0.01)
    dev = multi_scene(rows, SPLIT["dev"])
    out["dev5"] = dev
    if not dev["partial"]:
        g = {"G1_early": bool(dev["auc20"]["d"] >= -0.005 and min(dev["auc20"]["per_scene"].values()) >= -0.015),
             "G2_gain": bool(dev["auc100"]["d"] >= 0.010 or dev["final"]["d"] >= 0.010)}
        met_q = [q for q in ("auc100", "final") if dev[q]["d"] >= 0.010]
        g["G3_not_one_scene"] = bool(met_q and all(dev[q]["d_without_best"] >= 0 for q in met_q))
        g["G4_no_sacrifice"] = bool(sum(1 for s in dev["scenes_complete"] if dev["auc100"]["per_scene"][s] < -0.010 or dev["final"]["per_scene"][s] < -0.010) <= 1)
        g["G5_cost"] = bool(dev["plan_s_dv2"] <= 1.25 * dev["plan_s_d1"])
        out["dev5_gates"] = g
        out["dev5_pass"] = all(g.values())
    for name, scenes in (("train6", SPLIT["train"]), ("test3", SPLIT["test"]), ("all14", ALL14)):
        out[name] = multi_scene(rows, scenes)
    if F:
        fr = pair_rows(D, F)
        out["first_layer_ablation"] = {"pairs": len(fr), "scenes": scene_stage(fr, sorted({x["scene"] for x in fr}), (0, 1, 2)),
                                       "full_on_same_pairs": scene_stage([x for x in rows if (x["scene"], x["start"], x["rep"]) in {(y["scene"], y["start"], y["rep"]) for y in fr}],
                                                                         sorted({x["scene"] for x in fr}), (0, 1, 2))}
    json.dump(out, open(os.path.join(E3, "results", "closed_loop_summary.json"), "w"), indent=1, default=float)
    f3 = lambda v: "—" if v is None else f"{v:+.3f}"  # noqa: E731
    for st in ("stage_A", "stage_B"):
        for s, o in out[st].items():
            print(f"{st} {s}: pairs {o['pairs']} | " + " ".join(f"{q} {o[q]['d1']:.3f}->{o[q]['dv2']:.3f} ({f3(o[q]['d'])} [{f3(o[q]['ci'][0])},{f3(o[q]['ci'][1])}] {o[q]['better']}/{o[q]['worse']})" for q in QS)
                  + f" | plateau {o['plateau_d1']}->{o['plateau_dv2']} | plan {o['d1_plan_s']:.2f}->{o['dv2_plan_s']:.2f}s | first change t={o['first_changed_t']} jaccard {o['pos_jaccard']:.2f}"
                  + f" | positions {o['d1_positions']:.1f}->{o['dv2_positions']:.1f} sectors {o['d1_sectors8']:.1f}->{o['dv2_sectors8']:.1f} | late gain {o['d1_late_gain']:.3f}->{o['dv2_late_gain']:.3f}"
                  + f" | late pred value {o['d1_late_pred_value']:.0f}->{o['dv2_late_pred_value']:.0f} | all-zero steps {o['d1_allzero_steps']:.1f}->{o['dv2_allzero_steps']:.1f}"
                  + f" | cut L1 early/late {o['cut_L1_early']}/{o['cut_L1_late']} L2 {o['cut_L2_early']}/{o['cut_L2_late']}")
    print({k: out[k] for k in ("stage_A_positive", "stage_B_proceed", "dev5_gates", "dev5_pass", "table1_check") if k in out})
    for name in ("dev5", "train6", "test3", "all14"):
        o = out[name]
        if o["scenes_complete"]:
            print(f"{name}: scenes {len(o['scenes_complete'])} ep {o['episodes']} partial {o['partial']} | " + " ".join(
                f"{q} {o[q]['d1']:.3f}->{o[q]['dv2']:.3f} ({f3(o[q]['d'])} [{f3(o[q]['ci'][0])},{f3(o[q]['ci'][1])}] W{o[q]['wins']}/L{o[q]['losses']})" for q in QS)
                + f" | plateau {o['plateau_d1']}->{o['plateau_dv2']} | plan {o['plan_s_d1']:.2f}->{o['plan_s_dv2']:.2f}")
            print("   per scene final:", {s: round(v, 3) for s, v in o["final"]["per_scene"].items()}, "auc100:", {s: round(v, 3) for s, v in o["auc100"]["per_scene"].items()},
                  "auc20:", {s: round(v, 3) for s, v in o["auc20"]["per_scene"].items()})
    if "first_layer_ablation" in out:
        for s, o in out["first_layer_ablation"]["scenes"].items():
            g = out["first_layer_ablation"]["full_on_same_pairs"].get(s)
            print(f"first-layer-only {s}: pairs {o['pairs']} " + " ".join(f"{q} {f3(o[q]['d'])}" for q in QS) + f" plateau {o['plateau_d1']}->{o['plateau_dv2']}"
                  + (" || full on same pairs: " + " ".join(f"{q} {f3(g[q]['d'])}" for q in QS) + f" plateau {g['plateau_d1']}->{g['plateau_dv2']}" if g else ""))


if __name__ == "__main__":
    sys.exit(main())
