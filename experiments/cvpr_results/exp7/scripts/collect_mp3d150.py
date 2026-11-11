"""EXP7 -- MP3D-150 validation + canonical records + aggregation (CPU, seen2scene env; evaluation side, reads GT caches only).

Per episode (every manifest row whose episode.json exists, new or reused):
  V1 151 finite coverage values, moves 150, executed has 151 poses, C non-decreasing
  V2 C_obs replay: the level's frozen evaluator cache (preprocessing/coverage/<level>.npz: 300k GT points, visible set, eps)
     updated with the stored 10 % sample points of svc_t0..t150 reproduces the stored curve (|d| <= 1e-12)
  V3 legality: executed[t+1] is in legal_idx of svc_t<t> for every t (simulator legal list, GT single-segment rule)
  V4 same start pose as every other method of the same level / start
  V5 provenance: OURS / OURS-H0 planner_info ckpt sha256 = V3-T0@40K, P1 mode b2; OURS d_policy d1 with the EXP7 config;
     OURS legal R / T of every 10th record equal the D1 pose table bitwise
Metrics: AUC@T = trapz(C[0..T]) / T (T = 20, 50, 100, 150), Cov@T = C[T] (T = 20, 50, 100, 150), Final = C[150].
Aggregation: starts -> level mean -> level-equal mean; paired level bootstrap (10,000, seed 0) over levels where both methods
have every start VALID. Splits: ALL-22, HO-12 (held-out houses), VALSEL-10, T2-5 (the 5 Task 2 levels x 2 starts).
Outputs: work_dirs/cvpr_results/exp7/episodes/mp3d150/<METHOD>/<tag>.json (canonical) and experiments/cvpr_results/exp7/results/*.csv
Usage: python collect_mp3d150.py [--workers 8]"""
import argparse
import csv
import hashlib
import json
import os
from multiprocessing import Pool

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
E7 = os.path.join(R, "experiments", "cvpr_results", "exp7")
W7 = os.path.join(R, "work_dirs", "cvpr_results", "exp7")
COVC = os.path.join(R, "work_dirs", "exp5_future_view", "task0", "preprocessing", "coverage")
PT = os.path.join(W7, "pose_tables")
T0_SHA = "7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6"
D_CFG = os.path.join(E7, "configs", "d_config_mp3d.yaml")
TS = (20, 50, 100, 150)
METHODS = ["OURS", "MAG-B", "RANDOM", "OURS-H0"]
LABEL = {"OURS": "Ours (V3-T0 + D1)", "MAG-B": "MAGICIAN (beam)", "RANDOM": "Random walk", "OURS-H0": "V3-T0 + H0 (one-step)"}
PAIRS = [("OURS", "MAG-B"), ("OURS", "RANDOM"), ("OURS", "OURS-H0"), ("MAG-B", "RANDOM")]


def metrics(c):
    c = np.asarray(c, np.float64)
    m = {f"AUC@{T}": float(np.trapz(c[:T + 1]) / T) for T in TS}
    m.update({f"Cov@{T}": float(c[T]) for T in TS})
    m["Final"] = float(c[150])
    return m


def validate(row):
    p = os.path.join(R, row["output_path"])
    d = os.path.dirname(p)
    e = json.load(open(p))
    out = {"row": row, "checks": {}, "notes": []}
    c = np.asarray(e["coverage"], np.float64)
    ex = [tuple(int(x) for x in a) for a in e["executed"]]
    v1 = len(c) == 151 and e.get("moves", 150) == 150 and len(ex) == 151 and np.all(np.isfinite(c)) and bool(np.all(np.diff(c) >= -1e-15))
    out["checks"]["V1_shape"] = bool(v1)
    z = np.load(os.path.join(COVC, f"{row['scene_id']}.npz"))
    from scipy.spatial import cKDTree
    pts, vis, eps = z["pts"], z["visible"], float(z["eps_rt"])
    cov = np.zeros(len(pts), bool)
    rep, legal_ok, pose_ok = [], True, True
    lut = None
    if row["method"] == "OURS":
        tb = np.load(os.path.join(PT, f"{row['scene_id']}.npz"))
        lut = {tuple(int(x) for x in r): j for j, r in enumerate(tb["idx"])}
    for t in range(151):
        s = np.load(os.path.join(d, "svc", f"svc_t{t}.npz"))
        sp = np.asarray(s["sample_points"], np.float64)
        if len(sp):
            dd, _ = cKDTree(sp).query(pts, k=1, distance_upper_bound=eps, workers=1)
            cov |= np.isfinite(dd)
        rep.append(float((cov & vis).sum() / max(1, vis.sum())))
        if tuple(int(x) for x in s["pose_idx"]) != ex[t]:
            legal_ok = False
        if t < 150:
            legal = {tuple(int(x) for x in a) for a in s["legal_idx"]}
            if ex[t + 1] not in legal:
                legal_ok = False
        if lut is not None and t % 10 == 0:
            for i, a in enumerate(s["legal_idx"]):
                j = lut[tuple(int(x) for x in a)]
                if not (np.array_equal(tb["R"][j], s["legal_R"][i]) and np.array_equal(tb["T"][j], s["legal_T"][i])):
                    pose_ok = False
    dmax = float(np.max(np.abs(np.asarray(rep) - c))) if len(c) == 151 else float("inf")
    out["checks"]["V2_cobs_replay"] = dmax <= 1e-12
    out["replay_max_abs_diff"] = dmax
    out["checks"]["V3_legal"] = legal_ok
    if row["method"].startswith("OURS"):
        info = json.load(open(os.path.join(d, "planner_info.json")))
        ok = info.get("ckpt_sha256") == T0_SHA and info.get("t3_p1_mode") == "b2"
        if row["method"] == "OURS":
            ok &= info.get("d_policy") == "d1" and os.path.abspath(info.get("d_config", "")) == D_CFG and pose_ok
            err = open(os.path.join(d, "planner.err")).read()
            ok &= '"d_policy": "d1"' in err and "D_POLICY" in err
        out["checks"]["V5_provenance"] = bool(ok)
    else:
        out["checks"]["V5_provenance"] = True
    out["start"] = list(ex[0]) if ex else None
    out["coverage"] = c.tolist()
    out["executed"] = [list(a) for a in ex]
    out["metrics"] = metrics(c) if v1 else None
    out["wall_s"] = e.get("wall_s")
    out["decision_s_mean"] = float(np.mean([s_["decision_s"] for s_ in e["steps"]])) if e.get("steps") else None
    return out


def boot(diff, n=10000, seed=0):
    diff = np.asarray(diff, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    b = diff[idx].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(E7, "manifests", "mp3d150_manifest.csv"))))
    have = [r for r in rows if os.path.exists(os.path.join(R, r["output_path"]))]
    with Pool(a.workers) as pool:
        res = pool.map(validate, have)
    starts = {}
    for o in res:
        starts.setdefault((o["row"]["scene_id"], o["row"]["start_id"]), set()).add(tuple(o["start"]) if o["start"] else None)
    for o in res:
        o["checks"]["V4_same_start"] = len(starts[(o["row"]["scene_id"], o["row"]["start_id"])]) == 1
        o["valid"] = all(o["checks"].values())
    # canonical records
    ep_rows = []
    for o in res:
        r = o["row"]
        tag = f"{r['scene_id']}_S{r['start_id']}"
        rec = {"method": r["method"], "label": LABEL[r["method"]], "scene_id": r["scene_id"], "start_id": int(r["start_id"]), "split": r["split"],
               "task2_subset": int(r["task2_subset"]), "protocol_id": r["protocol_id"], "checkpoint_sha": r["checkpoint_sha"],
               "model_config_hash": r["model_config_hash"], "sim_config_hash": r["sim_config_hash"], "random_seed": r["random_seed"],
               "source": r["output_path"], "source_status": r["status"], "valid": o["valid"], "checks": o["checks"],
               "replay_max_abs_diff": o["replay_max_abs_diff"], "start_pose": o["start"], "coverage": o["coverage"], "executed": o["executed"],
               "metrics": o["metrics"], "wall_s": o["wall_s"], "decision_s_mean": o["decision_s_mean"],
               "gt_access": "simulator (render, single-segment legality) + evaluator (C_obs) only; planner reads service records"}
        od = os.path.join(W7, "episodes", "mp3d150", r["method"])
        os.makedirs(od, exist_ok=True)
        json.dump(rec, open(os.path.join(od, f"{tag}.json"), "w"), indent=1)
        ep_rows.append({"method": r["method"], "scene_id": r["scene_id"], "start_id": r["start_id"], "split": r["split"], "task2_subset": r["task2_subset"],
                        "valid": int(o["valid"]), **{k: int(v) for k, v in o["checks"].items()}, "replay_max_abs_diff": o["replay_max_abs_diff"],
                        "source_status": r["status"], "wall_s": o["wall_s"], "decision_s_mean": o["decision_s_mean"], **(o["metrics"] or {})})
    os.makedirs(os.path.join(E7, "results"), exist_ok=True)
    keys = ["AUC@20", "AUC@50", "AUC@100", "AUC@150", "Cov@20", "Cov@50", "Cov@100", "Final"]
    ep_rows.sort(key=lambda x: (METHODS.index(x["method"]), x["scene_id"], int(x["start_id"])))
    with open(os.path.join(E7, "results", "mp3d150_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_rows[0]))
        w.writeheader()
        w.writerows(ep_rows)
    # level table: only levels whose every manifest start is VALID for the method
    need = {}
    for r in rows:
        need.setdefault((r["method"], r["scene_id"]), set()).add(r["start_id"])
    got = {}
    for x in ep_rows:
        if x["valid"]:
            got.setdefault((x["method"], x["scene_id"]), {})[x["start_id"]] = x
    lv = {}
    for (m, s), st in need.items():
        g = got.get((m, s), {})
        if set(g) == st:
            lv[(m, s)] = {k: float(np.mean([g[i][k] for i in st])) for k in keys}
    meta = {r["scene_id"]: (r["split"], int(r["task2_subset"])) for r in rows}
    split_lv = {"ALL-22": sorted(meta), "HO-12": sorted(s for s in meta if meta[s][0] == "HO"), "VALSEL-10": sorted(s for s in meta if meta[s][0] == "VALSEL"),
                "T2-5": sorted({r["scene_id"] for r in rows if int(r["task2_subset"])})}
    with open(os.path.join(E7, "results", "mp3d150_levels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "scene_id", "split", "task2_subset"] + keys)
        for (m, s), v in sorted(lv.items(), key=lambda kv: (METHODS.index(kv[0][0]), kv[0][1])):
            w.writerow([m, s, meta[s][0], meta[s][1]] + [f"{v[k]:.6f}" for k in keys])
    summ, pair = [], []
    for sp, L in split_lv.items():
        for m in METHODS:
            done = [s for s in L if (m, s) in lv]
            row = {"split": sp, "method": m, "label": LABEL[m], "levels_complete": len(done), "levels_total": len(L), "status": "COMPLETE" if len(done) == len(L) else "PARTIAL"}
            for k in keys:
                row[k] = float(np.mean([lv[(m, s)][k] for s in done])) if done else float("nan")
            summ.append(row)
        for A, B in PAIRS:
            both = [s for s in L if (A, s) in lv and (B, s) in lv]
            if not both:
                continue
            row = {"split": sp, "A": A, "B": B, "levels": len(both), "levels_total": len(L)}
            for k in keys:
                dlt = [lv[(A, s)][k] - lv[(B, s)][k] for s in both]
                lo, hi = boot(dlt)
                row[f"d_{k}"], row[f"ci_lo_{k}"], row[f"ci_hi_{k}"] = float(np.mean(dlt)), lo, hi
                row[f"wins_{k}"] = int(sum(x > 0 for x in dlt))
            pair.append(row)
    for name, data in (("mp3d150_summary.csv", summ), ("mp3d150_paired.csv", pair)):
        if data:
            with open(os.path.join(E7, "results", name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    bad = [x for x in ep_rows if not x["valid"]]
    from collections import Counter
    print("episodes found", len(ep_rows), "valid", len(ep_rows) - len(bad), Counter((x["method"], x["valid"]) for x in ep_rows))
    for x in bad:
        print("INVALID", x["method"], x["scene_id"], x["start_id"], {k: x[k] for k in ("V1_shape", "V2_cobs_replay", "V3_legal", "V4_same_start", "V5_provenance")}, x["replay_max_abs_diff"])
    for row in summ:
        if row["split"] in ("ALL-22", "HO-12", "T2-5"):
            print(row["split"], row["method"], f"{row['levels_complete']}/{row['levels_total']}", " ".join(f"{k} {row[k]:.3f}" for k in keys))


if __name__ == "__main__":
    main()
