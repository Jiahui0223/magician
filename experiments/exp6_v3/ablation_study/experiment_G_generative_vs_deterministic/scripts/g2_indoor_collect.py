"""Experiment G2 item 5 -- validation + aggregation of the MP3D-150 indoor closed loops (protocol G2-v1 section 5; CPU only).
Rows = the 44 EXP7 manifest OURS rows (same levels / starts); outputs work_dirs/exp6_v3/experiment_G/g2/indoor/raw/<ARM>/V2M/<tag>/episode.json.
Per episode: experiments/cvpr_results/exp7/scripts/collect_mp3d150.validate() UNCHANGED with method "OURS" (V1 shape, V2 C_obs replay,
V3 legality, V5 provenance incl. D1 pose tables + EXP7 d config; for D0IN the module constant T0_SHA is set to the D0 sha for the call, the
only substitution), V4 = start pose equal to the archived EXP7 OURS record of the same level / start, plus: planner.err G_MODE line with the
arm's mode and K = 1, planner_info g_k = 1. G4 = archived EXP7 OURS canonical records (valid). Aggregation as EXP7: starts -> level mean ->
level-equal mean; paired level bootstrap (10,000, seed 0) over levels where both arms have every start VALID; ALL-22 / HO-12 / VALSEL-10.
Writes results/g2/indoor/{episodes.csv, summary.csv, paired.csv, G2_INDOOR.md}. Usage: python g2_indoor_collect.py [--workers 8] [--allow-incomplete]"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments/cvpr_results/exp7/scripts"))
import collect_mp3d150 as C7  # noqa: E402

GD = os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic")
WI = os.path.join(R, "work_dirs/exp6_v3/experiment_G/g2/indoor")
OUT = os.path.join(GD, "results/g2/indoor")
D0_SHA = "aeefe098e86517a1ff3b7455f3a83338e3972fb0f4fcd95a22607dd77d83f205"
ARMS = {"G1IN": ("flow", C7.T0_SHA), "D0IN": ("d0", D0_SHA)}
KEYS = ["AUC@20", "AUC@50", "AUC@100", "AUC@150", "Final"]


def ours_rows():
    return [r for r in csv.DictReader(open(os.path.join(C7.E7, "manifests", "mp3d150_manifest.csv"))) if r["method"] == "OURS"]


def check(args):
    arm, r = args
    tag = f"{r['scene_id']}_S{r['start_id']}"
    p = os.path.join(WI, "raw", arm, "V2M", tag, "episode.json")
    if not os.path.exists(p):
        return {"arm": arm, "row": r, "exists": False}
    mode, sha = ARMS[arm]
    row = dict(r, method="OURS", output_path=os.path.relpath(p, R))
    C7.T0_SHA = sha                                         # the only substitution: expected checkpoint sha of the arm
    try:
        o = C7.validate(row)
    except Exception as e:                                  # e.g. an unfinished / truncated episode
        return {"arm": arm, "row": r, "exists": True, "valid": False, "checks": {"validate_error": repr(e)}, "metrics": None}
    ref = json.load(open(os.path.join(C7.W7, "episodes", "mp3d150", "OURS", f"{tag}.json")))
    o["checks"]["V4_same_start"] = o["start"] == ref["start_pose"]
    d = os.path.dirname(p)
    err = open(os.path.join(d, "planner.err")).read()
    gl = [json.loads(x[len("G_MODE "):]) for x in err.splitlines() if x.startswith("G_MODE ")]
    info = json.load(open(os.path.join(d, "planner_info.json")))
    o["checks"]["G_mode_K1"] = bool(gl) and gl[0]["g_mode"] == mode and gl[0]["K"] == 1 and info.get("g_k") == 1 and info.get("g_mode") == mode
    o["valid"] = all(o["checks"].values())
    o.update({"arm": arm, "exists": True})
    o.pop("executed", None)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    rows = ours_rows()
    with Pool(a.workers) as pool:
        res = pool.map(check, [(arm, r) for arm in ARMS for r in rows])
    ep = {}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "scene_id", "start_id", "split", "exists", "valid", "checks", "replay_max_abs_diff", "decision_s_mean", "wall_s"] + KEYS)
        for o in res:
            r = o["row"]
            if o["exists"] and o["valid"]:
                ep[(o["arm"], r["scene_id"], r["start_id"])] = o["metrics"]
            w.writerow([o["arm"], r["scene_id"], r["start_id"], r["split"], o["exists"], o.get("valid"), json.dumps(o.get("checks")), o.get("replay_max_abs_diff"),
                        o.get("decision_s_mean"), o.get("wall_s")] + [(o.get("metrics") or {}).get(k) for k in KEYS])
    for r in rows:                                            # G4 = archived EXP7 OURS canonical records
        g = json.load(open(os.path.join(C7.W7, "episodes", "mp3d150", "OURS", f"{r['scene_id']}_S{r['start_id']}.json")))
        if g["valid"]:
            ep[("G4", r["scene_id"], r["start_id"])] = g["metrics"]
    need = {}
    for r in rows:
        need.setdefault(r["scene_id"], set()).add(r["start_id"])
    split = {r["scene_id"]: r["split"] for r in rows}
    lv = {}
    for arm in ("G4", "G1IN", "D0IN"):
        for s, st in need.items():
            if all((arm, s, i) in ep for i in st):
                lv[(arm, s)] = {k: float(np.mean([ep[(arm, s, i)][k] for i in st])) for k in KEYS}
    groups = {"ALL-22": sorted(need), "HO-12": sorted(s for s in need if split[s] == "HO"), "VALSEL-10": sorted(s for s in need if split[s] == "VALSEL")}
    n_valid = {arm: sum(1 for o in res if o["arm"] == arm and o.get("valid")) for arm in ARMS}
    n_exist = {arm: sum(1 for o in res if o["arm"] == arm and o["exists"]) for arm in ARMS}
    if not a.allow_incomplete and any(n_valid[x] < 44 for x in ARMS):
        print("valid episodes", n_valid, "existing", n_exist, "(use --allow-incomplete for a partial table)")
        for o in res:
            if o["exists"] and not o["valid"]:
                print("INVALID", o["arm"], o["row"]["scene_id"], o["row"]["start_id"], o["checks"])
        sys.exit(1)
    summ, pair = [], []
    md = ["# G2 item 5 — MP3D-150 indoor closed loop (protocol G2-v1 §5)", "",
          f"Valid episodes: {n_valid} of 44 each (existing {n_exist}); G4 = archived EXP7 OURS. Level-equal means; paired level bootstrap (10,000, seed 0).", ""]
    for g, L in groups.items():
        md += [f"## {g}", "", "| Arm | levels | " + " | ".join(KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
        for arm in ("D0IN", "G1IN", "G4"):
            done = [s for s in L if (arm, s) in lv]
            v = {k: float(np.mean([lv[(arm, s)][k] for s in done])) if done else None for k in KEYS}
            summ.append([g, arm, len(done), len(L)] + [v[k] for k in KEYS])
            md.append(f"| {arm} | {len(done)}/{len(L)} | " + " | ".join("—" if v[k] is None else f"{v[k]:.3f}" for k in KEYS) + " |")
        md += ["", "| Paired | levels | " + " | ".join(KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
        for A, B in (("G4", "D0IN"), ("G1IN", "D0IN"), ("G4", "G1IN")):
            both = [s for s in L if (A, s) in lv and (B, s) in lv]
            if not both:
                continue
            cells, prow = [], [g, A, B, len(both)]
            for k in KEYS:
                dl = [lv[(A, s)][k] - lv[(B, s)][k] for s in both]
                lo, hi = C7.boot(dl)
                cells.append(f"{np.mean(dl):+.3f} [{lo:+.3f}, {hi:+.3f}] ({sum(x > 0 for x in dl)}/{len(dl)})")
                prow += [float(np.mean(dl)), lo, hi, int(sum(x > 0 for x in dl))]
            pair.append(prow)
            md.append(f"| {A} − {B} | {len(both)} | " + " | ".join(cells) + " |")
        md.append("")
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["group", "arm", "levels", "levels_total"] + KEYS] + summ)
    with open(os.path.join(OUT, "paired.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["group", "A", "B", "levels"] + [f"{x}_{k}" for k in KEYS for x in ("d", "lo", "hi", "wins")]] + pair)
    allp = {(p[0], p[1], p[2]): p for p in pair}
    h = allp.get(("ALL-22", "G4", "D0IN"))
    h1 = allp.get(("ALL-22", "G1IN", "D0IN"))
    if h and h1:
        i = 4 + 4 * KEYS.index("AUC@150")
        ok = h[i + 1] > 0 and h1[i] >= 0
        bad = h[i + 2] < 0
        md += ["## H5（预注册）", "", f"G4 − D0 AUC@150 = {h[i]:+.3f} [{h[i + 1]:+.3f}, {h[i + 2]:+.3f}]; G1 − D0 = {h1[i]:+.3f} [{h1[i + 1]:+.3f}, {h1[i + 2]:+.3f}] → "
               f"**{'支持' if ok else ('反对' if bad else '不确定')}**", ""]
    open(os.path.join(OUT, "G2_INDOOR.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
