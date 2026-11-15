"""FisherRF-full (f1c4) indoor -- per-episode validation (CPU, seen2scene env; evaluation side, reads GT caches only).

Checks (generalised to N moves so the smoke can use them; for N = 150 the frozen EXP7 collect_mp3d150.validate is ALSO run on
the episode and every one of its checks must pass and agree):
  V1 N + 1 finite coverage values, moves N, executed has N + 1 poses, C non-decreasing
  V2 C_obs replay from the stored 10 % sample points of svc_t0..tN with the level's frozen evaluator cache == stored curve (<= 1e-12)
  V3 legality: pose_idx of svc_t<t> == executed[t] and executed[t+1] in legal_idx of svc_t<t> for every t
  V4 same start pose as the EXP7 OURS canonical record of the level / start
  V6 provenance of the variant: planner_info frontier_mode = magician, reach_filter = true, init_pose = docstring, hello env f1c4
  V7 no planner guard denial in any planner answer; textured-RGB zbuf == sensor zbuf every step (rgb_zbuf_check.json)
Writes <ep-dir>/validation.json; exit 0 iff every check passes.
Usage: python frf_in_validate.py --ep-dir DIR --scene-id LEVEL --start-id K [--moves 150]"""
import argparse
import json
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "exp7", "scripts"))
import collect_mp3d150 as CM  # noqa: E402  (frozen EXP7 validator; only imported)

W7_OURS = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "episodes", "mp3d150", "OURS")


def check(ep_dir, scene_id, start_id, moves):
    e = json.load(open(os.path.join(ep_dir, "episode.json")))
    c = np.asarray(e["coverage"], np.float64)
    ex = [tuple(int(x) for x in a) for a in e["executed"]]
    ck, out = {}, {}
    ck["V1_shape"] = bool(len(c) == moves + 1 and e.get("moves") == moves and len(ex) == moves + 1 and np.all(np.isfinite(c))
                          and np.all(np.diff(c) >= -1e-15))
    z = np.load(os.path.join(CM.COVC, f"{scene_id}.npz"))
    from scipy.spatial import cKDTree
    pts, vis, eps = z["pts"], z["visible"], float(z["eps_rt"])
    cov = np.zeros(len(pts), bool)
    rep, legal_ok = [], True
    for t in range(moves + 1):
        s = np.load(os.path.join(ep_dir, "svc", f"svc_t{t}.npz"))
        sp = np.asarray(s["sample_points"], np.float64)
        if len(sp):
            dd, _ = cKDTree(sp).query(pts, k=1, distance_upper_bound=eps, workers=1)
            cov |= np.isfinite(dd)
        rep.append(float((cov & vis).sum() / max(1, vis.sum())))
        if t < len(ex) and tuple(int(x) for x in s["pose_idx"]) != ex[t]:
            legal_ok = False
        if t < moves and t + 1 < len(ex) and ex[t + 1] not in {tuple(int(x) for x in a) for a in s["legal_idx"]}:
            legal_ok = False
    dmax = float(np.max(np.abs(np.asarray(rep) - c))) if len(c) == moves + 1 else float("inf")
    ck["V2_cobs_replay"] = dmax <= 1e-12
    ck["V3_legal"] = legal_ok
    ours = json.load(open(os.path.join(W7_OURS, f"{scene_id}_S{start_id}.json")))
    ck["V4_same_start_as_OURS"] = bool(ex) and list(ex[0]) == list(ours["start_pose"])
    info = json.load(open(os.path.join(ep_dir, "planner_info.json")))
    ck["V6_variant_f1c4"] = (info.get("frontier_mode") == "magician" and info.get("reach_filter") is True and info.get("init_pose") == "docstring"
                             and info.get("env") == {"FRF_FRONTIER_MODE": "magician", "FRF_REACH_FILTER": "1"} and info.get("variant") == "f1c4")
    denied = [a for line in open(os.path.join(ep_dir, "planner.jsonl")) for a in json.loads(line).get("guard_denied", [])]
    rz = json.load(open(os.path.join(ep_dir, "rgb_zbuf_check.json")))
    ck["V7_guard_and_rgb"] = not denied and rz["all_hit_mask_equal"] and rz["zbuf_maxabs_diff_max"] == 0.0 and rz["n"] == moves + 1
    out.update(replay_max_abs_diff=dmax, guard_denied=denied, rgb_zbuf=dict((k, rz[k]) for k in ("n", "all_hit_mask_equal", "zbuf_maxabs_diff_max")),
               start=list(ex[0]) if ex else None, ours_start=ours["start_pose"])
    if moves == 150:                                                  # the frozen EXP7 validator on the same episode
        row = {"scene_id": scene_id, "start_id": str(start_id), "method": "FISHERRF-FULL", "output_path": os.path.relpath(os.path.join(ep_dir, "episode.json"), R)}
        v = CM.validate(row)
        ck["EXP7_validate_all"] = all(v["checks"].values())
        ck["EXP7_validate_agrees"] = (v["checks"]["V1_shape"] == ck["V1_shape"] and v["checks"]["V2_cobs_replay"] == ck["V2_cobs_replay"]
                                      and v["checks"]["V3_legal"] == ck["V3_legal"] and v["replay_max_abs_diff"] == dmax)
        out["exp7_checks"] = v["checks"]
        out["metrics"] = v["metrics"]
    return ck, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-dir", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--start-id", type=int, required=True)
    ap.add_argument("--moves", type=int, default=150)
    a = ap.parse_args()
    ck, out = check(os.path.abspath(a.ep_dir), a.scene_id, a.start_id, a.moves)
    res = {"tag": f"{a.scene_id}_S{a.start_id}", "moves": a.moves, "valid": all(ck.values()), "checks": ck, **out}
    json.dump(res, open(os.path.join(a.ep_dir, "validation.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("tag", "valid", "checks", "replay_max_abs_diff")}, default=float))
    sys.exit(0 if res["valid"] else 1)


if __name__ == "__main__":
    main()
