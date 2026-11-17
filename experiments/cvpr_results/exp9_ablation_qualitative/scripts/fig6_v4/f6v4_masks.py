"""EXP9 Fig. 6 v4 -- per-GT-point observed masks after step 20 for every candidate on the UNSEEN evaluation units (CPU, seen2scene env,
read-only on inputs).  Usage:  $PY -I -B f6v4_masks.py

Cases (user request 2026-11-16: show all six Unseen-6 scenes and unseen indoor levels, then pick):
  outdoor  Unseen-6 (alhambra, neuschwanstein, eiffel, bridge, barts, sestino_museum) x every official start, EXCEPT starts whose pose
           lies inside the mesh (manifest/start_pose_backface_summary.json: alhambra s2, neuschwanstein s2 / s4 -- never used for
           figures) -> 27 episodes
  indoor   HO-12 (the held-out MP3D levels of Table 1 v4) x both starts -> 24 episodes
Mask code = the validated v1 functions scripts/fig6/fig6_masks.py (outdoor_run / indoor_run, imported unchanged) with the v1 / v2 / v3
hard checks: curve C[0..20] recomputed from the stored service records == stored record (<= 1e-9), svc pose_idx == executed keys,
both methods share start and C[0], official Cov@20 == manifests/fig6_cov20_candidates.csv (6 decimals), Ours sha256 == frozen V3-T0.
GT points: start 0 = geometry_roi_audit a1 (as v1-v3); start k > 0 = exp5 ablation gt_points dump, or (eiffel / sestino_museum s1-4,
missing there) the f6v4_gtpoints.py dump of the same frozen code -- validated here by the bitwise curve check.
Cases that exist in v3 are asserted bitwise identical to work_dirs/.../fig6_v3/<case>.npz.
Records: Table 1 canonical (MAG-B, V3-T0-D1) outdoor, EXP7 MP3D-150 (MAG-B, OURS) indoor -- the same records as Table 1 v4.
Output: work_dirs/cvpr_results/exp9_ablation_qualitative/fig6_v4/<case>.npz, v4_cases.csv, f6v4_masks_manifest.csv, f6v4_masks_summary.json
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/"
EXP9 = REPO + "experiments/cvpr_results/exp9_ablation_qualitative/"
sys.path.insert(0, EXP9 + "scripts/fig6")
sys.path.insert(0, EXP9 + "scripts/fig6_v2")
import fig6_masks as V1  # noqa: E402  (validated v1 mask code, unchanged)
import f6v2_masks as V2  # noqa: E402  (v2 path patterns)

CAND = EXP9 + "manifests/fig6_cov20_candidates.csv"
BFS = REPO + "experiments/cvpr_results/manifest/start_pose_backface_summary.json"
V3_DIR = REPO + "work_dirs/cvpr_results/exp9_ablation_qualitative/fig6_v3/"
OUT = REPO + "work_dirs/cvpr_results/exp9_ablation_qualitative/fig6_v4/"
G5 = REPO + "work_dirs/exp5_future_view/ablation_study/gt_points/"
G4V = OUT + "gt_points/"
U6 = ["alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"]
rel = lambda p: p.replace(REPO, "")  # noqa: E731


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def outdoor_gt(scene, start, a1):
    if start == 0:
        src = f"work_dirs/cvpr_results/geometry_roi_audit/a1/{scene}.npz"
        z = np.load(REPO + src)
        roi = a1[scene]["roi_box"]
        return z["gt_points"], float(roi["visibility_ratio"]), float(roi["eps_rt"]), src + ":gt_points", z
    for d in (G5, G4V):
        p = d + f"{scene}_s{start}.npz"
        if os.path.exists(p):
            z = np.load(p)
            return z["G"], float(z["vis"]), float(z["eps"]), rel(p) + ":G", np.load(REPO + f"work_dirs/cvpr_results/geometry_roi_audit/a1/{scene}.npz")
    raise SystemExit(f"STOP: no GT points for {scene} s{start}")


def main():
    os.makedirs(OUT, exist_ok=True)
    inside = set(json.load(open(BFS))["start_pose_inside_mesh"])
    cand = [x for x in csv.DictReader(open(CAND)) if x["row_type"] == "episode"]
    cases = []
    for x in cand:
        if x["domain"] == "outdoor" and x["scene"] in U6:
            key = f"{x['scene']}/{x['start']}"
            if key in inside or f"{x['scene']}_s{x['start']}" in inside or x["start_inside_mesh"] == "True":
                continue
        elif not (x["domain"] == "indoor" and x["split"].startswith("HO")):
            continue
        cases.append({"domain": x["domain"], "scene": x["scene"], "start": x["start"], "split": x["split"],
                      "magb_cov20": x["magb_cov20"], "rayscout_cov20": x["rayscout_cov20"], "delta": x["delta_rayscout_minus_magb"]})
    n_out = sum(c["domain"] == "outdoor" for c in cases)
    n_in = sum(c["domain"] == "indoor" for c in cases)
    assert n_out == 27 and n_in == 24, (n_out, n_in)
    a1 = json.load(open(V1.A1J))
    rows, same_v3 = [], {}
    for r in cases:
        r.update(V2.paths(r))
        name, start = V1.case_name(r), int(r["start"])
        if r["domain"] == "outdoor":
            G32, vis, eps, src, za1 = outdoor_gt(r["scene"], start, a1)
            r["gt_points_source"] = src
            assert eps == 1.0, eps
            G = G32.astype(np.float64)
            runs = [V1.outdoor_run(r[c], r["scene"], start, G, vis, eps) for c in ("magb_record", "rayscout_record")]
            extra = {"box_lo": za1["box_lo"], "box_hi": za1["box_hi"], "visibility_ratio": vis, "eps": eps}
            visible = np.ones(len(G), bool)
        else:
            src, keys = r["gt_points_source"].split(":")
            z = np.load(REPO + src)
            G32, visible, eps = z["pts"], z["visible"].astype(bool), float(z["eps_rt"])
            G = G32.astype(np.float64)
            runs = [V1.indoor_run(r[c], r["scene"], start, G, visible, eps) for c in ("magb_record", "rayscout_record")]
            card = json.load(open(V1.CARDS + r["scene"] + ".json"))
            extra = {"floor_m": card["floor_m"], "ceiling_m": card["ceiling_m"], "height_m": card["height_m"], "k_I": card["k_I"],
                     "s_rt": card["s_rt"], "eps": eps, "box_m": np.asarray(card["box_m"])}
        mag, ours = runs
        if mag["start_key"] != ours["start_key"] or mag["c0"] != ours["c0"]:
            raise SystemExit(f"STOP: {name}: methods do not share the start")
        if ours["sha"] != V1.OURS_SHA:
            raise SystemExit(f"STOP: {name}: Ours checkpoint sha {ours['sha']} != frozen V3-T0 final.pt")
        for run, col in ((mag, "magb_cov20"), (ours, "rayscout_cov20")):
            if abs(run["official"] - float(r[col])) > 5e-7:
                raise SystemExit(f"STOP: {name}: candidates {col} {r[col]} != record {run['official']}")
        if not (mag["bitwise"] and ours["bitwise"]):
            raise SystemExit(f"STOP: {name}: curve recompute not bitwise")
        arrs = dict(gt_points=G32, visible=visible, mask_mag=mag["mask"], mask_ours=ours["mask"],
                    first_mag=mag["first"], first_ours=ours["first"], traj_mag=mag["traj"], traj_ours=ours["traj"],
                    pose_idx_mag=mag["pose_idx"], pose_idx_ours=ours["pose_idx"],
                    cov20_official_mag=mag["official"], cov20_official_ours=ours["official"],
                    cov20_recomputed_mag=mag["recomputed"], cov20_recomputed_ours=ours["recomputed"],
                    domain=r["domain"], scene=r["scene"], start=start, step=V1.T, **extra)
        if os.path.exists(V3_DIR + name + ".npz"):
            o = np.load(V3_DIR + name + ".npz")
            if not (sorted(o.files) == sorted(arrs) and all(np.array_equal(o[k], np.asarray(arrs[k])) for k in o.files)):
                raise SystemExit(f"STOP: {name}: recomputed arrays differ from the v3 npz")
            same_v3[name] = True
        np.savez_compressed(OUT + name + ".npz", **arrs)
        for meth, run in (("MAGICIAN (MAG-B)", mag), ("Ours (V3-T0 + D1)", ours)):
            rows.append({"case": name, "domain": r["domain"], "scene": r["scene"], "start": start, "split": r["split"],
                         "method": meth, "step": V1.T, "n_gt": len(G), "n_observed": int(run["mask"].sum()),
                         "official_cov20": f"{run['official']:.12f}", "official_metric": run["official_metric"],
                         "recomputed_cov20": f"{run['recomputed']:.12f}", "check_metric": run["check_metric"],
                         "curve_0_20_max_abs_diff": f"{run['maxdiff']:.3e}", "curve_0_20_bitwise_equal": run["bitwise"],
                         "v3_npz_bitwise_identical": same_v3.get(name, "n/a (new case)"), "start_key": run["start_key"],
                         "c0": f"{run['c0']:.12f}", "ckpt_sha256": run["sha"],
                         "record": r["magb_record"] if run is mag else r["rayscout_record"], "episode_file": run["episode_file"],
                         "svc_dir": run["svc"], "gt_source": r["gt_points_source"], "npz": rel(OUT + name + ".npz")})
        print(f"{name:34s} MAG {mag['official']:.4f} | Ours {ours['official']:.4f} | bitwise {mag['bitwise']}/{ours['bitwise']} | "
              f"v3 identical {same_v3.get(name, 'new')}", flush=True)
    with open(OUT + "v4_cases.csv", "w", newline="") as f:
        keys = ["domain", "scene", "start", "split", "magb_cov20", "rayscout_cov20", "delta", "gt_points_source"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(cases)
    with open(OUT + "f6v4_masks_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump({"mask_code": rel(EXP9 + "scripts/fig6/fig6_masks.py"), "mask_code_sha256": sha256(EXP9 + "scripts/fig6/fig6_masks.py"),
               "candidates_csv_sha256": sha256(CAND), "n_cases": len(cases), "n_outdoor": n_out, "n_indoor": n_in,
               "all_bitwise": all(x["curve_0_20_bitwise_equal"] for x in rows), "v3_cases_bitwise_identical": same_v3},
              open(OUT + "f6v4_masks_summary.json", "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
