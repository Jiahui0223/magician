"""Evaluate official completion-test outputs with the frozen exp1 metrics (exp2 onwards).

Input: a run root written by experiments/exp2_baseline/run_completion.py, i.e.
  <run>/shard*/<timestamp>/patch_completion_*/<scene>_<bbox>/
      volume_tsdf (Generation)_k.npy, mesh_tsdf (Generation)_k.ply, volume_tsdf (Input).npy,
      volume_tsdf (Target).npy, meta_.json
For every patch and world k (metrics_known.py, frozen in exp1-task2):
- sanity: no NaN/inf; the exported target equals our GT (structure and band values);
- voxel metrics (band IoU, TSDF L1) in the known / observed / unobserved regions;
- mesh metrics (Chamfer-L1 mm, F@1, F@2) against the GT mesh_tsdf.ply, per region;
- observation violation (free / surface).
Per patch, over the K worlds: pairwise Chamfer-L1 (mm) and the fraction of unobserved known
voxels whose surface state differs between worlds (diversity).
It also writes <run>/gen/, symlinks to all patch folders, for the official
compute_iou_l1_l2_tmd.py (gen_dir), and runs it when --official is given.

Usage:
  python -m seen2scene.eval.eval_completion --run work_dirs/exp2_baseline/task1_official/runs/dev50_v2 --split dev --official
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import trimesh

from seen2scene.eval import metrics_known as mk

GT_ROOT = "work_dirs/exp1_infra/task2_gt_eval/gt"


DONE_MARKERS = ("image_grid.png", "completed.json", "failure.json")


def patch_dirs(run):
    """Finished patch folders only. The official test step writes image_grid.png last; with
    --task.skip-render it writes completed.json instead, and failure.json if generation failed
    (--task.record-failures). A folder without any of them is still being exported (or its process
    died) and is skipped."""
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(run, "shard*", "*", "patch_completion_*", "*",
                                                                       "volume_tsdf (Target).npy"))
                  if any(os.path.exists(os.path.join(os.path.dirname(p), m)) for m in DONE_MARKERS))


def newest_mtime(d):
    return max(os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d))


def boot(x, n=10_000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if len(x) == 0:
        return [float("nan")] * 3
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def eval_patch(d, gt_root):
    name = os.path.basename(d)
    scene = name[:36]
    gdir = glob.glob(os.path.join(gt_root, f"{scene}_*"))
    assert len(gdir) == 1, (scene, gdir)
    gdir = gdir[0]
    gt = np.load(os.path.join(gdir, "volume.npy"))
    bbox = np.array(json.load(open(os.path.join(gdir, "bbox.json")))["bbox"])
    tgt = np.load(os.path.join(d, "volume_tsdf (Target).npy"))[0]
    inp = np.load(os.path.join(d, "volume_tsdf (Input).npy"))[0]
    s_gt, s_tgt = mk.structure_from_volume(gt), mk.structure_from_volume(tgt)
    band = s_gt == mk.BAND
    r = {"patch": name, "scene": scene,
         "target_equals_gt": bool((s_gt == s_tgt).all() and np.abs(gt[band] - tgt[band]).max() < 1e-6),
         "observed_frac_known": None}
    s_in = mk.structure_from_volume(inp)
    masks = mk.region_masks(s_gt, s_in)
    r["observed_frac_known"] = float(masks["observed"].sum() / max(masks["known"].sum(), 1))
    r["input_coverage"] = mk.surface_coverage(s_in, s_gt)["surface_coverage"]
    gt_mesh = trimesh.load(os.path.join(gdir, "mesh_tsdf.ply"), process=False)
    gt_pts = (mk.sample_points(gt_mesh, 300_000, 1), mk.sample_points(gt_mesh, 2_000_000, 3))
    worlds, meshes, struct_unobs = [], [], []
    ks = sorted(int(p.rsplit("_", 1)[1][:-4]) for p in glob.glob(os.path.join(d, "volume_tsdf (Generation)_*.npy")))
    for k in ks:
        pred = np.load(os.path.join(d, f"volume_tsdf (Generation)_{k}.npy"))[0]
        w = {"k": k, "finite": bool(np.isfinite(pred).all())}
        pred = np.nan_to_num(pred, nan=mk.TRUNCATION)
        w.update(mk.voxel_metrics(pred, gt, masks))
        mpath = os.path.join(d, f"mesh_tsdf (Generation)_{k}.ply")
        m = trimesh.load(mpath, process=False) if os.path.exists(mpath) else trimesh.Trimesh()
        w["n_faces"] = int(len(m.faces))
        pa = (mk.sample_points(m, 300_000, 0), mk.sample_points(m, 2_000_000, 2)) if len(m.faces) else None
        if pa is None:
            w.update({f"{rg}/{kk}": float("nan") for rg in masks for kk in ("chamfer_l1_mm", "F@1", "F@2")})
        else:
            w.update(_mesh_metrics(pa, gt_pts, bbox[:3], masks))
        w.update(mk.observation_violation(pred, inp))
        worlds.append(w)
        meshes.append(m)
        struct_unobs.append(mk.structure_from_volume(pred)[masks["unobserved"]] == mk.BAND)
    r["worlds"] = worlds
    if len(meshes) > 1:
        r["pairwise_chamfer_l1_mm"] = mk.pairwise_chamfer_l1_mm([m for m in meshes if len(m.faces)])
        diffs = [float((struct_unobs[i] != struct_unobs[j]).mean()) for i in range(len(ks)) for j in range(i + 1, len(ks))]
        r["unobserved_state_disagreement"] = float(np.mean(diffs)) if diffs else float("nan")
    return r


def _mesh_metrics(pa, pb, bbox_min, masks):
    """mk.mesh_metrics with pre-sampled (query, reference) point sets."""
    from scipy.spatial import cKDTree
    d_ab = cKDTree(pb[1]).query(pa[0], workers=-1)[0]
    d_ba = cKDTree(pa[1]).query(pb[0], workers=-1)[0]
    out = {}
    for name, m in masks.items():
        ka = mk._points_in_mask(pa[0], m, np.asarray(bbox_min), mk.VOXEL_SIZE)
        kb = mk._points_in_mask(pb[0], m, np.asarray(bbox_min), mk.VOXEL_SIZE)
        if ka.sum() == 0 or kb.sum() == 0:
            out.update({f"{name}/chamfer_l1_mm": float("nan"), f"{name}/F@1": float("nan"), f"{name}/F@2": float("nan")})
            continue
        acc, comp = d_ab[ka], d_ba[kb]
        out[f"{name}/chamfer_l1_mm"] = 1000 * (acc.mean() + comp.mean()) / 2
        for t in (1, 2):
            p, rr = (acc < t * mk.VOXEL_SIZE).mean(), (comp < t * mk.VOXEL_SIZE).mean()
            out[f"{name}/F@{t}"] = 2 * p * rr / (p + rr) if p + rr > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--official", action="store_true", help="also run compute_iou_l1_l2_tmd.py (cd l2 tmd)")
    args = ap.parse_args()
    gt_root = os.path.join(GT_ROOT, args.split)
    dirs = patch_dirs(args.run)
    failed = [d for d in dirs if os.path.exists(os.path.join(d, "failure.json"))]
    dirs = [d for d in dirs if d not in failed]
    out_dir = os.path.join(args.run, "eval")
    os.makedirs(os.path.join(out_dir, "per_patch"), exist_ok=True)
    rows = []
    for d in dirs:
        f = os.path.join(out_dir, "per_patch", os.path.basename(d) + ".json")
        if os.path.exists(f) and os.path.getmtime(f) >= newest_mtime(d):  # cache is valid only if newer than the outputs
            rows.append(json.load(open(f))); continue
        r = eval_patch(d, gt_root)
        json.dump(r, open(f, "w"), indent=1)
        rows.append(r)
        w0 = r["worlds"][0]
        print(f"{r['scene'][:8]}: known F@1 {w0['known/F@1']:.3f} unobs F@1 {w0['unobserved/F@1']:.3f} "
              f"finite {all(w['finite'] for w in r['worlds'])} target==gt {r['target_equals_gt']}", flush=True)
    # summary: per patch, average over worlds; then mean and bootstrap CI over patches
    keys = [k for k in rows[0]["worlds"][0] if k not in ("k", "finite") and not k.endswith("n_voxels")]
    summ = {"n_failed": len(failed), "failed": [os.path.basename(d) for d in failed],
            "n_patches": len(rows), "n_worlds": sum(len(r["worlds"]) for r in rows),
            "all_finite": all(w["finite"] for r in rows for w in r["worlds"]),
            "empty_meshes": sum(w["n_faces"] == 0 for r in rows for w in r["worlds"]),
            "target_equals_gt": f"{sum(r['target_equals_gt'] for r in rows)}/{len(rows)}"}
    for k in keys:
        summ[k] = boot([np.nanmean([w[k] for w in r["worlds"]]) for r in rows])
    for k in ("pairwise_chamfer_l1_mm", "unobserved_state_disagreement", "observed_frac_known", "input_coverage"):
        if k in rows[0]:
            summ[k] = boot([r[k] for r in rows])
    # official metrics
    gen = os.path.join(args.run, "gen")
    os.makedirs(gen, exist_ok=True)
    for d in dirs:
        link = os.path.join(gen, os.path.basename(d))
        if os.path.lexists(link) and os.path.realpath(link) != os.path.realpath(d):
            os.remove(link)  # stale link, e.g. to an interrupted copy that was moved away
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(d), link)
    for link in glob.glob(os.path.join(gen, "*")):  # links to folders that no longer exist
        if not os.path.exists(link):
            os.remove(link)
    if args.official:
        off = os.path.join(out_dir, "official_metrics.json")
        cmd = [sys.executable, "-m", "seen2scene.eval.compute_iou_l1_l2_tmd", "--gen-dir", gen, "--gt-dir", gt_root,
               "--metrics", "cd", "l1", "l2", "tmd", "--output", off]
        rc = subprocess.run(cmd).returncode
        summ["official_returncode"] = rc
        if os.path.exists(off):
            summ["official"] = json.load(open(off))
    json.dump({"summary": summ}, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    for k, v in summ.items():
        if isinstance(v, list) and v and isinstance(v[0], float):
            print(f"{k:32s} {v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]")
        elif k != "official":
            print(f"{k:32s} {v}")


if __name__ == "__main__":
    main()
