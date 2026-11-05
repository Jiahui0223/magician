"""EXP6 V3 / ablation E -- evaluation-side GT counterfactuals of a finished closed-loop episode (MAGICIAN env, 1 GPU; offline
diagnostic ONLY: nothing here is ever read by a planner, and no planner/Table 1 file is written).
Replays the episode's executed actions with the frozen M-protocol EpisodeSim (m_t6_sim; asserts the stored official coverage
and legal lists at every step) and, at each step t, for every legal candidate c (snapshot / restore of camera + covered
scene + RNG = the frozen EXP3b m_gain_eval procedure; the frame of c is processed with the seeds of step t+1, exactly as if c
were executed):
  dcov_pp    official coverage gain of c (percentage points); for the executed c it must equal cov[t+1] - cov[t] (asserted)
  gt_value   the H1-H3 value of the TRUE view of c, same definition as h_policy.value_cov: S10 sample points of c farther
             than eps_cov = 1.0 from every observed S10 point of frames 0..t, unique cells of 1.0 (runtime frame), weight
             min(1, (z / z0)^2) with z = camera-frame depth, max per cell, summed (prediction units, except that the GT uses
             the ~10 % S10 pixel sample and the planner the 64 x 114 scoring rays -> compare ranks / ratios, not raw sizes)
  novel cell ids of c (for spatial comparison with predicted novel cells)
  with --two-step and a planner log of the H3 family: the planner's own imagined second view c2* of c (H3 per_c1.best_c2,
  D per_root.best_path[1]): c2*_legal = c2* in the simulator legal list after executing c (GT collision + unvisited-first);
  the true two-step coverage gain (c then c2*, seeds t+1 / t+2) and the true union value (cells of c and c2*).
Also: per-GT-point covered masks at t in {10, 20, ..., 100} (bit-packed, MAGICIAN cell order; the mean / visibility ratio
equals the official coverage, asserted) and the GT points themselves (for region maps).
Usage: $MAGICIAN_PYTHON e_oracle.py --episode EP.json [--planner-log steps.jsonl] [--two-step] --out OUT.npz"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(X, "exp3b_macaronspp_transfer"))
import m_t6_sim as T6S  # noqa: E402
import m_gain_eval as GE  # noqa: E402
import m_magician_replay as RP  # noqa: E402
import m_t6_common as T6  # noqa: E402

np, torch, P = RP.np, RP.torch, RP.P
EPS_COV, CELL = 1.0, 1.0
MASK_T = tuple(range(10, 101, 10))


def z0_of(scene):
    st = json.load(open(os.path.join(P.scene_dir(scene), "settings.json")))["scene"]
    return sum((b - a) * 10.0 for a, b in zip(st["x_min"], st["x_max"])) / 3.0 / 2.0


def gt_value(pc, R, T, tree, z0, gtree=None, out=None):
    """value of the true view (see module doc); if gtree (KD-tree of the GT evaluation points) is given, out[...] also gets the
    part of the value whose novel points lie within eps_cov of a GT evaluation point ("relevant": can raise the official
    coverage) and the rest ("irrelevant": e.g. ground / surfaces outside the evaluated GT set)."""
    p = pc.double().cpu().numpy().reshape(-1, 3)
    if out is not None:
        out.update(gt_value_rel=0.0, gt_value_irr=0.0)
    if len(p) == 0:
        return 0.0, np.zeros(0, np.int64), np.zeros(0), 0
    z = (p @ R[0].double().cpu().numpy() + T[0].double().cpu().numpy())[:, 2]
    d, _ = tree.query(p, k=1, distance_upper_bound=EPS_COV)
    nov = ~np.isfinite(d)
    if not nov.any():
        return 0.0, np.zeros(0, np.int64), np.zeros(0), 0
    p, z = p[nov], z[nov]
    w = np.minimum((z / z0) ** 2, 1.0)
    c = np.floor(p / CELL).astype(np.int64) + (1 << 19)
    key = (c[:, 0] << 40) | (c[:, 1] << 20) | c[:, 2]
    ids, inv = np.unique(key, return_inverse=True)
    wv = np.zeros(len(ids))
    np.maximum.at(wv, inv, w)
    if gtree is not None and out is not None:
        dg, _ = gtree.query(p, k=1, distance_upper_bound=EPS_COV)
        rel = np.isfinite(dg)
        for nm, m in (("gt_value_rel", rel), ("gt_value_irr", ~rel)):
            if m.any():
                i2, inv2 = np.unique(key[m], return_inverse=True)
                w2 = np.zeros(len(i2))
                np.maximum.at(w2, inv2, w[m])
                out[nm] = float(w2.sum())
    return float(wv.sum()), ids, wv, int(nov.sum())


def union_w(a_ids, a_w, b_ids, b_w):
    ids = np.concatenate([a_ids, b_ids])
    w = np.concatenate([a_w, b_w])
    if len(ids) == 0:
        return 0.0
    u, inv = np.unique(ids, return_inverse=True)
    wv = np.zeros(len(u))
    np.maximum.at(wv, inv, w)
    return float(wv.sum())


def covered_mask(gt_scene, covered, eps):
    out = []
    for key in gt_scene.cells:
        gt = gt_scene.cells[key].cell_pts
        if len(gt) == 0:
            continue
        rec = covered.cells[key].cell_pts
        if len(rec) > 0:
            d = torch.min(torch.cdist(gt.double(), rec.double(), p=2.0), dim=-1)[0]
            out.append((eps - d > 0).cpu().numpy())
        else:
            out.append(np.zeros(len(gt), bool))
    return np.concatenate(out)


def gt_points(gt_scene):
    return np.concatenate([gt_scene.cells[k].cell_pts.cpu().numpy() for k in gt_scene.cells if len(gt_scene.cells[k].cell_pts) > 0])


def plan_c2(step_log):
    """first-step-candidate index -> imagined second view (pose key string) of the H3 family, else {}."""
    a = (step_log or {}).get("all_scores") or {}
    out = {}
    if "h3" in a:
        for i, r in enumerate(a["h3"].get("per_c1", [])):
            if r.get("best_c2"):
                out[i] = str(list(r["best_c2"]))
    elif "d" in a:
        for i, r in enumerate(a["d"].get("per_root", [])):
            bp = r.get("best_path") or []
            if len(bp) > 1:
                out[i] = str(list(bp[1]))
    return out


def main():
    from scipy.spatial import cKDTree
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--planner-log", default=None)
    ap.add_argument("--two-step", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmax", type=int, default=None, help="smoke only: stop after observing step tmax")
    a = ap.parse_args()
    t_all = time.time()
    torch.use_deterministic_algorithms(True, warn_only=True)
    ep = json.load(open(a.episode))
    s, k, st = ep["scene"], int(ep["start"]), ep["steps"]
    logs = {}
    if a.planner_log:
        for line in open(a.planner_log):
            r = json.loads(line)
            logs[r["t"]] = r
    tmp = a.out + f".frames_{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    sim = T6S.EpisodeSim(s, k, tmp)
    z0 = z0_of(s)
    gpts = gt_points(sim.gt_scene)
    gtree = cKDTree(gpts.astype(np.float64))
    rows, ids_all, w_all, masks = [], [], [], {}
    obs = []
    idx_of = sim.camera.get_idx_from_key
    pending = None                                                   # (t, executed dcov) to check against the next observation
    for t in range(len(st) if a.tmax is None else a.tmax + 1):
        ob = sim.observe(t)
        assert ob["cov"] == st[t]["cov"], ("coverage replay differs", t, ob["cov"], st[t]["cov"])
        if pending is not None:
            assert abs(pending[1] - 100.0 * (ob["cov"] - st[pending[0]]["cov"])) < 1e-9, ("executed counterfactual differs", pending)
        obs.append(ob["part_pc"].double().cpu().numpy().reshape(-1, 3))
        if t in MASK_T:
            m = covered_mask(sim.gt_scene, sim.covered, sim.eps)
            assert abs(m.mean() / sim.vis - ob["cov"]) < 1e-9, ("covered mask differs from the official coverage", t)
            masks[t] = np.packbits(m)
        if t == len(st) - 1 or t == a.tmax or st[t].get("hover"):
            if st[t].get("hover"):
                sim.move(None)
            continue
        legal, _ = sim.legal()
        keys = [T6.key_of(x[0]) for x in legal]
        assert keys == st[t]["legal_keys"], ("legal list differs", t)
        ex = keys.index(st[t]["action_key"])
        tree = cKDTree(np.concatenate(obs))
        c2s = plan_c2(logs.get(t)) if a.two_step else {}
        snap = GE.snapshot(sim.camera, sim.covered)
        tt = time.time()
        cands = []
        for j, key in enumerate(keys):
            cov_scene = GE.restore(sim.camera, snap)
            sim.camera.update_camera(idx_of(key), interpolation_step=1)
            sim.camera.capture_image(sim.mesh)
            cov_a, frame, R, T, fov, pc = GE.process_frame(sim.camera, sim.params, sim.dev, s, k, t + 1, cov_scene, sim.gt_scene, sim.eps, sim.vis)
            extra = {}
            val, ids, wv, n_nov = gt_value(pc, R, T, tree, z0, gtree, extra)
            c = {**extra, "key": key, "dcov_pp": 100.0 * (cov_a - ob["cov"]), "gt_value": val, "gt_cells": int(len(ids)), "gt_novel_pts": n_nov,
                 "gt_pts": int(len(pc)), "cell_off": int(sum(len(x) for x in ids_all))}
            ids_all.append(ids)
            w_all.append(wv)
            if j in c2s:
                c2 = c2s[j]
                leg2, _ = sim.legal()
                keys2 = [T6.key_of(x[0]) for x in leg2]
                c["c2"] = c2
                c["c2_legal"] = c2 in keys2
                c["c2_n_legal"] = len(keys2)
                try:
                    sim.camera.update_camera(idx_of(c2), interpolation_step=1)
                    sim.camera.capture_image(sim.mesh)
                    cov_b, _, R2, T2, _, pc2 = GE.process_frame(sim.camera, sim.params, sim.dev, s, k, t + 2, cov_scene, sim.gt_scene, sim.eps, sim.vis)
                    _, ids2, wv2, _ = gt_value(pc2, R2, T2, tree, z0)
                    c["two_dcov_pp"] = 100.0 * (cov_b - ob["cov"])
                    c["two_union_value"] = union_w(ids, wv, ids2, wv2)
                    c["c2_value_alone"] = float(wv2.sum())
                except Exception as e:                                          # pose outside the pose space etc.
                    c["c2_error"] = repr(e)[:200]
            cands.append(c)
            del cov_scene
        sim.covered = GE.restore(sim.camera, snap)
        del snap
        rows.append({"t": t, "cov": ob["cov"], "pose_key": st[t]["pose_key"], "executed": ex, "n_legal": len(keys), "cands": cands,
                     "eval_s": time.time() - tt})
        pending = (t, cands[ex]["dcov_pp"])
        sim.move(st[t]["action_key"])
        if t % 10 == 0:
            print(f"t={t} cov={ob['cov']:.4f} n={len(keys)} {time.time() - tt:.1f}s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    np.savez_compressed(a.out, gt_points=gpts.astype(np.float32), cell_ids=np.concatenate(ids_all) if ids_all else np.zeros(0, np.int64),
                        cell_w=np.concatenate(w_all) if w_all else np.zeros(0), **{f"covered_t{t}": v for t, v in masks.items()})
    json.dump({"episode": os.path.abspath(a.episode), "planner_log": a.planner_log, "scene": s, "start": k, "z0": z0, "eps_cov": EPS_COV,
               "cell": CELL, "vis": float(sim.vis), "n_gt_points": int(len(gpts)), "two_step": a.two_step, "seconds": time.time() - t_all,
               "steps": rows}, open(a.out.replace(".npz", ".json"), "w"), default=float)
    print("oracle done", a.out, f"{time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
