"""FisherRF audit A3 -- EVALUATION side (magician env, 1 GPU; GT mesh; OFFLINE diagnosis only, never fed to a planner).

For one recorded FisherRF-full closed loop and the probe JSON of a2_probe_planner.py: replay the executed pose keys with the
frozen M-protocol EpisodeSim (official coverage must equal the stored curve at every step, else STOP -- as t25_out_cov.py /
exp5 ablation s1_oracle_out.py), and at every probe state t evaluate every candidate key of the probe JSON with the frozen
true-gain evaluator (exp3b m_gain_eval: snapshot -> update_camera(key) + capture_image -> process_frame with the seeds of step
t + 1 -> restore):
  gain_mono      newly covered GT points (monotone rule of t25_out_cov) / |GT| / visibility ratio   (primary)
  gain_official  official coverage(t+1 | candidate) - coverage(t)
Candidate positions inside the mesh (MAGICIAN occupied_pose.pt) are flagged ("occupied"; the legal list can contain them, since
get_valid_neighbors does not test occupancy) and evaluated like the others.
The gain of a far (non-neighbour) candidate is the value of its view as if the camera were there at t + 1 (path ignored).
Output: <probe dir>/<scene>_s<k>_gt.json
Usage: python a3_gt_gain.py --scene S --start K --probe DIR [--root closed_loop_f1c4 dir]
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))
import m_t6_sim as T6S  # noqa: E402
import m_gain_eval as GE  # noqa: E402

np, torch = T6S.np, T6S.torch
ROOT = os.path.join(REPO, "work_dirs/cvpr_results/fisherrf_reproduction/full/outdoor/closed_loop_f1c4")


def main():
    from scipy.spatial import cKDTree
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    s, k = a.scene, a.start
    tag = f"{s}_s{k}"
    pr = json.load(open(os.path.join(a.probe, f"{tag}.json")))
    want = {st["t"]: st for st in pr["states"]}
    ep = json.load(open(os.path.join(a.root, tag, "episodes", f"ep_FisherRF_{tag}_r0.json")))
    keys = ep["executed_keys"]
    fr_dir = os.path.join(a.probe, "tmp_frames", f"{tag}_{os.getpid()}")
    shutil.rmtree(fr_dir, ignore_errors=True)
    sim = T6S.EpisodeSim(s, k, fr_dir)
    dev = sim.dev
    gt, eps, vis = sim.gt_scene, sim.eps, sim.vis
    cells = [c for c in gt.cells if len(gt.cells[c].cell_pts) > 0]
    Gp = torch.cat([gt.cells[c].cell_pts for c in cells]).double().cpu().numpy()
    tree = cKDTree(Gp)
    covered = np.zeros(len(Gp), bool)
    occ = sim.camera.pose_is_occupied if getattr(sim.camera, "use_occupied_pose", False) else {}

    def newly(pts):
        new = np.zeros(len(Gp), bool)
        if len(pts):
            for idx in tree.query_ball_point(pts, r=eps * (1 - 1e-12), workers=16):
                new[idx] = True
        return new & ~covered

    out = {"scene": s, "start": k, "states": []}
    t0 = time.time()
    for t in range(max(want) + 1):
        assert sim.key() == keys[t], (t, sim.key(), keys[t])
        ob = sim.observe(t)
        if ob["cov"] != ep["coverage"][t]:
            raise SystemExit(f"STOP: official coverage replay differs {tag} t={t}")
        pts = ob["part_pc"].double().cpu().numpy()
        if len(pts):
            for idx in tree.query_ball_point(pts, r=eps * (1 - 1e-12), workers=16):
                covered[idx] = True
        if t in want:
            tb = time.time()
            cov_t, cov_mono_t = ob["cov"], float(covered.sum()) / len(Gp) / vis
            snap = GE.snapshot(sim.camera, sim.covered)
            res = []
            for c in want[t]["candidates"]:
                key = c["key"]
                pos = str(json.loads(key)[:3])
                is_occ = bool(occ.get(pos, False))
                cs = GE.restore(sim.camera, snap)
                sim.camera.update_camera(sim.camera.get_idx_from_key(key), interpolation_step=1)
                sim.camera.capture_image(sim.mesh)
                cov_a, _, _, _, _, pc = GE.process_frame(sim.camera, sim.params, dev, s, k, t + 1, cs, gt, eps, vis)
                p = pc.double().cpu().numpy()
                res.append({"key": key, "occupied": is_occ, "gain_mono": float(newly(p).sum()) / len(Gp) / vis,
                            "gain_official": float(cov_a - cov_t), "n_pts": int(len(p))})
                del cs
            sim.covered = GE.restore(sim.camera, snap)
            out["states"].append({"t": t, "cov_official_t": cov_t, "cov_mono_t": cov_mono_t, "candidates": res,
                                  "seconds": time.time() - tb})
            g = [r["gain_mono"] for r in res]
            print(tag, t, "n %d max gain %.4f mean %.4f  %.1fs" % (len(g), max(g), np.mean(g), time.time() - tb), flush=True)
        if t + 1 < len(keys):
            sim.move(keys[t + 1])
    shutil.rmtree(fr_dir, ignore_errors=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(a.probe, f"{tag}_gt.json"), "w"), indent=1)
    print("done", tag, "%.0fs" % out["seconds"])


if __name__ == "__main__":
    main()
