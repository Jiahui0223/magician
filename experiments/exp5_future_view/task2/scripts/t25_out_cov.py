"""EXP5 Task 2 -- outdoor coverage of the final comparison, evaluation side (magician env, 1 GPU; protocol §5.2).

Replays an episode's executed poses with the frozen M-protocol EpisodeSim (as the EXP4 per-point replay
abl_1/common/abl_cov_replay.py: seeded observation, same GT sample, same eps) and computes per step t = 0..moves
  cov_official  the MAGICIAN-style coverage of the simulator (covered cells capped, points can be replaced -> may
                decrease); must equal the stored episode curve bitwise (else STOP)
  cov_mono      PRIMARY (frozen before the runs): the same GT points, eps and visibility normalisation, but every observed
                10 % sample is kept (no cap): a GT point is covered once any observed point so far lies within eps;
                monotone by construction
Output W2/eval/outdoor/metrics/<group>/<tag>.json.
Usage: python t25_out_cov.py --group G --tag TAG
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.EXP5, "task1", "scripts"))
import v2_common as LC  # noqa: E402

sys.path.insert(1, LC.EXP3B)
import m_t6_sim as T6S  # noqa: E402

np, torch = T6S.np, T6S.torch


def main():
    from scipy.spatial import cKDTree
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = a.root or os.path.join(C.W2, "eval", "outdoor", "closed_loop", a.group)
    out = os.path.join(a.out or os.path.join(C.W2, "eval", "outdoor", "metrics"), a.group, f"{a.tag}.json")
    if os.path.exists(out):
        return
    t0 = time.time()
    ep = json.load(open(os.path.join(root, "episodes", f"ep_{a.tag}.json")))
    s, k, keys = ep["scene"], ep["start"], ep["executed_keys"]
    fr = os.path.join(C.W2, "eval", "outdoor", "replay_frames", f"{a.group}_{a.tag}")
    shutil.rmtree(fr, ignore_errors=True)
    sim = T6S.EpisodeSim(s, k, fr)
    gt, eps, vis = sim.gt_scene, sim.eps, sim.vis
    cells = [c for c in gt.cells if len(gt.cells[c].cell_pts) > 0]
    G = torch.cat([gt.cells[c].cell_pts for c in cells]).double().cpu().numpy()
    tree = cKDTree(G)
    covered = np.zeros(len(G), bool)
    off, mono = [], []
    for t in range(len(keys)):
        assert sim.key() == keys[t], (t, sim.key(), keys[t])
        ob = sim.observe(t)
        off.append(ob["cov"])
        pts = ob["part_pc"].double().cpu().numpy()
        if len(pts):
            for idx in tree.query_ball_point(pts, r=eps * (1 - 1e-12), workers=16):
                covered[idx] = True
        mono.append(float(covered.sum()) / len(G) / vis)
        if t + 1 < len(keys):
            sim.move(keys[t + 1])
    shutil.rmtree(fr, ignore_errors=True)
    equal = off == ep["coverage"]
    if not equal:
        raise SystemExit(f"STOP: official coverage replay differs for {a.group} {a.tag}")
    c = np.asarray(mono)
    moves = ep["moves"]
    res = {"group": a.group, "tag": a.tag, "scene": s, "start": k, "moves": moves, "n_gt": int(len(G)), "eps": float(eps), "vis": float(vis),
           "cov_official": off, "cov_mono": mono, "official_equal_stored": equal,
           "auc_mono": float(np.trapz(c, dx=1.0) / moves), "cov_mono_final": float(c[-1]),
           "auc_official": float(np.trapz(np.asarray(off), dx=1.0) / moves), "cov_official_final": float(off[-1]),
           "official_max_drop": float(max(0.0, max(off[i] - off[j] for i in range(len(off)) for j in range(i, len(off))))),
           "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out + ".tmp", "w"), indent=1)
    os.replace(out + ".tmp", out)
    print(a.group, a.tag, "mono AUC %.4f final %.4f | official AUC %.4f final %.4f | %.0f s" % (
        res["auc_mono"], res["cov_mono_final"], res["auc_official"], res["cov_official_final"], res["seconds"]), flush=True)


if __name__ == "__main__":
    main()
