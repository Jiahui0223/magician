"""EXP6 V3 Task 2 / Experiment 2 -- R3 candidate-space counterfactual on the MP3D fixed states (magician env, 1 GPU;
evaluation side only: GT mesh, frozen sensor and C_obs evaluator; protocol §4 R3).

Per Pool CL MP3D state (records 0..t replayed into the frozen Coverage, must reproduce the stored curve):
  * the ORIGINAL legal set (= oracle.npz candidates): true one-step gains (reproduced, |d| <= 1e-12);
  * the EXTENDED set = poses reachable by TWO legal moves (every legal neighbour p1, then t12_sim.legal_actions(p1):
    the official neighbour rule + the GT segment legality, i.e. the environment's own rule), minus the current pose and
    the 1-step set: the true gain of each pose evaluated alone against the current covered state with the frame-(t+1)
    sample seed (no intermediate observation; an optimistic 2-step reach);
  * the frontier map of the records 0..t (planner-side quantity): which extended poses enter cells that are UNKNOWN /
    frontier (new territory) and the share of the remaining surface near them.
Reports max gain 1-step vs 2-step, by stall / non-stall; nothing here changes the online candidate generator.
Output work_dirs/exp6_v3/task2/exp2/r3/<sid>.json;  Usage: python e2_r3_candidates.py --part i/n
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
T2 = os.path.dirname(EXP)
E6 = os.path.dirname(T2)
REPO = os.path.dirname(os.path.dirname(E6))
EXP5 = os.path.join(REPO, "experiments", "exp5_future_view")
sys.path.insert(0, os.path.join(T2, "common"))
sys.path.insert(0, os.path.join(EXP5, "ablation_study", "common"))
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
sys.path.insert(0, os.path.join(EXP5, "task0", "magician_comparison", "scripts"))
sys.path.insert(0, os.path.join(EXP5, "task0", "magician_comparison", "adapters"))
import ab_common as A  # noqa: E402
import t12_sim as T12  # noqa: E402
import grid_env as GE  # noqa: E402
import t2_frontier as TF  # noqa: E402

S = T12.S
WR3 = os.path.join(REPO, "work_dirs", "exp6_v3", "task2", "exp2", "r3")


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    pi, pn = map(int, a.part.split("/"))
    dev = torch.device("cuda")
    man = A.load_manifest("indoor")
    by = {}
    for st in man["states"]:
        by.setdefault((st["sources"][0]["source"], st["tag"]), []).append(st)
    jobs = sorted(by.items())[pi::pn]
    os.makedirs(WR3, exist_ok=True)
    cache = {}
    for (src, tag), sts in jobs:
        todo = [s for s in sts if not os.path.exists(os.path.join(WR3, s["sid"] + ".json"))]
        if not todo:
            continue
        scene = sts[0]["scene"]
        scan, lv = sts[0]["scan"], sts[0]["level"]
        if scene not in cache:
            L = S.Level(scan, lv)
            v, f = S.mesh_zup(scan)
            sen = S.Sensor(S.zup_m_to_rt(v), f, dev)
            card = json.load(open(os.path.join(S.W, "preprocessing", "scenes", f"{L.key}.json")))
            cache[scene] = (L, v, f, sen, card, GE.grid_def(card), T12.Blocker(v, f))
        L, v, f, sen, card, g, blk = cache[scene]
        cov = S.Coverage(L, sen)
        nvis = max(1, int(cov.visible.sum()))
        ep = json.load(open(os.path.join(A.CL_IN, src, tag, "episode.json")))
        fm = TF.FrontierMap(card)
        want = {int(s["t"]): s for s in sts}
        for t in range(max(want) + 1):
            rec = np.load(os.path.join(A.CL_IN, src, tag, "svc", f"svc_t{t}.npz"))
            cov.update(rec["sample_points"])
            if abs(cov.value() - ep["coverage"][t]) > 1e-12:
                raise SystemExit(f"STOP: coverage replay differs {src} {tag} t={t}")
            fm.add_record({k: rec[k] for k in ("cam_center", "R", "T", "zbuf", "mask")})
            if t not in want:
                continue
            st = want[t]
            tb = time.time()
            rem = cov.visible & ~cov.covered
            seed = S.derived_seed(L.key, st["start"], t + 1, "sample")
            pose = tuple(int(x) for x in rec["pose_idx"])
            legal1 = [tuple(int(x) for x in p) for p in rec["legal_idx"]]
            hist = {tuple(int(x) for x in p) for p in rec["executed_pose_idx_history"]}

            def gain_of(p):
                R, T = sen.RT(GE.position(g, p), *GE.angles(g, p))
                fr = sen.render(R, T)
                pts = sen.sample(fr, seed)[0].cpu().numpy()
                return float((cov._near(cov.tree, pts) & rem).sum() / nvis)
            g1 = [gain_of(p) for p in legal1]
            orc = np.load(os.path.join(A.state_dir("indoor", st["sid"]), "oracle.npz"))
            if np.max(np.abs(np.asarray(g1) - np.asarray(orc["gain"]))) > 1e-12:
                raise SystemExit(f"STOP: 1-step gains differ from oracle.npz {src} {tag} t={t}")
            ext = {}
            for p1 in legal1:
                for p2 in T12.legal_actions(g, p1, blk):
                    p2 = tuple(int(x) for x in p2)
                    if p2 != pose and p2 not in legal1 and p2 not in ext:
                        ext[p2] = p1
            g2 = {p: gain_of(p) for p in ext}
            free, occ, unk = fm.states()
            front = fm.frontier(free, unk)
            Gxz = S.rt_to_zup_m(cov.pts)
            def cell_info(p):
                l, h = fm.to_cell(GE.position(g, p))
                return {"cell": [int(l), int(h)], "unknown": bool(unk[l, h]), "frontier": bool(front[l, h]), "free": bool(free[l, h]), "visited": bool(fm.visited[l, h])}
            res = {"sid": st["sid"], "scene": scene, "start": st["start"], "t": t, "source": src, "tag": tag, "pose": list(pose), "cov_t": cov.value(), "remaining": float(rem.sum() / nvis),
                   "n_legal1": len(legal1), "n_ext2": len(ext), "gain1": g1, "gain1_unvisited": [gq for p, gq in zip(legal1, g1) if p not in hist],
                   "gain2": [g2[p] for p in ext], "ext_poses": [list(p) for p in ext], "ext_via": [list(ext[p]) for p in ext],
                   "ext_cells": [cell_info(p) for p in ext], "legal1_cells": [cell_info(p) for p in legal1], "cur_cell": cell_info(pose),
                   "max1": float(max(g1)) if g1 else 0.0, "max2": float(max(g2.values())) if g2 else 0.0,
                   "n_ext_translations": int(sum(1 for p in ext if p[:3] != pose[:3])), "n_ext_new_cells": int(sum(1 for p in ext if cell_info(p)["unknown"])),
                   "frontier": {"n_frontier": int(front.sum()), "n_free": int(free.sum())}, "seconds": time.time() - tb}
            res["best2_cell"] = cell_info(max(g2, key=g2.get)) if g2 else None
            json.dump(res, open(os.path.join(WR3, st["sid"] + ".json.tmp"), "w"), default=float)
            os.replace(os.path.join(WR3, st["sid"] + ".json.tmp"), os.path.join(WR3, st["sid"] + ".json"))
            print(src, tag, t, "max1 %.4f max2 %.4f (%d ext, %d new cells) %.1fs" % (res["max1"], res["max2"], len(ext), res["n_ext_new_cells"], res["seconds"]), flush=True)


if __name__ == "__main__":
    main()
