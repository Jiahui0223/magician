"""EXP5 Ablation S1 (outdoor) -- EVALUATION side (magician env, 1 GPU; GT mesh, frozen M-protocol EpisodeSim).

Per stored source episode (manifest groups): replay the executed pose keys with the frozen EpisodeSim (as
task2/scripts/t25_out_cov.py: official coverage must equal the stored curve at every step, else STOP; the monotone
coverage of Task 2 -- the primary outdoor evaluator -- is accumulated alongside). At every manifest state t, the T7a
true-action-gain evaluator (exp3b m_gain_eval: snapshot of camera / covered scene / RNG -> for each legal action:
restore, update_camera + capture_image, process_frame with the seeds of step t + 1 -> restore) gives per legal action
  gain_mono      newly covered GT points (monotone rule of t25_out_cov) / |GT| / visibility ratio   (PRIMARY)
  gain_official  official coverage(t+1 | a) - coverage(t)  (per-cell cap, may be negative)              (secondary)
plus t* of the 64 x 114 V2 scoring rays of every legal candidate (T1.3 dev_tstar renderer, jitter 0).
Checks: sim legal keys == stored legal keys; the executed action's sample / coverage equal the stored record t + 1.
Output W/states/outdoor/<sid>/oracle.npz and W/oracle/outdoor/<source>_<tag>.json.
Usage: python s1_oracle_out.py --part i/n
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ab_common as A  # noqa: E402

sys.path.insert(0, os.path.join(A.EXP5, "task1", "scripts"))
import v2_common as LC  # noqa: E402

sys.path.insert(1, LC.EXP3B)
import m_t6_sim as T6S  # noqa: E402
import m_gain_eval as GE  # noqa: E402
import m_t6_common as T6  # noqa: E402

np, torch = T6S.np, T6S.torch


def groups(man):
    g = {}
    for st in man["states"]:
        g.setdefault((st["sources"][0]["source"], st["scene"], st["start"]), []).append(st)
    return sorted(g.items())


def load_mesh(scene, dev):
    import exp3b_paths as P
    from pytorch3d.io import load_obj
    from pytorch3d.structures import Meshes
    assert scene in A.SPLIT_OF, f"GT GUARD: {scene} is not a Dev-3 / Test-3 evaluation scene"
    objs = sorted(glob.glob(os.path.join(P.scene_dir(scene), "*.obj")))
    assert len(objs) == 1, objs
    verts, faces, _ = load_obj(objs[0], load_textures=False, device=dev)
    return Meshes(verts=[verts * 10.0], faces=[faces.verts_idx])


def run_group(src, s, k, states):
    from scipy.spatial import cKDTree
    import t11_labels as TL
    tg = f"{src}_{s}_s{k}_r0"
    ep = json.load(open(os.path.join(A.CL_OUT, src, "episodes", f"ep_{tg}.json")))
    keys = ep["executed_keys"]
    fr_dir = os.path.join(A.W, "tmp_frames", f"{tg}_{os.getpid()}")
    shutil.rmtree(fr_dir, ignore_errors=True)
    sim = T6S.EpisodeSim(s, k, fr_dir)
    dev = sim.dev
    gt, eps, vis = sim.gt_scene, sim.eps, sim.vis
    cells = [c for c in gt.cells if len(gt.cells[c].cell_pts) > 0]
    G = torch.cat([gt.cells[c].cell_pts for c in cells]).double().cpu().numpy()
    tree = cKDTree(G)
    covered = np.zeros(len(G), bool)
    ren = TL.Renderer(load_mesh(s, dev), dev, 0, None, 8)
    k_M = TL.frame_of(s).k_M
    want = {x["t"]: x for x in states}
    summ = {"source": src, "tag": tg, "states": [], "executed_equal": []}

    def mono_new(pts):
        new = np.zeros(len(G), bool)
        if len(pts):
            for idx in tree.query_ball_point(pts, r=eps * (1 - 1e-12), workers=16):
                new[idx] = True
        return new & ~covered

    for t in range(max(want) + 1):
        assert sim.key() == keys[t], (t, sim.key(), keys[t])
        ob = sim.observe(t)
        if ob["cov"] != ep["coverage"][t]:
            raise SystemExit(f"STOP: official coverage replay differs {tg} t={t}")
        pts = ob["part_pc"].double().cpu().numpy()
        if len(pts):
            for idx in tree.query_ball_point(pts, r=eps * (1 - 1e-12), workers=16):
                covered[idx] = True
        if t in want:
            st = want[t]
            tb = time.time()
            legal, _ = sim.legal()
            lk = [T6.key_of(x[0]) for x in legal]
            if lk != ep["steps"][t]["legal_keys"]:
                raise SystemExit(f"STOP: legal keys differ {tg} t={t}")
            rec = np.load(A.svc_path("outdoor", src, {"scene": s, "start": k}, t))
            assert np.array_equal(rec["legal_idx"], np.asarray([x[0] for x in legal]).reshape(-1, 5))
            cov_t, cov_mono_t = ob["cov"], float(covered.sum()) / len(G) / vis
            snap = GE.snapshot(sim.camera, sim.covered)
            g_off, g_mono, cov_as, pts_all, off = [], [], [], [], [0]
            for key in lk:
                cs = GE.restore(sim.camera, snap)
                sim.camera.update_camera(sim.camera.get_idx_from_key(key), interpolation_step=1)
                sim.camera.capture_image(sim.mesh)
                cov_a, _, _, _, _, pc = GE.process_frame(sim.camera, sim.params, dev, s, k, t + 1, cs, gt, eps, vis)
                p = pc.double().cpu().numpy()
                g_off.append(float(cov_a - cov_t))
                cov_as.append(cov_a)
                g_mono.append(float(mono_new(p).sum()) / len(G) / vis)
                pts_all.append(p.astype(np.float32))
                off.append(off[-1] + len(p))
                del cs
            sim.covered = GE.restore(sim.camera, snap)
            ex = ep["steps"][t]["pick"]
            if ex is not None:
                nxt = np.load(A.svc_path("outdoor", src, {"scene": s, "start": k}, t + 1))["sample_points"]
                eq = bool(np.array_equal(pts_all[ex], nxt) and cov_as[ex] == ep["coverage"][t + 1])
                summ["executed_equal"].append(eq)
                if not eq:
                    raise SystemExit(f"STOP: executed action differs from record t+1 ({tg} t={t})")
            tstar = ren.tstar(list(rec["legal_R"]), list(rec["legal_T"]), [(0.0, 0.0)] * len(lk), k_M).astype(np.float32)
            od = A.state_dir("outdoor", st["sid"])
            os.makedirs(od, exist_ok=True)
            np.savez_compressed(os.path.join(od, "oracle.tmp.npz"), gain=np.asarray(g_mono), gain_official=np.asarray(g_off),
                                cov_t=np.asarray(cov_mono_t), cov_official_t=np.asarray(cov_t),
                                remaining=np.asarray(1.0 - covered.sum() / len(G) / vis), tstar=tstar, k_M=np.asarray(k_M),
                                cand_pts=np.concatenate(pts_all) if pts_all else np.zeros((0, 3), np.float32),
                                cand_off=np.asarray(off), legal_idx=rec["legal_idx"], eps=np.asarray(eps), vis=np.asarray(vis))
            os.replace(os.path.join(od, "oracle.tmp.npz"), os.path.join(od, "oracle.npz"))
            summ["states"].append({"sid": st["sid"], "t": t, "cov_mono_t": cov_mono_t, "max_gain": max(g_mono),
                                   "seconds": time.time() - tb})
            print(tg, t, "max gain mono %.4f official %.4f" % (max(g_mono), max(g_off)), "%.1fs" % (time.time() - tb), flush=True)
        if t + 1 < len(keys):
            sim.move(keys[t + 1])
    shutil.rmtree(fr_dir, ignore_errors=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    pi, pn = map(int, a.part.split("/"))
    man = A.load_manifest("outdoor")
    od = os.path.join(A.W, "oracle", "outdoor")
    os.makedirs(od, exist_ok=True)
    for (src, s, k), sts in groups(man)[pi::pn]:
        p = os.path.join(od, f"{src}_{s}_s{k}.json")
        if os.path.exists(p):
            continue
        tb = time.time()
        summ = run_group(src, s, k, sts)
        summ["seconds"] = time.time() - tb
        json.dump(summ, open(p + ".tmp", "w"), indent=1)
        os.replace(p + ".tmp", p)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
