"""EXP5 Ablation A1 diagnostic (DEVIATIONS #8; magician env, 1 GPU; evaluation side): dump the GT evaluation point set of
every outdoor episode (frozen EpisodeSim initialisation, exactly as task2 t25_out_cov: points of the non-empty GT cells,
eps, visibility ratio). Output W/gt_points/<scene>_s<start>.npz
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import ab_common as A  # noqa: E402

sys.path.insert(0, os.path.join(A.EXP5, "task1", "scripts"))
import v2_common as LC  # noqa: E402

sys.path.insert(1, LC.EXP3B)
import m_t6_sim as T6S  # noqa: E402

np, torch = T6S.np, T6S.torch


def main():
    od = os.path.join(A.W, "gt_points")
    os.makedirs(od, exist_ok=True)
    for ep in A.episodes("outdoor"):
        p = os.path.join(od, f"{ep['scene']}_s{ep['start']}.npz")
        if os.path.exists(p):
            continue
        fr = os.path.join(A.W, "tmp_frames", f"gtp_{ep['scene']}_{ep['start']}_{os.getpid()}")
        shutil.rmtree(fr, ignore_errors=True)
        sim = T6S.EpisodeSim(ep["scene"], ep["start"], fr)
        gt = sim.gt_scene
        cells = [c for c in gt.cells if len(gt.cells[c].cell_pts) > 0]
        G = torch.cat([gt.cells[c].cell_pts for c in cells]).double().cpu().numpy()
        np.savez_compressed(p, G=G.astype(np.float32), eps=np.asarray(sim.eps), vis=np.asarray(sim.vis))
        shutil.rmtree(fr, ignore_errors=True)
        print(ep["scene"], ep["start"], len(G), flush=True)
        del sim
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
