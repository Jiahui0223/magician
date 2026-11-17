"""EXP9 Fig. 6 v4 -- dump the GT evaluation point sets that Fig. 6 v3 lacked (eiffel s1-4, sestino_museum s1-4; Unseen-6 scenes).
magician env, 1 GPU.  Code path = experiments/exp5_future_view/ablation_study/A1_prediction_scoring/a1_gtpoints_out.py UNCHANGED
(frozen EpisodeSim initialisation, exactly as task2 t25_out_cov: points of the non-empty GT cells, eps, visibility ratio), only the
episode list and the output directory differ (work_dirs/cvpr_results/exp9_ablation_qualitative/fig6_v4/gt_points/<scene>_s<k>.npz,
same keys G / eps / vis).  Validation happens downstream: f6v4_masks.py recomputes the mono coverage curve C[0..20] from these points
and the stored service records and requires it to equal the canonical Table 1 record (<= 1e-9), else STOP.
Usage: CUDA_VISIBLE_DEVICES=0 $MAGPY -I -B f6v4_gtpoints.py"""
import os
import shutil
import sys

A1DIR = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/exp5_future_view/ablation_study/A1_prediction_scoring"
sys.path.insert(0, A1DIR)
import a1_gtpoints_out as G  # noqa: E402  (imports ab_common, v2_common, m_t6_sim exactly as the original)

A, T6S, np, torch = G.A, G.T6S, G.np, G.torch
OUT = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/cvpr_results/exp9_ablation_qualitative/fig6_v4/gt_points"
EPISODES = [(s, k) for s in ("eiffel", "sestino_museum") for k in (1, 2, 3, 4)]


def main():
    os.makedirs(OUT, exist_ok=True)
    for scene, start in EPISODES:
        p = os.path.join(OUT, f"{scene}_s{start}.npz")
        if os.path.exists(p):
            continue
        fr = os.path.join(A.W, "tmp_frames", f"gtp_{scene}_{start}_{os.getpid()}")
        shutil.rmtree(fr, ignore_errors=True)
        sim = T6S.EpisodeSim(scene, start, fr)
        gt = sim.gt_scene
        cells = [c for c in gt.cells if len(gt.cells[c].cell_pts) > 0]
        Gp = torch.cat([gt.cells[c].cell_pts for c in cells]).double().cpu().numpy()
        np.savez_compressed(p, G=Gp.astype(np.float32), eps=np.asarray(sim.eps), vis=np.asarray(sim.vis))
        shutil.rmtree(fr, ignore_errors=True)
        print(scene, start, len(Gp), float(sim.eps), float(sim.vis), flush=True)
        del sim
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
