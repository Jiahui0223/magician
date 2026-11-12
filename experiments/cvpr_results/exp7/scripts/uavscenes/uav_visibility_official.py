"""EXP7 / UAVScenes -- visibility ratio on the evaluator's own GT points (magician env, 1 GPU). Diagnostic, does not modify settings.

Why: on the sealed v1 tiles the official coverage exceeded 1 (AMvalley_01_v1 max 1.039, monotone 1.047; HKisland_00_v1 monotone
1.007). visibility_ratio (uav_tile_gpu.py step 7) is the visible fraction of 100k AREA-WEIGHTED points, whereas the evaluator
(MAGICIAN setup_test_scene -> gt_scene.fill_cells) keeps a per-cell-capped GT subset (~29k points); the two distributions differ, most
on tiles with large hidden plug walls. A finer union does not change the area-weighted ratio (visibility_fine.json: 0.86389 vs 0.8638).

What: the same observation union as uav_visibility_fine.union_and_gt (all 50 views at every position reachable from open sky by the
official segment rule, all pixels z < 70, cells of 0.05) and, for each official start k = 0..9, the evaluator's GT points of that
start (torch seed derived_seed(tile, k, -1, "gt"), MP.setup_test_scene exactly as m_t6_sim.EpisodeSim) -> fraction within the
evaluator epsilon (2 * TEST_RESOLUTION * scene_scale_factor = 1.0); bounds with eps -/+ the union cell error (0.0433). Also the
area-weighted ratio from the same union. Output <tile>/visibility_official.json. Usage: python -I uav_visibility_official.py TILE"""
import json
import os
import sys
import time

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W7 = os.path.join(R, "work_dirs", "cvpr_results", "exp7")
assert os.path.abspath(os.environ.get("MACARONS_PLUSPLUS_ROOT", "")) == os.path.join(W7, "uav_macpp")
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "exp7", "scripts", "uavscenes"))
import uav_visibility_fine as VF  # noqa: E402  (also puts MAGICIAN_clean and exp5 task0 scripts on the path)

sys.path.insert(0, os.path.join(R, "experiments", "exp3b_macaronspp_transfer"))
import m_t6_sim as T6S  # noqa: E402

RP, MP, MC, torch = T6S.RP, T6S.MP, T6S.MC, T6S.torch


def main():
    from scipy.spatial import cKDTree
    tile = sys.argv[1]
    t0 = time.time()
    cen, Garea, nreach, vr_settings = VF.union_and_gt(tile, 0.05)
    tree = cKDTree(cen)
    err = 0.5 * 0.05 * 3 ** 0.5
    d, _ = tree.query(Garea, k=1, distance_upper_bound=1.0 + err, workers=16)
    res = {"tile": tile, "reachable": nreach, "union_cells": int(len(cen)), "visibility_ratio_settings": vr_settings,
           "area_weighted_fine": float((d < 1.0).mean()), "starts": {}}
    dev = torch.device("cuda")
    sdir = os.path.join(W7, "uav_macpp", tile)
    params = RP.load_params(os.path.join(T6S.P.MAGICIAN_CLEAN_ROOT, "configs", "macarons", "macarons_default_training_config.json"))
    settings = RP.Settings(json.load(open(os.path.join(sdir, "settings.json"))), dev, params.scene_scale_factor)
    mesh = RP.load_scene(os.path.join(sdir, f"{tile}.obj"), params.scene_scale_factor, dev, mirror=False, mirrored_axis=None)
    eps = 2 * RP.TEST_RESOLUTION * params.scene_scale_factor
    for k in range(10):
        torch.manual_seed(MC.derived_seed(tile, k, -1, "gt"))
        gt, covered, surface, proxy = MP.setup_test_scene(params, mesh, settings, False, dev, mirrored_axis=None, test_resolution=RP.TEST_RESOLUTION)
        G = torch.cat([gt.cells[c].cell_pts for c in gt.cells if len(gt.cells[c].cell_pts) > 0]).double().cpu().numpy()
        del gt, covered, surface, proxy
        d, _ = tree.query(G, k=1, distance_upper_bound=eps + err, workers=16)
        res["starts"][k] = {"n_gt": int(len(G)), "ratio": float((d < eps).mean()), "ratio_lo": float((d < eps - err).mean()),
                            "ratio_hi": float((d < eps + err).mean())}
        torch.cuda.empty_cache()
    res["evaluator_gt_mean"] = float(np.mean([v["ratio"] for v in res["starts"].values()]))
    res["eps"], res["union_error"], res["wall_s"] = eps, err, time.time() - t0
    json.dump(res, open(os.path.join(sdir, "visibility_official.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "starts"}))


if __name__ == "__main__":
    main()
