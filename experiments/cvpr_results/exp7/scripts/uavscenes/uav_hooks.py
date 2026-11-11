"""EXP7 / UAVScenes -- process-local hooks that let the frozen Macarons++ outdoor chain run on UAV-v0 tiles (no frozen file edited).

* Data root: every process of a UAV run is started with MACARONS_PLUSPLUS_ROOT = work_dirs/cvpr_results/exp7/uav_macpp (the
  exp3b_paths override); tiles are Macarons++-format folders (OBJ + settings.json + occupied_pose.pt). The planner LeakGuard
  therefore forbids every file of the tile except settings.json, as for Macarons++.
* z0: abl_common.z0_of hard-codes the Macarons++ folder; for a `uav_*` scene the same rule (sum of the scene-box extents x 10
  / 3 / 2) is applied to the tile settings.json.
* Frame: t5_diagnostic.frame_of reads the frozen E-M2 table, which has only the 15 Macarons++ scenes. For a `uav_*` scene the
  entry is computed with the frozen E-M2 rule (t1_protocol_audit.py: k_M = scene height (y extent x 10) / 2.816; window edge =
  height, overlap 0.2); Macarons++ scenes still read the frozen table.
"""
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
UAV_ROOT = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "uav_macpp")
UAV_POSE_TABLES = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "uav_pose_tables")
D_CONFIG_UAV = os.path.join(R, "experiments", "cvpr_results", "exp7", "configs", "d_config_uav.yaml")


def em2_rule(settings):
    import numpy as np
    sc = settings["scene"]
    bmin, bmax = 10.0 * np.asarray(sc["x_min"], float), 10.0 * np.asarray(sc["x_max"], float)
    H = float(bmax[1] - bmin[1])
    ext = bmax - bmin
    w = H
    nx = max(1, int(np.ceil((ext[0] - w) / (0.8 * w) + 1 - 1e-9)))
    nz = max(1, int(np.ceil((ext[2] - w) / (0.8 * w) + 1 - 1e-9)))
    return {"height_H": H, "k_M": H / 2.816, "window_edge_runtime_units": w, "voxel_runtime_units": 0.011 * H / 2.816,
            "horizontal_extent_x_z": [float(ext[0]), float(ext[2])], "n_windows_x": nx, "n_windows_z": nz, "n_windows": nx * nz,
            "window_bottom": "scene bbox floor (y_min)", "source": "EXP7 UAV-v0: frozen E-M2 rule applied to the tile settings"}


def install_frame_hook():
    import t5_diagnostic as TD
    MC, P = TD.MC, TD.P
    assert os.path.abspath(P.MACARONS_PLUSPLUS_ROOT) == UAV_ROOT, ("UAV run without the UAV data root", P.MACARONS_PLUSPLUS_ROOT)
    orig = TD.frame_of

    def frame_of(scene):
        if not scene.startswith("uav_"):
            return orig(scene)
        settings = MC.load_json(os.path.join(P.scene_dir(scene), "settings.json"))
        return MC.SceneFrame(scene, settings, em2_rule(settings))
    TD.frame_of = frame_of
    sys.stderr.write(f"UAV_HOOK frame_of (E-M2 rule) data_root={P.MACARONS_PLUSPLUS_ROOT}\n")


def install_z0_hook():
    import json
    import abl_common as AC
    orig = AC.z0_of

    def z0_of(scene):
        if not scene.startswith("uav_"):
            return orig(scene)
        st = json.load(open(os.path.join(UAV_ROOT, scene, "settings.json")))["scene"]
        return sum((b - a) * 10.0 for a, b in zip(st["x_min"], st["x_max"])) / 3.0 / 2.0
    AC.z0_of = z0_of
    sys.stderr.write("UAV_HOOK z0_of (scene-box rule)\n")
