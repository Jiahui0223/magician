"""EXP5 Task 0 -- planner-side indoor scene frames (seen2scene env; reads only the scene card, never GT).

region_frame(card, center_rt): SceneFrame of the n_win x n_win window region around a camera centre (runtime y-up),
clipped to the level box, height = level height; returns (frame, dims, domain_1d, windows) like
t5_diagnostic.scene_geometry, so P1 fusion, the world models and V2 run unchanged.
"""
import json
import os
import sys

import numpy as np

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task0")
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))
import m_common as MC  # noqa: E402

REGION_SHRINK = 0.99    # keeps the region strictly inside n_win x n_win official windows (overlap 0.2)


def load_card(key):
    return json.load(open(os.path.join(W, "preprocessing", "scenes", f"{key}.json")))


def zup_m_to_rt(p, s_rt):
    p = np.asarray(p, np.float64)
    return s_rt * np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def rt_to_zup_m(p, s_rt):
    p = np.asarray(p, np.float64) / s_rt
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def region_box_m(card, center_rt):
    box = np.asarray(card["box_m"])
    c = rt_to_zup_m(center_rt, card["s_rt"])[:2]
    half = 0.5 * (1 + (card["n_win"] - 1) * 0.8) * card["height_m"] * REGION_SHRINK
    lo, hi = np.maximum(c - half, box[0, :2]), np.minimum(c + half, box[1, :2])
    return np.array([[lo[0], lo[1], box[0, 2]], [hi[0], hi[1], box[1, 2]]])


def frame_of_box(key, box_m, s_rt):
    c = zup_m_to_rt(box_m, s_rt)
    rt_min, rt_max = c.min(0), c.max(0)
    settings = {"scene": {"x_min": (rt_min / MC.SCALE).tolist(), "x_max": (rt_max / MC.SCALE).tolist()}}
    k = (rt_max[1] - rt_min[1]) / (MC.PATCH * MC.VS)
    return MC.SceneFrame(key, settings, {"k_M": k})


def region_geometry(card, center_rt):
    import m_world as MW
    box = region_box_m(card, center_rt)
    fr = frame_of_box(card["key"], box, card["s_rt"])
    dims = MW.grid_dims(fr)
    w = MW.official_windows(fr, MW.vae_factor())
    return fr, dims, MW.domain_1d(fr, dims, w["axis_starts"]), w, box


def z0_of_card(card):
    """MAGICIAN scene_scale / 2 of the LEVEL box (runtime units), as abl_common.z0_of on settings.json."""
    return float(card["z0_rt"])
