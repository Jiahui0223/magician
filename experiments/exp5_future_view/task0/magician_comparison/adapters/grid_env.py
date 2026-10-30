"""EXP5 Task 0 / T0.12 -- the released-code-style camera grid on a Task 0 level (protocol v1.3 §19). A pure function of
the scene card (level box, floor, ceiling; no GT geometry), shared by the simulator and the planners.

Camera box (z-up metres): level box shrunk by 0.25 m horizontally; heights from floor + 0.5 m to ceiling - 0.5 m.
Counts per axis = max(1, round(extent / 1.0 m)) -> the official pose formula of macarons Camera.__init__:
  position_i = x_min + (0.5 + i) * (x_max - x_min) / n        (runtime frame, axes l = x, w = y (up), h = z)
  elevation  = -90 + 180 * (1 + i_theta) / (n_theta + 1),  n_theta = 5  -> -60, -30, 0, 30, 60 degrees
  azimuth    = 360 * i_azim / n_azim,                       n_azim = 8   -> 45 degree bins
Neighbours (official get_neighboring_poses): +-1 along ONE of l, w, h OR +-1 elevation / azimuth bin; indices clamped
(azimuth wraps); duplicates removed. Environment legality (official layer-0 rule, line_segment_mesh_intersection): a
move is blocked if the GT mesh is hit on the segment between the two camera centres; rotations are always legal; the
current pose itself is not an action. Start: the T0 start panorama centre snapped to the nearest grid position whose
segment from the panorama centre is not blocked; elevation 0 (index 2), azimuth 0.
"""
import numpy as np

S_RT, STEP_M, SHRINK_M, Z_MARGIN_M = 7.0, 1.0, 0.25, 0.5
N_THETA, N_AZIM = 5, 8


def grid_def(card):
    lo, hi = np.asarray(card["box_m"][0], np.float64), np.asarray(card["box_m"][1], np.float64)
    lo_m = np.array([lo[0] + SHRINK_M, lo[1] + SHRINK_M, card["floor_m"] + Z_MARGIN_M])
    hi_m = np.array([hi[0] - SHRINK_M, hi[1] - SHRINK_M, card["ceiling_m"] - Z_MARGIN_M])
    ext = hi_m - lo_m
    n_l, n_w, n_h = (max(1, int(round(ext[0] / STEP_M))), max(1, int(round(ext[2] / STEP_M))), max(1, int(round(ext[1] / STEP_M))))
    # runtime frame: rt = 7 (x, z, -y)
    x_min = S_RT * np.array([lo_m[0], lo_m[2], -hi_m[1]])
    x_max = S_RT * np.array([hi_m[0], hi_m[2], -lo_m[1]])
    return {"x_min": x_min, "x_max": x_max, "n": (n_l, n_w, n_h), "n_theta": N_THETA, "n_azim": N_AZIM,
            "step_m": (ext[0] / n_l, ext[2] / n_w, ext[1] / n_h)}


def position(g, idx):
    n = np.asarray(g["n"], np.float64)
    return g["x_min"] + (0.5 + np.asarray(idx[:3], np.float64)) * (g["x_max"] - g["x_min"]) / n


def angles(g, idx):
    return -90.0 + 180.0 * (1 + idx[3]) / (g["n_theta"] + 1), 360.0 * idx[4] / g["n_azim"]


def neighbours(g, idx, keep_self=False):
    """Official get_neighboring_poses (unique, lexicographic). The pose itself appears there only when a shift is
    clamped at the grid border; keep_self=True keeps it (MAGICIAN beam layers > 0), False drops it (environment)."""
    out = set()
    for ax in range(5):
        for d in (-1, 1):
            q = list(idx)
            q[ax] += d
            for a in range(4):
                q[a] = min(max(q[a], 0), (list(g["n"]) + [g["n_theta"]])[a] - 1)
            q[4] %= g["n_azim"]
            out.add(tuple(int(v) for v in q))
    if not keep_self:
        out.discard(tuple(int(v) for v in idx))
    return sorted(out)


def settings_camera(g):
    """The camera block of a MAGICIAN settings dict (official unit = runtime / 10)."""
    return {"x_min": (g["x_min"] / 10.0).tolist(), "x_max": (g["x_max"] / 10.0).tolist(), "pose_l": g["n"][0], "pose_w": g["n"][1],
            "pose_h": g["n"][2], "pose_n_theta": g["n_theta"], "pose_n_azim": g["n_azim"], "start_positions": [[0, 0, 0, 2, 0]],
            "contrast_factor": 1.0}
