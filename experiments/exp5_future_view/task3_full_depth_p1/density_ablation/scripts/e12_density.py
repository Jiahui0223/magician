"""EXP5 Task 3 / E1.2 -- nested BAND-density P1 (10 % c 50 % c 75 % c 100 % of the valid depth pixels of each frame).

D10  = the frozen m_p1.P1Fusion (the simulator's stored 10 % sample; B0, untouched)
Dx   = the stored 10 % sample points (bitwise, first) + the back-projection of the first floor(x * n_valid) - n10 pixels of
       a per-frame permutation of the other valid pixels, re-sorted by pixel index (x = 1: exactly p1_full.full_points)
The permutation seed is a hash of the frame content (R, T, sample_pixel_flat), so the same observation gets the same
subset in a fixed-state replay and in a closed loop. Everything else (integrate_band, integrate_free from the full
depth, structure, input_volume) is the frozen fusion. Caller sets the import paths (m_common, m_p1) first.
"""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
import p1_full as PF  # noqa: E402

MC = PF.MC
DENSITIES = (10, 50, 75, 100)


def frame_seed(rec):
    h = hashlib.sha256(b"t3e12|")
    for k in ("R", "T"):
        h.update(np.ascontiguousarray(np.asarray(rec[k], np.float64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(rec["sample_pixel_flat"], np.int64)).tobytes())
    return int(h.hexdigest()[:15], 16)


def extra_pixels(rec, x):
    """(sorted flat indices of the non-sample pixels of density x, depth [H*W], n_valid, n10)."""
    depth = np.clip(np.asarray(rec["zbuf"], np.float64), 0.5, 750.0).reshape(-1)
    valid = np.asarray(rec["mask"], bool).reshape(-1) & (depth < MC.SENSOR_RANGE)
    sf = np.asarray(rec["sample_pixel_flat"], np.int64)
    rest = valid.copy()
    rest[sf] = False
    idx = np.flatnonzero(rest)
    n_valid, n10 = int(valid.sum()), int(len(sf))
    if x >= 100:
        return idx, depth, n_valid, n10
    n_extra = max(0, (n_valid * x) // 100 - n10)
    perm = np.random.default_rng(frame_seed(rec)).permutation(idx)
    return np.sort(perm[:n_extra]), depth, n_valid, n10


def density_points(rec, x):
    """Points [N, 3] (y-up runtime, float32) of density x and the counts (n_valid, n10, n_x)."""
    idx, depth, n_valid, n10 = extra_pixels(rec, x)
    u, v = idx % MC.W, idx // MC.W
    d_y, dz, c = MC.pixel_rays(rec["R"], rec["T"], u, v)
    p = c[None] + d_y * (depth[idx] / dz)[:, None]
    pts = np.concatenate([np.asarray(rec["sample_points"], np.float32), p.astype(np.float32)])
    return pts, (n_valid, n10, int(len(pts)))


def fusion_class(x):
    """P1 class of density x (D10 = the frozen class itself, D100 = P1FusionFull)."""
    if x == 10:
        return PF._P1
    if x == 100:
        return PF.P1FusionFull

    class P1FusionDensity(PF._P1):
        density = x

        def integrate_record(self, rec):
            pts, _ = density_points(rec, self.density)
            self.integrate_band(pts, rec["cam_center"])
            self.integrate_free(rec["zbuf"], rec["mask"], rec["R"], rec["T"])
            self.n_frames += 1
    P1FusionDensity.__name__ = f"P1FusionD{x}"
    return P1FusionDensity
