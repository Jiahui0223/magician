"""EXP5 Task 3 / Experiment 1 -- FullDepth-P1: the frozen P1 fusion (experiments/exp3b_macaronspp_transfer/m_p1.py) with
the BAND layer fed by ALL valid depth pixels of each observation instead of the simulator's 10 % sample.

Minimal change: P1FusionFull.integrate_record = the frozen integrate_record except that the BAND points are the stored
10 % sample points (bitwise, as in B0) PLUS the back-projection of every other valid pixel of the same record.
  valid pixel   = mask & clamp(zbuf, 0.5, 750) < 70 -- the exact pixel set the simulator samples its 10 % from
                  (mp3d_sim.Sensor.sample / MAGICIAN compute_partial_point_cloud after apply_perfect_depth_simple)
  back-projection = m_common.pixel_rays (the camera model of integrate_free): p = c + d * z / dz
Everything else is inherited unchanged: integrate_band (same voxel traversal, truncation, signed distance, fixed-point
sums and counts), integrate_free (EMPTY carving from the full depth), input_volume / structure (BAND over EMPTY),
frames, domains. No interpolation, no point beyond a measured ray end, no future information.
patch(): replace m_p1.P1Fusion in the current process (planner children only; mode from env T3_P1_MODE=full).
"""
import os

import numpy as np

import m_common as MC
import m_p1

_P1 = m_p1.P1Fusion


def full_points(rec):
    """Points [N, 3] (y-up runtime, float32) of all valid pixels of a service record: the stored 10 % sample points first
    (bitwise), then the back-projection of the other valid pixels."""
    depth = np.clip(np.asarray(rec["zbuf"], np.float64), 0.5, 750.0).reshape(-1)
    valid = np.asarray(rec["mask"], bool).reshape(-1) & (depth < MC.SENSOR_RANGE)
    sf = np.asarray(rec["sample_pixel_flat"], np.int64)
    rest = valid.copy()
    rest[sf] = False
    idx = np.flatnonzero(rest)
    u, v = idx % MC.W, idx // MC.W
    d_y, dz, c = MC.pixel_rays(rec["R"], rec["T"], u, v)
    p = c[None] + d_y * (depth[idx] / dz)[:, None]
    return np.concatenate([np.asarray(rec["sample_points"], np.float32), p.astype(np.float32)]), int(valid.sum())


class P1FusionFull(_P1):
    """P1 with the BAND layer from all valid depth pixels (Task 3 B1)."""

    def integrate_record(self, rec):
        pts, _ = full_points(rec)
        self.integrate_band(pts, rec["cam_center"])
        self.integrate_free(rec["zbuf"], rec["mask"], rec["R"], rec["T"])
        self.n_frames += 1


class P1FusionB2(P1FusionFull):
    """Task 3 B2: the FullDepth-P1 (model inputs) plus the frozen Sparse-P1 of the same records (DistP1 novelty only)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse = _P1(*args, **kwargs)

    def integrate_record(self, rec):
        super().integrate_record(rec)
        self.sparse.integrate_record(rec)


def _patch_geometry_b2():
    """v2_plan_core.Geometry.from_fusion: when the fusion carries a sparse map (B2), replace ONLY caster.band_dil -- the
    dilated BAND that DistP1 (scoring.new_set_from_depth, via score_state) uses to decide novelty -- by the one of the
    sparse map, computed exactly as m_score.SceneCaster does (|v| < 0.999 trunc within the domain, 3x3x3 max pool).
    Inputs (coarse grid, projection channels, decoding) keep the FullDepth-P1."""
    import torch
    import torch.nn.functional as F
    import v2_plan_core as PC
    orig = PC.Geometry.from_fusion.__func__

    def from_fusion(cls, fr, dims, dom, fu, device):
        geo = orig(cls, fr, dims, dom, fu, device)
        sp = getattr(fu, "sparse", None)
        if sp is not None:
            c = geo.caster
            v = sp.input_volume().to(c.dev).to(c.dt)
            ob = (v.abs() < MC.TRUNC * 0.999) & c.domain
            c.band_dil = F.max_pool3d(ob[None, None].half(), 3, 1, 1)[0, 0] > 0
            del v, ob
            torch.cuda.empty_cache()
        return geo
    PC.Geometry.from_fusion = classmethod(from_fusion)


def patch(mode=None):
    """Install the P1 variant of this process from mode (default env T3_P1_MODE): sparse (B0, nothing changes),
    full (B1: FullDepth-P1 for inputs and scoring), b2 (B2: FullDepth-P1 inputs, Sparse-P1 DistP1 novelty)."""
    mode = mode or os.environ.get("T3_P1_MODE", "sparse")
    if mode == "full":
        m_p1.P1Fusion = P1FusionFull
    elif mode == "b2":
        m_p1.P1Fusion = P1FusionB2
        _patch_geometry_b2()
    elif mode != "sparse":
        raise ValueError(f"T3_P1_MODE must be sparse, full or b2, got {mode}")
    return mode
