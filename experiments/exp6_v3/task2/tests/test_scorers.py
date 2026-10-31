"""Task 2 -- unit tests of the coverage-consistent scorers on synthetic geometry (seen2scene env, CPU).
Run: python -m pytest experiments/exp6_v3/task2/tests/test_scorers.py -q"""
import math
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import t2_scorers as TS  # noqa: E402

torch.manual_seed(0)


class Frame:
    k_M = 7.0
    origin_z = np.array([0.0, 0.0, 0.0])


class Caster:
    def __init__(self, band_dil):
        self.band_dil = band_dil


class Geo:
    def __init__(self, dims, domain, band_dil):
        self.dims, self.domain, self.caster = list(dims), domain, Caster(band_dil)


def make_rays(z_depth=0.5):
    """A camera at the grid centre looking along +x; 64 x 114 rays with a small fov; a flat wall at x = o_x + z_depth."""
    o = torch.tensor([0.1, 0.5, 0.5], dtype=torch.float32)
    i, j = torch.meshgrid(torch.arange(TS.GH, dtype=torch.float32), torch.arange(TS.GW, dtype=torch.float32), indexing="ij")
    u = (j - (TS.GW - 1) / 2) * TS.ALPHA
    v = (i - (TS.GH - 1) / 2) * TS.ALPHA
    r = torch.stack([torch.ones_like(u), u, v], -1).reshape(-1, 3)
    d = r / r.norm(dim=-1, keepdim=True)
    dz = 1.0 / r.norm(dim=-1)
    t = z_depth / dz                                   # wall perpendicular to +x at distance z_depth (model units)
    return o, d, dz, t


def test_s0_matches_frozen_distp1_and_generic():
    dims = (100, 100, 100)
    domain = torch.ones(dims, dtype=torch.bool)
    band = torch.zeros(dims, dtype=torch.bool)
    geo = Geo(dims, domain, band)
    o, d, dz, t = make_rays(0.5)
    hist = TS.Hist(); hist.add(np.zeros((0, 3)))
    view10 = {"o": o, "d": d, "dz": dz, "pr": {"tmax": torch.full_like(t, 10.0), "t_known": torch.full_like(t, float("inf"))}}
    view100 = {"pr": {"t_known": torch.full_like(t, float("inf"))}, "struct": torch.ones(dims, dtype=torch.int8)}
    out, diag = TS.score_view(t, view10, view100, geo, hist, eps_rt=0.5, z0=5.0, fr=Frame())
    from seen2scene.futureview import scoring as SCO
    ids, w = SCO.new_set_from_depth(t, o, d, dz, Frame.k_M, dims, domain, band, 5.0)
    assert out["S0"] == float(w.sum())
    assert diag["s0_generic_absdiff"] <= 1e-9 * max(1.0, out["S0"])
    # no history -> everything novel: S2 == S0 generic, S2h == S0; S3 identity (no dense surface)
    assert abs(out["S2"] - out["S0"]) <= 1e-9 * out["S0"] and abs(out["S2h"] - out["S0"]) <= 1e-9 * out["S0"]
    assert abs(out["S3"] - out["S0"]) <= 1e-9 * out["S0"]


def test_area_is_footprint_and_cos():
    """Flat wall facing the camera: S1a == S1c (cos = 1 for the interior) up to dz; total area ~ (alpha z)^2 per ray."""
    dims = (100, 100, 100)
    geo = Geo(dims, torch.ones(dims, dtype=torch.bool), torch.zeros(dims, dtype=torch.bool))
    o, d, dz, t = make_rays(0.5)
    hist = TS.Hist()
    view10 = {"o": o, "d": d, "dz": dz, "pr": {"tmax": torch.full_like(t, 10.0)}}
    view100 = {"pr": {"t_known": torch.full_like(t, float("inf"))}}
    out, diag = TS.score_view(t, view10, view100, geo, hist, eps_rt=0.5, z0=5.0, fr=Frame())
    assert diag["n_good_normals"] > 0.9 * (TS.GH - 2) * (TS.GW - 2)
    assert abs(diag["mean_cos"] - np.mean(dz.numpy())) < 0.02        # wall normal = +x -> |n . d| = dz
    # per-ray area = (alpha z)^2 dz / cos with cos = dz -> (alpha z)^2 -> Ahat = (z/z0)^2 ; S1a ~ S1c (dedup identical)
    assert abs(out["S1a"] - out["S1c"]) / out["S1c"] < 0.03
    # uncapped vs capped: z = 0.5 * 7 = 3.5 < z0 = 5 -> cap inactive -> S1c == S0
    assert abs(out["S1c"] - out["S0"]) <= 1e-9 * out["S0"]
    # far wall beyond z0: S1c grows as z^2, S0 saturates
    o2, d2, dz2, t2 = make_rays(1.5)                                  # z = 10.5 > z0
    out2, _ = TS.score_view(t2, {"o": o2, "d": d2, "dz": dz2, "pr": {"tmax": torch.full_like(t2, 10.0)}}, view100, geo, hist, 0.5, 5.0, Frame())
    assert out2["S1c"] > 3 * out2["S0"]


def test_oblique_wall_cos():
    """A wall tilted 60 degrees about the vertical axis: cos(theta) ~ 0.5 -> S1a ~ 2 x S1c."""
    dims = (300, 300, 100)
    geo = Geo(dims, torch.ones(dims, dtype=torch.bool), torch.zeros(dims, dtype=torch.bool))
    o, d, dz, _ = make_rays()
    n = torch.tensor([math.cos(math.radians(60)), math.sin(math.radians(60)), 0.0])
    p0 = o + torch.tensor([0.5, 0.0, 0.0])
    t = ((p0 - o) @ n) / (d @ n)                                       # ray-plane intersection
    t = torch.where(t > 0, t, torch.full_like(t, float("inf")))
    hist = TS.Hist()
    out, diag = TS.score_view(t, {"o": o, "d": d, "dz": dz, "pr": {"tmax": torch.full_like(t, 10.0)}}, {"pr": {"t_known": torch.full_like(t, float("inf"))}}, geo, hist, 0.5, 5.0, Frame())
    # per-ray |n . d| of the interior rays must match the analytic value (the fov is wide, so the mean is not 0.5)
    cos, good = TS.cos_incidence(t, o, d)
    ana = (d @ n).abs()
    fin = torch.isfinite(t)
    err = (cos - ana.clamp(min=TS.COS_MIN)).abs()[good]
    assert good.sum() > 0.7 * fin.sum() and float(err.max()) < 1e-3          # exact on a plane where the normal is valid
    assert bool((cos[fin & ~good] == 1.0).all())                              # grazing / discontinuous -> conservative fallback
    # oblique wall -> the area estimate exceeds the plain z^2 weight, bounded by the fully analytic ratio (grazing rays
    # fall back to cos = 1, so the realised ratio lies between 1 and the analytic mean of dz / cos)
    ratio_ana = float((dz / ana.clamp(min=TS.COS_MIN))[fin].mean())
    assert 1.05 < out["S1a"] / out["S1c"] <= ratio_ana + 1e-6


def test_novelty_against_history():
    """Hits within eps of past sample points get N_hard 0 / N_lin small; far ones 1."""
    dims = (100, 100, 100)
    geo = Geo(dims, torch.ones(dims, dtype=torch.bool), torch.zeros(dims, dtype=torch.bool))
    o, d, dz, t = make_rays(0.5)
    q = (o[None] + d * t[:, None]).double() * Frame.k_M                 # runtime z-up = model * k_M (origin 0)
    q_rt = (q @ TS.Y_UP_TO_Z_UP).numpy()                                # y-up runtime
    hist = TS.Hist(); hist.add(q_rt[: len(q_rt) // 2])                   # half of the wall already observed
    out, diag = TS.score_view(t, {"o": o, "d": d, "dz": dz, "pr": {"tmax": torch.full_like(t, 10.0)}}, {"pr": {"t_known": torch.full_like(t, float("inf"))}}, geo, hist, eps_rt=0.05, z0=5.0, fr=Frame())
    assert 0.4 < diag["frac_novel_hard"] < 0.6
    assert out["S2h"] < 0.6 * out["S0"] and out["S2"] < out["S0"]
    n_lin, n_hard, n_disc = TS.novelty_from_dist(torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0]), eps=1.0)
    assert n_lin.tolist() == [0.0, 0.25, 0.5, 1.0, 1.0] and n_hard.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert n_disc[0] == 0.0 and n_disc[3] == 1.0 and 0.0 < n_disc[1] < n_disc[2] < 1.0


def test_consistency_thin_front_and_behind():
    dims = (100, 100, 100)
    geo = Geo(dims, torch.ones(dims, dtype=torch.bool), torch.zeros(dims, dtype=torch.bool))
    o, d, dz, t = make_rays(0.5)
    hist = TS.Hist()
    tk = t + 0.02                                                         # dense surface 2 voxels behind the hit -> thin front
    out, diag = TS.score_view(t, {"o": o, "d": d, "dz": dz, "pr": {"tmax": torch.full_like(t, 10.0)}}, {"pr": {"t_known": tk}}, geo, hist, 0.5, 5.0, Frame())
    assert diag["n_thin_front"] == len(t) and out["S3"] == 0.0 and out["S2S3"] == 0.0
    tk = t - 0.1                                                          # hit behind the dense surface
    out, diag = TS.score_view(t, {"o": o, "d": d, "dz": dz, "pr": {"tmax": torch.full_like(t, 10.0)}}, {"pr": {"t_known": tk}}, geo, hist, 0.5, 5.0, Frame())
    assert diag["n_behind_dense"] == len(t) and out["S3"] == 0.0
    tk = t + 0.5                                                          # far behind: identity
    out, diag = TS.score_view(t, {"o": o, "d": d, "dz": dz, "pr": {"tmax": torch.full_like(t, 10.0)}}, {"pr": {"t_known": tk}}, geo, hist, 0.5, 5.0, Frame())
    assert out["S3"] == out["S0"]


def test_candidate_order_invariance_and_no_hits():
    dims = (50, 50, 50)
    geo = Geo(dims, torch.ones(dims, dtype=torch.bool), torch.zeros(dims, dtype=torch.bool))
    o, d, dz, t = make_rays(0.5)
    t_none = torch.full_like(t, float("inf"))
    out, diag = TS.score_view(t_none, {"o": o, "d": d, "dz": dz, "pr": {"tmax": torch.full_like(t, 10.0)}}, {"pr": {"t_known": t_none}}, geo, TS.Hist(), 0.5, 5.0, Frame())
    assert all(v == 0.0 for v in out.values()) and diag["n_hit"] == 0
