"""G0: world-voxel de-duplication and the DistP1 distance weight. Identical rays hitting the same voxels count once
(max over rays); a ray's mass split over sub-points inside one voxel sums to its interval mass; w_dist = min(1, (z / z0)^2)
with z the runtime z-depth; with an (almost) delta first hit the visible term reproduces the frozen DistP1
(scoring.new_set_from_depth) of the same depth; the number of intervals barely changes the score."""
import sys

import torch

from _common import FakeGeo, Synth, VS, run, score, views_x
from seen2scene.futureview import scoring as SCO


def test_identical_rays_count_once():
    v1, v8 = views_x(n_rays=1, identical=True), views_x(n_rays=8, identical=True)
    g = FakeGeo()
    m = Synth([2.8], [0.999], [0.9])
    s1, s8 = score(m, g, v1)["s_vis"], score(m, g, v8)["s_vis"]
    assert abs(s1 - s8) < 1e-6 and abs(s1 - 0.999) < 1e-3, (s1, s8)


def test_distance_weight():
    g = FakeGeo()
    m = Synth([2.8], [0.999], [0.9])
    v = views_x(n_rays=1, identical=True)
    far = score(m, g, v, k_M=1.0, z0=1.0)["s_vis"]                                # z = 2.8 > z0 -> weight 1
    near = score(m, g, v, k_M=1.0, z0=5.6)["s_vis"]                               # z / z0 = 0.5 -> weight 0.25
    assert abs(far - 0.999) < 1e-3 and abs(near / far - 0.25) < 0.02, (far, near)


def test_delta_first_hit_matches_distp1():
    v = views_x(n_rays=256, spread=0.3)
    g = FakeGeo()
    t_w = 0.6
    m = Synth([t_w], [1.0 - 1e-9], [0.9], n_bins=2048)
    s = score(m, g, v, k_M=1.0, z0=1.0)["s_vis"]
    dims = torch.tensor(g.dims)
    ids, wv = SCO.new_set_from_depth(torch.full((256,), t_w), v["o"], v["d"], v["dz"], 1.0, dims, g.domain, g.caster.band_dil, 1.0)
    ref = float(wv.sum())
    assert abs(s - ref) / ref < 0.05, (s, ref)


def test_bins_robust():
    v = views_x(n_rays=128, spread=0.2)
    g = FakeGeo()
    vals = [score(Synth([2.8, 8.4], [0.7, 0.999], [0.9, 0.9], n_bins=n), g, v)["s_vis"] for n in (64, 128, 256)]
    assert max(vals) / min(vals) < 1.10, vals


if __name__ == "__main__":
    sys.exit(run([test_identical_rays_count_once, test_distance_weight, test_delta_first_hit_matches_distp1, test_bins_robust]))
