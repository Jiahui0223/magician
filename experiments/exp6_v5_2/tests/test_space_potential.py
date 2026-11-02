"""G0: unexplored-space potential. Counts UNKNOWN coarse cells reached by the candidate's rays (survival T), frontier
cells (next to confirmed FREE) at full weight and other UNKNOWN cells at eps; observed / out-of-domain cells score 0;
cells behind an opaque predicted wall score ~0; the term does not depend on the surface belief b (1 - b is not free)."""
import sys

from _common import FakeGeo, G, Synth, run, score, views_x


def test_frontier_vs_interior():
    v = views_x(n_rays=64, spread=0.05)
    from _common import VS
    g = FakeGeo()
    g.struct[:, int(0.44 / VS):int(0.53 / VS) + 1, :] = G.EMPTY                  # observed free slab beside the ray bundle -> frontier
    r = score(Synth([], [], []), g, v)
    g2 = FakeGeo()                                                               # same UNKNOWN cells on the rays, no observed free
    r2 = score(Synth([], [], []), g2, v)
    assert r["state"]["frontier_cells"] > 0 and r2["state"]["frontier_cells"] == 0, (r["state"], r2["state"])
    assert r["p_space"] > 2 * r2["p_space"], (r["p_space"], r2["p_space"])


def test_observed_cells_zero():
    v = views_x(n_rays=64, spread=0.05)
    g = FakeGeo()
    g.struct[:] = G.EMPTY                                                        # everything observed free
    r = score(Synth([], [], []), g, v)
    assert r["p_space"] == 0.0


def test_occluded_space_zero():
    v = views_x(n_rays=64, spread=0.05)
    g = FakeGeo(); g.mark("free", 0.0, 0.5, v["o"].tolist())
    clear = score(Synth([], [], []), g, v)["p_space"]
    blocked = score(Synth([0.6], [0.9999], [0.9]), g, v)["p_space"]
    assert blocked < 0.25 * clear, (blocked, clear)


def test_space_independent_of_belief():
    v = views_x(n_rays=64, spread=0.05)
    g = FakeGeo(); g.mark("free", 0.0, 1.0, v["o"].tolist())
    lo = score(Synth([3.0], [0.5], [0.05]), g, v)["p_space"]
    hi = score(Synth([3.0], [0.5], [0.95]), g, v)["p_space"]
    assert abs(lo - hi) < 1e-9, (lo, hi)


if __name__ == "__main__":
    sys.exit(run([test_frontier_vs_interior, test_observed_cells_zero, test_occluded_space_zero, test_space_independent_of_belief]))
