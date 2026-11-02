"""G0: ordered visibility. Walls A (2.8) and B (8.4 model units, ~3 m / 9 m at the MP3D model scale) on every ray:
an opaque A gets the visible first-hit mass and B none, while B keeps its multi-surface belief and appears in P_surf;
a semi-transparent A lets B become visible; a known A (S10 band_dil) is not counted as new but still occludes; mass on
D100-confirmed FREE voxels is removed (and reported); no hit / beyond-range walls score nothing."""
import sys

from _common import FakeGeo, Synth, run, score, views_x

A, B = 2.8, 8.4


def test_opaque_A_hides_B_but_B_keeps_belief():
    v = views_x()
    g = FakeGeo(); g.mark("band", A - 0.05, A + 0.05, v["o"].tolist())        # A already known: only B could be new
    s_opq = score(Synth([A, B], [0.999, 0.999], [0.9, 0.9]), g, v)
    s_a = score(Synth([A], [0.999], [0.9]), g, v)
    assert s_opq["s_vis"] < 1e-2, s_opq["s_vis"]                                 # B is not visible behind an opaque A
    assert s_opq["p_surf"] > 10 * max(s_a["p_surf"], 1e-3), (s_opq["p_surf"], s_a["p_surf"])   # ... but kept as (discounted) potential


def test_transparent_A_makes_B_visible():
    v = views_x()
    g = FakeGeo(); g.mark("band", A - 0.05, A + 0.05, v["o"].tolist())
    s_opq = score(Synth([A, B], [0.999, 0.999], [0.9, 0.9]), g, v)["s_vis"]
    s_tr = score(Synth([A, B], [0.3, 0.999], [0.9, 0.9]), g, v)["s_vis"]
    assert s_tr > 10 * max(s_opq, 1e-3) and s_tr > 1.0, (s_opq, s_tr)


def test_visible_goes_to_first_wall():
    v = views_x()
    g = FakeGeo()
    r = score(Synth([A, B], [0.999, 0.999], [0.9, 0.9]), g, v)
    gA = FakeGeo(); gA.mark("band", A - 0.05, A + 0.05, v["o"].tolist())
    rA = score(Synth([A, B], [0.999, 0.999], [0.9, 0.9]), gA, v)
    assert r["s_vis"] > 1.0 and rA["s_vis"] < 1e-2 * r["s_vis"]                  # all visible value sits on A


def test_free_check_removes_and_reports():
    v = views_x()
    g = FakeGeo(); g.mark("free", A - 0.05, A + 0.05, v["o"].tolist())
    r = score(Synth([A], [0.999], [0.9]), g, v)
    assert r["s_vis"] < 1e-2 and r["diag"]["mass_free"] > 0.9 * len(v["d"]), r["diag"]


def test_no_hit_and_out_of_range():
    v = views_x(tcap=9.4)
    g = FakeGeo()
    r0 = score(Synth([], [], []), g, v)
    r1 = score(Synth([9.9], [0.999], [0.9]), g, v)                               # beyond tcap
    assert r0["s_vis"] == 0.0 and r0["p_surf"] == 0.0 and abs(r0["diag"]["p_nohit_mean"] - 1.0) < 1e-6
    assert r1["s_vis"] == 0.0 and abs(r1["diag"]["p_nohit_mean"] - 1.0) < 1e-6


def test_known_band_wall_still_occludes():
    """A known (D100 BAND, S10 band_dil) opaque wall: nothing new is visible behind it."""
    v = views_x()
    g = FakeGeo(); g.mark("band", A - 0.05, A + 0.05, v["o"].tolist()); g.mark("known_band", A - 0.05, A + 0.05, v["o"].tolist())
    r = score(Synth([A, B], [0.999, 0.999], [0.9, 0.9]), g, v)
    assert r["s_vis"] < 1e-2, r["s_vis"]


if __name__ == "__main__":
    sys.exit(run([test_opaque_A_hides_B_but_B_keeps_belief, test_transparent_A_makes_B_visible, test_visible_goes_to_first_wall,
                  test_free_check_removes_and_reports, test_no_hit_and_out_of_range, test_known_band_wall_still_occludes]))
