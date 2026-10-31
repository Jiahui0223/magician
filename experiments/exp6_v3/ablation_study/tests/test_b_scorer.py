"""Unit tests of the DepthU scorer (protocol §3): valid / invalid rays, HIT / NO-HIT handling, span normalisation,
permutation invariance, P2 normalisation."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_B_depth_uncertainty", "scripts"))
import b_scorer as B  # noqa: E402


def test_entropy_and_terms():
    K, R = 4, 6
    t = np.full((K, R), np.inf); hit = np.zeros((K, R), bool)
    # ray 0: all 4 HIT at the same depth -> u_hit 0, u_depth 0
    hit[:, 0] = True; t[:, 0] = 10.0
    # ray 1: 2 HIT at 10 and 14 (span 8) -> u_hit 1, u_depth = std(10,14)/8 = 2/8
    hit[:2, 1] = True; t[0, 1], t[1, 1] = 10.0, 14.0
    # ray 2: 1 HIT -> u_hit H(1/4), u_depth 0 (fewer than 2 hits)
    hit[0, 2] = True; t[0, 2] = 5.0
    # ray 3: no UNKNOWN (n_u = 0) although samples "hit" -> masked out
    hit[:, 3] = True; t[:, 3] = 3.0
    # ray 4: invalid ray
    hit[:2, 4] = True; t[:2, 4] = 1.0
    # ray 5: NO-HIT all -> 0
    valid = np.array([1, 1, 1, 1, 0, 1], bool); n_u = np.array([5, 5, 5, 0, 5, 5]); d_lo = np.zeros(R); d_hi = np.full(R, 8.0)
    r = B.depth_u_rays(t, hit, valid, n_u, d_lo, d_hi)
    assert r["u_hit"][0] == 0 and r["u_depth"][0] == 0
    assert abs(r["u_hit"][1] - 1.0) < 1e-12 and abs(r["u_depth"][1] - 0.25) < 1e-12
    assert abs(r["u_hit"][2] - B.binary_entropy(0.25)) < 1e-12 and r["u_depth"][2] == 0
    assert r["u"][3] == 0 and not r["m_unknown"][3]
    assert not r["valid"][4]
    assert r["u"][5] == 0
    s = B.s_depth_u(t, hit, valid, n_u, d_lo, d_hi)
    expect = (0 + (0.5 * 1.0 + 0.5 * 0.25) + 0.5 * B.binary_entropy(0.25) + 0 + 0) / 5   # mean over the 5 valid rays
    assert abs(s - expect) < 1e-12
    # no 'far' substitution: a NO-HIT sample never enters the depth std
    hit2 = hit.copy(); t2 = t.copy(); t2[2, 1] = 1e6          # a NO-HIT sample with a garbage depth value
    r2 = B.depth_u_rays(t2, hit2, valid, n_u, d_lo, d_hi)
    assert r2["u_depth"][1] == r["u_depth"][1]


def test_permutation_and_p2():
    rng = np.random.default_rng(0)
    n, K = 7, 4
    dk = rng.random((n, K)) * 100; du = rng.random(n)
    sc = B.scores_of_state(dk, du)
    perm = rng.permutation(n)
    sp = B.scores_of_state(dk[perm], du[perm])
    for k in sc:
        assert np.allclose(sc[k][perm], sp[k]), k
    assert sc["P2"].min() >= 0 and sc["P2"].max() <= 1 + B.LAMBDA + 1e-12
    assert np.allclose(B.scores_of_state(np.ones((3, K)), np.ones(3))["P2"], 0)        # constant -> 0
    assert np.allclose(sc["RiskAware"], dk.mean(1) - 0.5 * dk.std(1))
