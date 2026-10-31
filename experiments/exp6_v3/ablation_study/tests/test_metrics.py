"""Unit tests of the ranking metrics and online choice rules in ab_data (protocol §2)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared"))
import ab_data as D  # noqa: E402


def test_rank_metrics():
    g = np.array([0.01, 0.03, 0.03 - 5e-5, 0.0])      # candidates 1 and 2 tied within tau
    s = np.array([5.0, 9.0, 7.0, 1.0])
    m = D.rank_metrics(s, g, 1)
    assert m["top1"] and m["top1_strict"] and m["regret"] == 0 and m["n_best"] == 2
    m2 = D.rank_metrics(s, g, 2)
    assert m2["top1"] and not m2["top1_strict"] and abs(m2["regret"] - 5e-5 * 100) < 1e-12
    m3 = D.rank_metrics(s, g, 0)
    assert not m3["top1"] and abs(m3["regret"] - 2.0) < 1e-12 and abs(m3["nregret"] - 2 / 3) < 1e-12
    assert m["pairwise"] == 1.0                         # all decided pairs ordered correctly
    assert abs(m["rho"] - 1.0) < 1e-12 or m["rho"] > 0.9
    r = D.random_reference(g)
    assert abs(r["top1"] - 0.5) < 1e-12 and abs(r["regret"] - (0.03 - g.mean()) * 100) < 1e-12
    assert D.rank_metrics(np.array([1.0, 1.0]), np.array([0.1, 0.2]), 0)["rho"] is None
    assert D.rank_metrics(np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0)["pairwise"] == 1.0


def test_pick_rules():
    sc = np.array([0.0, 3.0, 3.0, 1.0])
    assert D.pick_indoor(sc, [0, 1, 2, 3], "X", "tag", 5) == (1, False)       # ties -> lowest index
    assert D.pick_indoor(sc, [2, 3], "X", "tag", 5) == (2, False)             # restricted to the candidate set
    pk, fb = D.pick_indoor(np.zeros(4), [0, 1, 2, 3], "X", "tag", 5)
    assert fb and pk in (0, 1, 2, 3)
    assert D.pick_indoor(np.zeros(4), [0, 1, 2, 3], "X", "tag", 5) == (pk, True)   # deterministic fallback
    assert D.pick_outdoor(sc, "barts", 0, 5) == (1, False)
    pk2, fb2 = D.pick_outdoor(np.zeros(5), "barts", 0, 5)
    assert fb2 and 0 <= pk2 < 5


def test_hier_and_boot():
    recs = [{"scene": "A", "episode": "a1", "v": 1.0}, {"scene": "A", "episode": "a1", "v": 3.0}, {"scene": "A", "episode": "a2", "v": 10.0},
            {"scene": "B", "episode": "b1", "v": 0.0}]
    hm = D.hier_mean(recs, lambda r: r["v"])
    assert hm == {"A": 6.0, "B": 0.0}                   # episode means (2, 10) -> scene mean 6
    b = D.boot(hm.values(), n=2000)
    assert abs(b["mean"] - 3.0) < 1e-12 and b["lo"] <= 3.0 <= b["hi"]
    p = D.paired({"A": 1.0, "B": 2.0}, {"A": 2.0, "B": 1.0})
    assert p["mean"] == 0.0 and p["wins"] == 1 and p["losses"] == 1
