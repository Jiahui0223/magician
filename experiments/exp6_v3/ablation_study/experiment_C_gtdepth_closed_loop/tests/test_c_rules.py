"""Experiment C -- CPU unit tests (seen2scene env): cq side-channel schema, choice-rule identity on gain vectors,
oracle summary, region box. Run: python -m pytest tests/test_c_rules.py -q"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import c_common as CC  # noqa: E402


def test_cq_schema_roundtrip(tmp_path):
    legal = np.arange(15, dtype=np.int64).reshape(3, 5)
    p = CC.write_cq(str(tmp_path / "cq_t3.npz"), "C1", 3, legal, tstar=np.full((3, 7296), np.inf, np.float32))
    z = CC.read_cq(p, "C1", 3, legal)
    assert set(z) == {"t", "legal_idx", "tstar"} and z["tstar"].shape == (3, 7296)
    with pytest.raises(AssertionError):
        CC.read_cq(p, "C1", 4, legal)                       # wrong step
    with pytest.raises(AssertionError):
        CC.read_cq(p, "C1", 3, legal[::-1])                 # wrong legal list
    with pytest.raises(AssertionError):
        CC.write_cq(str(tmp_path / "bad.npz"), "C1", 3, legal, tstar=np.zeros((3, 7296)), gain=np.zeros(3))   # extra key
    with pytest.raises(AssertionError):
        CC.read_cq(p, "C2", 3, legal)                       # C2 must not read a C1 file


def test_choice_rules_on_gains():
    """The frozen choice rules applied to a gain vector: argmax, ties -> lowest index, all-zero -> seeded fallback."""
    sys.path.insert(0, os.path.join(CC.EXP5, "task0", "magician_comparison", "adapters"))
    sys.path.insert(0, os.path.join(CC.REPO, "experiments", "exp3b_macaronspp_transfer"))
    import m_common as MC
    import v2nv_planner as NV
    rec = {"legal_idx": np.array([[0, 0, 0, 2, 0], [1, 0, 0, 2, 0], [0, 0, 0, 2, 1], [0, 0, 0, 1, 0]]),
           "executed_pose_idx_history": np.array([[0, 0, 0, 2, 1], [0, 0, 0, 2, 0]])}   # pose 2 already visited
    pk, fb, C = NV.nv_choice([0.1, 0.5, 0.9, 0.5], rec, "T2|C2|x", 0)
    assert C == [0, 1, 3] and pk == 1 and not fb                               # 0.9 is outside the candidate set
    pk, fb, _ = NV.nv_choice([0.0, 0.0, 0.0, 0.0], rec, "T2|C2|x", 0)
    assert fb and pk in (0, 1, 3)
    assert MC.argmax_lowest([0.2, 0.7, 0.7]) == 1
    import m_score as MS
    ch = MS.choose([0.0, 0.0], "barts", 0, 5, True)
    assert ch["fallback"] == "all_zero" and ch["pick"] == MC.fallback_choice("barts", 0, 5, 2)
    assert MS.choose([0.1, 0.3, 0.3], "barts", 0, 5, True)["pick"] == 1


def test_oracle_summary():
    g = np.array([0.0, 0.02, 0.02, 0.01])
    s = CC.oracle_summary(g, [0, 1, 2, 3], 3)
    assert s["best"] == 1 and abs(s["regret"] - 0.01) < 1e-15 and s["n_best"] == 2 and not s["top1"]
    s = CC.oracle_summary(g, [0, 3], 3)
    assert s["top1"] and s["regret"] == 0.0
    assert CC.oracle_summary(g, [0, 1], None)["regret"] is None


def test_region_box_matches_task2():
    sys.path.insert(0, os.path.join(CC.EXP5, "task2", "scripts"))
    import t2_common as C2
    card = {"box_m": [[-1.0, -2.0, 0.0], [12.0, 15.0, 3.0]], "height_m": 2.8, "s_rt": 7.0}
    center_rt = C2.zup_m_to_rt(np.array([4.0, 5.0, 1.5]))
    lo, hi = CC.region_box_xy(card, center_rt)
    ref = C2.region_box_m(card, center_rt, C2.N_WIN)
    assert np.allclose(lo, ref[0, :2]) and np.allclose(hi, ref[1, :2])


def test_sha_arrays_sensitive():
    a = np.zeros(10)
    b = a.copy(); b[3] = 1e-300
    assert CC.sha_arrays(a) != CC.sha_arrays(b) and CC.sha_arrays(a) == CC.sha_arrays(np.zeros(10))
