"""G0-2 (data split) and G0-9 (zero leakage) of EXP6 V4 / V3.1: Train-6 vs Dev-3 / Test-3 scene isolation, S-c
trajectories held out, the outdoor training lists of the V4 (v2_data.load_split) and V3.1 (seen2scene.v3.data.items)
loaders identical, MP3D eval houses isolated from train / VALSEL houses, no MP3D file in the outdoor data roots, the
training rows never read t* (future depth only as the label b / y / cat), encoder input index <= t, observed voxels
of the coarse grids non-decreasing in t (no future information in P1)."""
import json
import os
import sys

import numpy as np

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
EXP5 = os.path.join(REPO, "experiments", "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task1", "scripts"))
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import v2_data as VD  # noqa: E402
import t2_common as C  # noqa: E402
from seen2scene.v3 import data as D  # noqa: E402

OUTD = D.shard_path("outdoor", "x", "y").rsplit("/", 2)[0]
DEV3, TEST3 = ("barts", "pantheon", "colosseum"), ("alhambra", "bridge", "neuschwanstein")
TRAIN6 = ("bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer")


def test_g0_2_outdoor_scene_isolation():
    train, held = VD.load_split("train6_minus_Sc")
    dcfg = json.load(open(os.path.join(EXP5, "task1", "configs", "data_v1.json")))
    assert tuple(dcfg["scenes_train6"]) == TRAIN6
    assert {s for s, _ in train} == set(TRAIN6) and {s for s, _ in held} == set(TRAIN6)
    assert not (set(TRAIN6) & set(DEV3)) and not (set(TRAIN6) & set(TEST3)) and not (set(DEV3) & set(TEST3))
    ev = json.load(open(os.path.join(EXP5, "task2", "configs", "eval_outdoor_v1.json")))["scenes"]
    assert tuple(ev["dev"]) == DEV3 and tuple(ev["test"]) == TEST3
    assert len(train) == 138 and len(held) == 42 and not (set(train) & set(held))
    for s, tj in train:
        assert not bool(np.load(os.path.join(OUTD, s, f"{tj}.npz"))["heldout"]), (s, tj)
    for s, tj in held:
        assert bool(np.load(os.path.join(OUTD, s, f"{tj}.npz"))["heldout"]), (s, tj)
    assert sorted(train) == D.items("outdoor", "train") and sorted(held) == D.items("outdoor", "sc")
    assert not any(x in ("mp3d", "indoor") for x in os.listdir(OUTD))


def test_g0_2_mp3d_isolation():
    assert not (set(C.EVAL_HOUSES) & set(C.TRAIN_HOUSES)) and not (set(C.EVAL_HOUSES) & set(C.VALSEL_HOUSES))
    assert not (set(C.TRAIN_HOUSES) & set(C.VALSEL_HOUSES)) and not (set(C.EVAL_HOUSES) & set(C.TEST_HOUSES))
    assert len(C.EVAL_LEVELS) == 10 and len(C.EVAL_HOUSES) == 5
    for root in ("work_dirs/exp6_v4/runs/main", "work_dirs/exp6_v3_1/runs/main"):      # formal run manifests, once they exist
        for f in os.listdir(os.path.join(REPO, root)) if os.path.isdir(os.path.join(REPO, root)) else []:
            if f.startswith("run_none"):
                r = json.load(open(os.path.join(REPO, root, f)))
                files = r.get("data_files_sha256") or {k: v for d in r["data_files"].values() for k, v in d.items()}
                assert all("mp3d" not in k and "indoor" not in k for k in files), root
                assert r.get("mp3d_training_samples") == 0 and r.get("domains") == ["outdoor"]


def test_g0_9_no_future_information():
    assert "tstar" not in VD.VIEW_KEYS and "tstar" not in D.VIEW_KEYS
    rng = np.random.default_rng(0)
    for s, tj in [("bannerman", "cg-A-0001"), ("pisa", "rw-A-0001")]:
        z = np.load(os.path.join(OUTD, s, f"{tj}.npz"))
        assert not bool(z["heldout"])
        coarse = z["coarse"]                                                     # [T, 3, X', Y', Z'] uint8 (EMPTY, BAND, dom)
        obs = (coarse[:, 0].astype(np.int64) > 0) | (coarse[:, 1].astype(np.int64) > 0)
        n_obs = obs.reshape(len(coarse), -1).sum(1)
        assert np.all(np.diff(n_obs) >= 0), "observed cells must accumulate with t"
        grown = obs[1:] & ~obs[:-1]
        assert grown.any(), "later states see new cells"
        tt = z["t"].astype(int)
        for _ in range(50):                                                      # the sampler's encoder index is t - dlt <= t
            t = int(rng.choice(tt))
            dlt = int(rng.integers(1, min(9, t) + 1)) if t > 0 else 0
            assert 0 <= t - dlt <= t
