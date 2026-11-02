"""G0-11 (+ part of G0-2 / G0-5): the planner is unchanged for V5. Frozen eval configs (K = 4, MP3D n_win = 5), the V5
closed loops run the unchanged task2 planners through the V4 planner children (only Core's model class differs), the
default P1 mode is b2 (D100 inputs + S10 sparse DistP1 scoring, not the Experiment-A D100 scoring), the scoring code
never imports the V5 model or a belief, and V5 training reads only Train-6 shards / labels (no MP3D, Dev-3, Test-3)."""
import hashlib
import json
import os
import re
import sys

from _common import REPO, run

T2C = os.path.join(REPO, "experiments", "exp5_future_view", "task2", "configs")
SC5 = os.path.join(REPO, "experiments", "exp6_v5", "scripts")


def test_frozen_eval_configs():
    ei = json.load(open(os.path.join(T2C, "eval_indoor_v1.json")))
    eo = json.load(open(os.path.join(T2C, "eval_outdoor_v1.json")))
    txt = json.dumps(ei) + json.dumps(eo)
    assert re.findall(r'"K": (\d+)', txt) and set(re.findall(r'"K": (\d+)', txt)) == {"4"}
    assert set(re.findall(r'"n_win": (\d+)', json.dumps(ei))) == {"5"}


def test_v5_children_use_unchanged_planners():
    o = open(os.path.join(SC5, "v5_out_planner.py")).read()
    i = open(os.path.join(SC5, "v5_in_planner.py")).read()
    assert "V4P.P24.main()" in o and "v5_core.install()" in o
    assert "V4P.P23.main()" in i and "v5_core.install()" in i
    for f in ("sim_out_v5.py", "sim_in_v5.py"):
        s = open(os.path.join(SC5, f)).read()
        assert 'os.environ.get("V4_P1_MODE", "b2")' in s, f


def test_scoring_does_not_see_the_belief():
    for f in (os.path.join(REPO, "seen2scene", "futureview", "scoring.py"), os.path.join(REPO, "experiments", "exp5_future_view", "task1", "scripts", "v2_plan_core.py"),
              os.path.join(REPO, "seen2scene", "futureview", "predictor.py")):
        s = open(f).read()
        assert "v5_model" not in s and "belief" not in s.lower() and "surface_logits" not in s, f
    core = open(os.path.join(SC5, "v5_core.py")).read()
    assert "surface_logits" not in core and "probe_logit" not in core


def test_training_data_isolation():
    t = open(os.path.join(SC5, "train_v5.py")).read()
    assert 'VD.load_split("train6_minus_Sc")' in t and "mp3d" not in t.lower().replace("mp3d_training_samples", "")
    for c in ("train_v5_outdoor.json", "train_v4_continue.json"):
        cfg = json.load(open(os.path.join(REPO, "experiments", "exp6_v5", "configs", c)))
        assert cfg["train_set"] == "train6_minus_Sc" and cfg["mp3d_training_samples"] == 0
    lab = os.path.join(REPO, "work_dirs", "exp6_v5", "labels", "v1")
    if os.path.isdir(lab):
        assert set(os.listdir(lab)) - {"logs"} <= {"bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer"}


if __name__ == "__main__":
    sys.exit(run([test_frozen_eval_configs, test_v5_children_use_unchanged_planners, test_scoring_does_not_see_the_belief, test_training_data_isolation]))
