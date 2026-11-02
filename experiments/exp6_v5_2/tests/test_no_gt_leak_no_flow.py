"""G0: no GT leak, no flow, no depth image, no DistP1. V5.2 has no flow U-Net (no 'unet.' tensor, the V5 parent's flow
weights are not loaded); the scoring API takes no GT argument and never calls the flow sampler, the depth decoder or the
DistP1 functions (counted); scoring does not modify the planner's P1 tensors (struct / band_dil / domain hashes); the
warm-start copies the V5 encoder / decoder exactly and leaves the parent file untouched; the planner patch raises on any
guarded call while a V5.2 core is active; training reads only Train-6 shards and labels."""
import hashlib
import inspect
import json
import os
import sys

import torch

from _common import REPO, FakeGeo, Synth, V52, SC, run, score, views_x

PARENT = os.path.join(REPO, "work_dirs", "exp6_v5", "checkpoints", "v5_final.pt")
PARENT_SHA = "5ef30ac4fd0b4a175d1ddd72d78c504c82b31b40b56da68b1d9454533918d543"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_no_flow_in_model_and_exact_warm_start():
    cfg = json.load(open(f"{REPO}/experiments/exp6_v5_2/configs/model_v52.json"))
    m = V52.V52Model(cfg)
    assert not any(k.startswith("unet.") for k in m.state_dict()) and not hasattr(m, "unet") and not hasattr(m, "velocity")
    mt = os.path.getmtime(PARENT)
    ck = torch.load(PARENT, map_location="cpu", weights_only=False)
    rep = V52.load_v5_warmstart(m, ck["ema"])
    sd = m.state_dict()
    for k, v in ck["ema"].items():
        if k.startswith(("encoder.", "dec.")):
            assert torch.equal(sd[k], v), k
    assert rep["not_loaded_from_v5"] == ["unet"]
    assert sha(PARENT) == PARENT_SHA and os.path.getmtime(PARENT) == mt


def test_scoring_api_has_no_gt_and_no_forbidden_calls():
    for fn in (SC.score_state, SC.score_view, SC.combine):
        args = inspect.signature(fn).parameters
        assert not any(x in a for a in args for x in ("tstar", "gt", "oracle", "gain", "mesh")), (fn.__name__, list(args))
    from seen2scene.futureview import geometry as G, model as M, scoring as SCO
    calls = []
    saved = (M.sample, G.decode, SCO.new_set_from_depth, SCO.new_mask_from_depth)
    M.sample = lambda *a, **k: calls.append("sample")
    G.decode = lambda *a, **k: calls.append("decode")
    SCO.new_set_from_depth = lambda *a, **k: calls.append("distp1")
    SCO.new_mask_from_depth = lambda *a, **k: calls.append("distp1_mask")
    try:
        g = FakeGeo()
        h0 = [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in (g.struct, g.caster.band_dil, g.domain)]
        score(Synth([2.8, 8.4], [0.7, 0.999], [0.9, 0.9]), g, views_x(), lam=0.1)
        h1 = [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in (g.struct, g.caster.band_dil, g.domain)]
    finally:
        M.sample, G.decode, SCO.new_set_from_depth, SCO.new_mask_from_depth = saved
    assert calls == [], calls
    assert h0 == h1, "scoring modified the P1 tensors"


def test_planner_guards_raise():
    sys.path.insert(0, f"{REPO}/experiments/exp5_future_view/task1/scripts")
    sys.path.insert(0, f"{REPO}/experiments/exp6_v5_2/scripts")
    import v52_core
    v52_core.install(0.1, {})
    from seen2scene.futureview import model as M, scoring as SCO
    v52_core.STATE["active"] = True
    for f in (lambda: M.sample(None, None, None, None), lambda: SCO.new_set_from_depth(*([None] * 8))):
        try:
            f()
            raise AssertionError("guard did not raise")
        except RuntimeError as e:
            assert "V5.2 guard" in str(e)
    assert v52_core.GUARD_COUNTS["flow_sample"] == 1 and v52_core.GUARD_COUNTS["distp1_new_set"] == 1
    v52_core.STATE["active"] = False


def test_training_reads_train6_only():
    t = open(f"{REPO}/experiments/exp6_v5_2/scripts/train_v5_2.py").read()
    assert 'VD.load_split("train6_minus_Sc")' in t and "mp3d" not in t.lower().replace("mp3d_training_samples", "").replace("mp3d 0", "")
    c = json.load(open(f"{REPO}/experiments/exp6_v5_2/configs/train_v52.json"))
    assert c["train_set"] == "train6_minus_Sc" and c["mp3d_training_samples"] == 0
    assert "flow" not in c["loss"].lower().replace("no flow", "") and set(c["loss"].split("=")[1].replace("*", " ").split()) <= {"L_surface", "+", "beta", "L_firsthit", "(nothing", "else)"}


if __name__ == "__main__":
    sys.exit(run([test_no_flow_in_model_and_exact_warm_start, test_scoring_api_has_no_gt_and_no_forbidden_calls, test_planner_guards_raise, test_training_reads_train6_only]))
