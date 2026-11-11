"""EXP8 -- synchronised accumulating timers (no side effects at import). Used by eff_in_prof_planner.py (indoor). Same logic as the
timers inside eff_prof_planner.py (outdoor), kept separate so the indoor child does not import the outdoor planner chain."""
import time
from collections import defaultdict

ACC = defaultdict(float)
CNT = defaultdict(int)
STEP = {"gen_calls": 0}


def _sync():
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def timed(name, fn):
    def w(*a, **k):
        t0 = _sync()
        r = fn(*a, **k)
        ACC[name] += _sync() - t0
        CNT[name] += 1
        return r
    w.__wrapped__ = fn
    return w


def install_hooks(DP):
    """Component timers on the V3 model, depth decoding, h_policy / d_policy helpers, and an inner score_state wrapper that
    d_policy.install will capture as orig_score (call 1 = first-layer candidates, later calls = second-layer predictions)."""
    import v2_plan_core as PC
    from seen2scene.v3 import model as V3
    from seen2scene.futureview import geometry as G
    HP = DP.HP
    V3.V3Model.encode = timed("vae_encode", V3.V3Model.encode)
    V3.V3Model.view_cond = timed("view_cond", V3.V3Model.view_cond)
    V3.LatentRayEncoder.forward = timed("ray_encoder", V3.LatentRayEncoder.forward)
    G.decode = timed("depth_decode", G.decode)
    HP.value_cov = timed("value_cov", HP.value_cov)
    HP.neighbours = timed("neighbours", HP.neighbours)
    HP.segment_blocked = timed("segment", HP.segment_blocked)
    DP.union_sets = timed("union", DP.union_sets)
    inner = PC.score_state

    def gen(*a, **k):
        layer = "gen_l1" if STEP["gen_calls"] == 0 else "gen_l2"
        STEP["gen_calls"] += 1
        t0 = _sync()
        out, tm, views = inner(*a, **k)
        ACC[layer] += _sync() - t0
        CNT[layer] += 1
        CNT[layer + "_views"] += len(views)
        for kk, v in tm.items():
            ACC[f"{layer}.{kk}"] += float(v)
        return out, tm, views
    PC.score_state = gen
    return PC
