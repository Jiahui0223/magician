"""EXP8 -- component-profiling planner child (seen2scene env). The frozen Table 3 chain of kd_out_planner.py (t24_v2_out_planner under
p1_full.patch() b2, planner_v3.install(), kd_policy.install(D_POLICY, KD_K)) with synchronised timers wrapped around module
functions (no frozen file edited; decisions unchanged -- picks are compared with the recorded ones by eff_replay.py):
  vae_encode        seen2scene.v3.model.V3Model.encode               (GeoAnchor 3D VAE encoder; runs at refreshes only, t % 10 == 0)
  ray_encoder       seen2scene.v3.model.LatentRayEncoder.forward      (latent sampling along rays + 1D ray encoder)
  view_cond         V3Model.view_cond                                 (ray conditioning incl. ray_encoder)
  depth_decode      seen2scene.futureview.geometry.decode             (UNKNOWN-arc depth decoding)
  gen_l1 / gen_l2   v2_plan_core.score_state as captured by d_policy  (1st call = first-layer candidates, later calls = the
                    second-layer predictions of the D1 lookahead); each call's own stage timing (projection / cond / sampling /
                    decode_score) is accumulated per layer
  value_cov         h_policy.value_cov                                (novel-surface test + weighted coverage per view / sample)
  neighbours / segment / union   h_policy.neighbours, h_policy.segment_blocked, d_policy.union_sets (second-step expansion,
                    path validity, joint coverage union)
The accumulators are written into the timing dict of the step (key "eff"), which the planner logs in planner_logs/*.steps.jsonl.
Synchronising around many small calls adds overhead; the complete decision latency is therefore measured separately with the
unmodified kd_out_planner.py (eff_replay.py --method OURS-K4D2)."""
import os
import sys
import time
from collections import defaultdict

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
X = os.path.join(R, "experiments")
EXP5 = os.path.join(X, "exp5_future_view")
T3 = os.path.join(X, "cvpr_results", "scripts", "table3")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_v2_out_planner as P24  # noqa: E402

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
sys.path.insert(0, T3)
import p1_full as PF  # noqa: E402
import planner_v3 as PV3  # noqa: E402
import kd_policy as KD  # noqa: E402

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


def install_hooks():
    import v2_plan_core as PC
    from seen2scene.v3 import model as V3
    from seen2scene.futureview import geometry as G
    HP = KD.DP.HP
    V3.V3Model.encode = timed("vae_encode", V3.V3Model.encode)
    V3.V3Model.view_cond = timed("view_cond", V3.V3Model.view_cond)
    V3.LatentRayEncoder.forward = timed("ray_encoder", V3.LatentRayEncoder.forward)
    G.decode = timed("depth_decode", G.decode)
    HP.value_cov = timed("value_cov", HP.value_cov)
    HP.neighbours = timed("neighbours", HP.neighbours)
    HP.segment_blocked = timed("segment", HP.segment_blocked)
    KD.DP.union_sets = timed("union", KD.DP.union_sets)
    inner = PC.score_state

    def gen(*a, **k):                                   # captured by d_policy.install as orig_score
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


if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    PC = install_hooks()
    KD.install(os.environ["D_POLICY"], os.environ["KD_K"])
    policy_score = PC.score_state

    def outer(*a, **k):
        t0 = _sync()
        out, tm, views = policy_score(*a, **k)
        total = _sync() - t0
        tm = dict(tm, eff=dict({f"{n}_s": v for n, v in ACC.items()}, **{f"n_{n}": c for n, c in CNT.items()}, outer_score_state_s=total))
        ACC.clear(); CNT.clear(); STEP["gen_calls"] = 0
        return out, tm, views
    PC.score_state = outer
    P24.main()
