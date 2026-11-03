"""Camera Pose Phase 1 -- C1 hook for the V2-family planner children (B2 / E2-D100; V4 / V5 planners use the same
v2_plan_core.score_state). install() wraps v2_plan_core.score_state: the original call runs unchanged (same model, same
K = 4 flow samples, same DistP1); then FrontierPrior features are computed from the returned views and, for lam > 0, the
per-candidate list out["distp1"] -- the list the planner's own choice rule (m_score.choose / nv_choice) consumes -- is
replaced by Score_C1. lam == 0 leaves it untouched (C0). One JSON line per call to env CP_STEPLOG (D, F, score, picks)."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task1", "scripts"))
import frontier as FR  # noqa: E402


def install(lam=None, log=None):
    import v2_plan_core as PC
    lam = float(os.environ.get("CP_LAMBDA", "0") if lam is None else lam)
    log = log or os.environ.get("CP_STEPLOG")
    orig = PC.score_state
    state = {"n": 0}

    def score_state(core, geo, h, legal_R, legal_T, eps, z0, *a, **k):
        out, tm, views = orig(core, geo, h, legal_R, legal_T, eps, z0, *a, **k)
        t0 = time.time()
        feats = FR.features(views, geo) if views else []
        D = [float(x) for x in out.get("distp1", [])]
        F = [f["F"] for f in feats]
        S = FR.combine(D, F, lam) if D else []
        tm["frontier_s"] = time.time() - t0
        if lam > 0 and S:
            out["distp1_c0"] = D
            out["distp1"] = S
        out["c1"] = {"lam": lam, "F": F}
        if log:
            am = lambda x: (max(range(len(x)), key=lambda i: (x[i], -i)) if x else None)  # noqa: E731
            with open(log, "a") as fh:
                fh.write(json.dumps({"call": state["n"], "lam": lam, "n": len(D), "D": D, "F": F, "score": S, "argmax_D": am(D), "argmax_score": am(S),
                                     "frontier_s": tm["frontier_s"], "feats": [{k_: v_ for k_, v_ in f.items() if k_ != "F"} for f in feats]}) + "\n")
        state["n"] += 1
        return out, tm, views
    PC.score_state = score_state
    return {"lam": lam, "log": log}
