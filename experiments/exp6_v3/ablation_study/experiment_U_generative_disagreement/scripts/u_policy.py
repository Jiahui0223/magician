"""Experiment U -- planner-side policy switch (seen2scene env, planner process only). Protocol U-v1 = configs/u_config.yaml.

Nothing frozen is edited. The FROZEN d_policy.py (protocol D-v1) is imported read-only and installed unchanged, exactly as the
CVPR Table 3 one-step rows do (cvpr_results/scripts/table3/kd_policy.py):
  1. d_policy.lookahead is replaced by a depth-1 function that returns the one-step score s_one unchanged and predicts NO
     second-layer view (same as kd_policy._lookahead_depth1) -> after d_policy's wrapper, out["distp1"] = C1 score
     = mean_k of the per-sample h_policy.value_cov sum (lambda 0) = the score of the archived V3-T0-K4D1 rows.
  2. An outer wrapper around that score_state (keep_hits=True: the SAME frozen V3 predictions, same R10 features, same shared
     K = 4 noise maps) computes U1 / U2 / U3 of every legal candidate from the decoded samples (u_metrics.view_metrics) and,
     for policy u1 / u2 / u3, writes that metric into out["distp1"]. The C1 score is kept as out["u_c1"] for every policy.
  m_score.choose (argmax, ties -> lowest index, all-zero -> seeded fallback), the simulator legal list, P1, noise and the
  official coverage are untouched. GT is never read (the planner's LeakGuard stays installed).
Env: U_POLICY in {c1, u1, u2, u3}; U_CONFIG = configs/u_config.yaml (passed to d_policy as D_CONFIG)."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))          # experiments/
sys.path.insert(0, os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "scripts"))
import d_policy as DP  # noqa: E402  (frozen, read-only)

sys.path.insert(0, HERE)
import u_metrics as UM  # noqa: E402

HP = DP.HP
POLICIES = ("c1", "u1", "u2", "u3")


def _lookahead_depth1(core, geo, h, eps, z0, sets, s_one, pc, orig_score, stats):
    """= kd_policy._lookahead_depth1: depth 1, lambda 0, s_one returned unchanged, no layer-2 prediction."""
    assert int(pc["depth"]) == 1 and float(pc["lambda_risk"]) == 0.0, pc
    rec = HP.STATE["rec"]
    legal = [tuple(int(x) for x in r) for r in np.asarray(rec["legal_idx"]).reshape(-1, 5)]
    assert len(legal) == len(s_one), (len(legal), len(s_one))
    score = np.asarray(s_one, float).copy()
    return score, [{"c1": list(c1), "nodes": 0, "best_path": None, "q": None, "s_one": float(s_one[i])} for i, c1 in enumerate(legal)]


def install(policy):
    import v2_plan_core as PC
    import v2_common as LC
    assert policy in POLICIES, policy
    os.environ["D_CONFIG"] = os.environ["U_CONFIG"]
    cfg, cfg_sha = DP.load_cfg()
    pc = cfg["policies"][policy]
    assert int(pc["depth"]) == 1 and float(pc["lambda_risk"]) == 0.0 and pc.get("beam") is None and pc["score"] == policy, pc
    for g in ("V2M", "V2O"):
        if g in LC.ARMS:
            assert int(LC.ARMS[g].get("K") or 4) == 4, LC.ARMS[g]
    DP.lookahead = _lookahead_depth1
    dinfo = DP.install(policy)                       # frozen: obs capture + value_cov sets + s_one (+ our depth-1 lookahead)
    inner = PC.score_state
    info = {"u_policy": policy, "u_config_sha256": cfg_sha, "u_config_version": cfg["version"],
            "u_policy_file_sha256": HP.sha256_file(os.path.abspath(__file__)), "u_metrics_file_sha256": HP.sha256_file(UM.__file__),
            "d_policy_file_sha256": dinfo["d_policy_file_sha256"], "h_policy_file_sha256": dinfo["h_policy_file_sha256"],
            "score": "C1 = d_policy one-step value_cov mean (depth 1)" if policy == "c1" else f"{policy.upper()} of u_metrics.view_metrics"}
    sys.stderr.write("U_POLICY " + json.dumps(info) + "\n")
    for g in ("V2M", "V2O"):
        if g in LC.ARMS:
            LC.ARMS[g]["u_policy"] = info

    def score_state(core, geo, h, legal_R, legal_T, eps, z0, *a, **kw):
        want_hits = kw.pop("keep_hits", False)
        out, tm, views = inner(core, geo, h, legal_R, legal_T, eps, z0, *a, keep_hits=True, **kw)
        if not views:
            if not want_hits:
                out.pop("hits", None)
            return out, tm, views
        tt = time.time()
        c1 = [float(x) for x in out["distp1"]]
        mets = [UM.view_metrics(out["hits"][m]["t"], out["hits"][m]["hit_u"], v, geo.fr) for m, v in enumerate(views)]
        out["u_c1"] = c1
        out["u_metrics"] = mets
        for nm in ("u1", "u2", "u3"):
            out[f"u_{nm}"] = [float(x[nm]) for x in mets]
        if policy != "c1":
            out["distp1"] = list(out[f"u_{policy}"])
        tm = dict(tm, u_metrics_s=time.time() - tt)
        if not want_hits:
            out.pop("hits", None)
        return out, tm, views
    PC.score_state = score_state
    return info
