"""CVPR Table 3 (K x planning depth) -- planner-side switch on top of the FROZEN D policy (seen2scene env, planner process).

Nothing in experiments/exp6_v3 is edited. d_policy.py (protocol D-v1) is imported read-only and installed unchanged:
  depth 2  D_POLICY = d1 with the frozen d_config.yaml -> the Table 1 planner of V3-T0-D1 / V3-D1-H3-NORISK, byte-for-byte.
  depth 1  D_POLICY = depth1 with kd_config.yaml (KD-v1). d_policy.score_state computes, with the frozen h_policy.value_cov,
           s_one[c1] = mean_k sum(weights of the novel cells of c1 in sample k) - lambda * std_k (lambda = 0) and then calls the
           module-level lookahead(); for depth 1 that function is replaced here by one that returns s_one unchanged WITHOUT
           predicting any second-layer view (the frozen lookahead would return the same scores for depth 1 but would first run
           the model on every layer-2 child for nothing). So the depth-1 score is the D1 one-step coverage value (mean over the
           K samples of value_cov), not the H0 DistP1.
  K        the planner reads K from the arm table (v2_common.ARMS[group]["K"], set to 4 by t24_v2_out_planner); it is set to
           KD_K here before the planner main starts. The noise maps are drawn per k with independent generators seeded by
           seed_key(scene, start, rep, t_r, k) (futureview.predictor.noise), so K = 1 uses exactly the k = 0 map of the K = 4
           set; everything else (R10, Euler 10, D100 input / S10 novelty, legal list, m_score.choose) is untouched.
  noise    KD_NOISE_OFFSET = j in {1, 2, 3} (K = 1 only, default 0): the single noise map is map k = j of the K = 4 set
           (v2_common.seed_key wrapped as k -> k + j in this planner process only). Used for the pre-declared K = 1
           noise-set replicates; j = 0 leaves seed_key untouched.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))                      # experiments/
sys.path.insert(0, os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "scripts"))
import d_policy as DP  # noqa: E402  (frozen, read-only)

HP = DP.HP


def _lookahead_depth1(core, geo, h, eps, z0, sets, s_one, pc, orig_score, stats):
    assert int(pc["depth"]) == 1 and float(pc["lambda_risk"]) == 0.0, pc
    rec = HP.STATE["rec"]
    legal = [tuple(int(x) for x in r) for r in np.asarray(rec["legal_idx"]).reshape(-1, 5)]
    assert len(legal) == len(s_one), (len(legal), len(s_one))
    score = np.asarray(s_one, float).copy()
    return score, [{"c1": list(c1), "nodes": 0, "best_path": None, "q": None, "s_one": float(s_one[i])} for i, c1 in enumerate(legal)]


def install(policy, K):
    import v2_common as LC
    K = int(K)
    assert K in (1, 4), K
    for g in ("V2M", "V2O"):
        if g in LC.ARMS:
            LC.ARMS[g]["K"] = K
    off = int(os.environ.get("KD_NOISE_OFFSET", "0"))
    assert off == 0 or (K == 1 and off in (1, 2, 3)), (K, off)
    if off:
        orig_key = LC.seed_key
        LC.seed_key = lambda scene, start, rep, t_r, k: orig_key(scene, start, rep, t_r, k + off)
    cfg, _ = DP.load_cfg()
    depth = int(cfg["policies"][policy]["depth"])
    assert depth in (1, 2) and float(cfg["policies"][policy]["lambda_risk"]) == 0.0 and cfg["policies"][policy].get("beam") is None
    if depth == 1:
        DP.lookahead = _lookahead_depth1
    info = DP.install(policy)
    kd = {"K": K, "depth": depth, "noise_offset": off, "kd_policy_file_sha256": HP.sha256_file(os.path.abspath(__file__)), "d_config": os.environ["D_CONFIG"],
          "noise": "map k = seed_key(scene, start, rep, t_r, k + noise_offset); K = 1 -> the single map k = noise_offset of the K = 4 set",
          "depth1_score": "d_policy d_one_step = mean_k value_cov (lambda 0), no layer-2 prediction" if depth == 1 else None}
    sys.stderr.write("KD_POLICY " + json.dumps(kd) + "\n")
    for g in ("V2M", "V2O"):
        if g in LC.ARMS:
            LC.ARMS[g]["kd"] = kd
    return info, kd
