"""Table 3 -- how close is the K = 1 sample to sample 0 of the K = 4 set? (CPU, read-only)
At t = 0 the state is identical for every configuration of the same (scene, start): same first observation, same latent, same
candidates. So the per-candidate one-step values of the K = 1 run (h_value_k[:, 0], noise map j) can be compared with column j of the
K = 4 run (V3-T0-K4D1 or V3-T0-D1). Same noise seed; not bitwise equal because the K = 1 forward batches one sample per view and the
K = 4 forward four (bf16 autocast). A state counts as paired when the distance to sample j is the minimum over the 4 samples (ties
allowed: at some states every candidate has value 0 in several samples, so several samples are at distance 0); the noise offset
recorded in the planner handshake is checked too. Output: fig/table3/k1_noise_pairing.csv (one row per configuration and (scene, start))."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cvpr_lib as L  # noqa: E402


def t0_values(rec):
    p = os.path.join(L.REPO, rec["episode_file"].replace("/episodes/ep_", "/planner_logs/pl_").replace(".json", ".steps.jsonl"))
    with open(p) as f:
        st = json.loads(f.readline())
    return np.asarray(st["all_scores"]["h_value_k"], float), st["choice"]["pick"]


rows = []
K4 = L.load("V3-T0-D1")
for m in ["V3-T0-K1D1", "V3-T0-K1D2"] + [f"V3-T0-K1D{d}-N{j}" for d in (1, 2) for j in (1, 2, 3)]:
    j = L.METHODS[m].get("noise", 0)
    for k, r in sorted(L.load(m).items()):
        if k not in K4:
            continue
        a, pa = t0_values(K4[k])
        b, pb = t0_values(r)
        if a.shape[0] != b.shape[0]:
            continue
        ref = a[:, j]
        rel = np.abs(b[:, 0] - ref) / np.maximum(np.abs(ref), 1e-9)
        spread = a.std(1) / np.maximum(a.mean(1), 1e-9)
        dist = np.asarray([np.max(np.abs(b[:, 0] - a[:, q]) / np.maximum(np.abs(a[:, q]), 1e-9)) for q in range(4)])
        closest = int(np.argmin(dist))
        tie = int(np.sum(dist <= dist.min() + 1e-12) > 1)
        ep = json.load(open(os.path.join(L.REPO, r["episode_file"])))
        off = ep["planner_hello"][0]["planner"]["kd"]["noise_offset"]
        rows.append([m, j, k[0], k[1], len(ref), repr(float(np.median(rel))), repr(float(rel.max())), repr(float(np.median(spread))), closest, int(dist[j] <= dist.min() + 1e-12), tie,
                     int(off == j), int(np.argmax(b[:, 0]) == np.argmax(ref)), int(bool((b[:, 0] == ref).all()))])
L.write_csv(os.path.join(L.FIG, "table3", "k1_noise_pairing.csv"), ["method_id", "noise_map", "scene_id", "start_id", "n_candidates_t0", "median_rel_diff_to_K4_sample_j", "max_rel_diff_to_K4_sample_j",
                                                                  "median_rel_spread_of_the_4_K4_samples", "closest_K4_sample(lowest index on ties)", "closest_is_j(ties allowed)", "tie_between_K4_samples",
                                                                  "handshake_noise_offset_is_j", "same_best_candidate_as_K4_sample_j", "bitwise_equal"], rows)
by = {}
for r in rows:
    by.setdefault(r[0], []).append(r)
for m, v in by.items():
    print(f"{m}: n {len(v)} | median rel diff to K4 sample {v[0][1]}: {np.median([float(x[5]) for x in v]):.4f} (max over states of the max: {max(float(x[6]) for x in v):.3f}) | "
          f"median spread among the 4 K4 samples {np.median([float(x[7]) for x in v]):.3f} | closest sample is j (ties allowed) in {sum(x[9] for x in v)}/{len(v)} (ties {sum(x[10] for x in v)}) | "
          f"handshake noise offset = j in {sum(x[11] for x in v)}/{len(v)} | bitwise equal in {sum(x[13] for x in v)}/{len(v)}")
