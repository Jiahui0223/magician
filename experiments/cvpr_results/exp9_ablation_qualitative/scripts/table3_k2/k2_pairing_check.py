"""T3K2 -- noise-map pairing check for the K = 2 replicate with noise offset 2 (CPU, read-only). Analogue of scripts/table3/k1_pairing_check.py.
At t = 0 the state of a (scene, start) is identical for every configuration (same first observation, latent, candidates), so the per-candidate
one-step values of the K = 2, offset 2 run (h_value_k[:, 0..1]) can be compared with columns 2, 3 of the K = 4 run of the same planner
(depth 2: V3-T0-D1; depth 1: V3-T0-K4D1), and as a control with columns 0, 1 (= the archived V3-T0-K2D{d} run, maps 0, 1). Same noise seeds;
not necessarily bitwise because the K = 2 forward batches two samples per view and the K = 4 forward four (bf16 autocast). Sample i of the
replicate counts as paired with K4 sample 2 + i when that is (one of) the closest of the 4 K4 samples (ties allowed). The handshake noise
offset / maps are checked too.
Usage: python k2_pairing_check.py                       -> all canonical records of V3-T0-K2D{1,2}-N2 -> fig/table3/k2_noise_pairing.csv
       python k2_pairing_check.py --smoke DIR DEPTH     -> one smoke / regression output dir (episodes/, planner_logs/) vs liberty etc."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts"))
import cvpr_lib as L  # noqa: E402

K4 = {1: "V3-T0-K4D1", 2: "V3-T0-D1"}
K2 = {1: "V3-T0-K2D1", 2: "V3-T0-K2D2"}


def first_step(pl):
    with open(pl) as f:
        st = json.loads(f.readline())
    return np.asarray(st["all_scores"]["h_value_k"], float), st["choice"]["pick"]


def pl_of(rec):
    return os.path.join(L.REPO, rec["episode_file"].replace("/episodes/ep_", "/planner_logs/pl_").replace(".json", ".steps.jsonl"))


def compare(b, pb, a, pa, c, pc, kd):
    """b = replicate [n, 2]; a = K4 [n, 4]; c = archived K2 maps 0, 1 [n, 2]."""
    row = {}
    for i in range(2):
        dist = np.asarray([np.max(np.abs(b[:, i] - a[:, q]) / np.maximum(np.abs(a[:, q]), 1e-9)) for q in range(4)])
        row[f"s{i}_closest_K4_sample"] = int(np.argmin(dist))
        row[f"s{i}_closest_is_{2 + i}"] = int(dist[2 + i] <= dist.min() + 1e-12)
        row[f"s{i}_max_rel_diff_to_K4_{2 + i}"] = float(dist[2 + i])
        row[f"s{i}_bitwise_equal_K4_{2 + i}"] = int(bool((b[:, i] == a[:, 2 + i]).all()))
        if c is not None:
            row[f"s{i}_bitwise_equal_archived_K2_map{i}"] = int(bool((b[:, i] == c[:, i]).all()))
    if c is not None:
        row["K2map01_vs_K4_01_max_rel_diff"] = float(np.max(np.abs(c - a[:, :2]) / np.maximum(np.abs(a[:, :2]), 1e-9)))
    row["median_rel_spread_K4_samples"] = float(np.median(a.std(1) / np.maximum(a.mean(1), 1e-9)))
    row["handshake_offset_2_maps_23"] = int(kd.get("noise_offset") == 2 and kd.get("noise_maps") == [2, 3] and kd.get("K") == 2)
    return row


def main():
    rows = []
    if "--smoke" in sys.argv:
        o, d = sys.argv[sys.argv.index("--smoke") + 1], int(sys.argv[sys.argv.index("--smoke") + 2])
        ep = json.load(open(glob.glob(os.path.join(o, "episodes", "ep_*.json"))[0]))
        kd = (ep["planner_hello"][0] if isinstance(ep["planner_hello"], list) else ep["planner_hello"])["planner"]["kd"]
        key = (ep["scene"], int(ep["start"]))
        b, pb = first_step(glob.glob(os.path.join(o, "planner_logs", "*.steps.jsonl"))[0])
        a, pa = first_step(pl_of(L.load(K4[d])[key]))
        c, pc = first_step(pl_of(L.load(K2[d])[key]))
        r = {"source": o, "depth": d, "scene": key[0], "start": key[1], "n_candidates": len(b)}
        r.update(compare(b, pb, a, pa, c, pc, kd))
        print(json.dumps(r))
        json.dump(r, open(os.path.join(o, "pairing.json"), "w"), indent=1)
        return
    for d in (1, 2):
        m = f"V3-T0-K2D{d}-N2"
        A, C = L.load(K4[d]), L.load(K2[d])
        for key, r in sorted(L.load(m).items()):
            if key not in A:
                continue
            ep = json.load(open(os.path.join(L.REPO, r["episode_file"])))
            kd = (ep["planner_hello"][0] if isinstance(ep["planner_hello"], list) else ep["planner_hello"])["planner"]["kd"]
            b, pb = first_step(pl_of(r))
            a, pa = first_step(pl_of(A[key]))
            c, pc = first_step(pl_of(C[key])) if key in C else (None, None)
            if a.shape[0] != b.shape[0]:
                continue
            row = {"method_id": m, "scene_id": key[0], "start_id": key[1], "n_candidates_t0": len(b)}
            row.update(compare(b, pb, a, pa, c, pc, kd))
            rows.append(row)
    if not rows:
        print("no records yet")
        return
    hdr = list(rows[0].keys())
    L.write_csv(os.path.join(L.FIG, "table3", "k2_noise_pairing.csv"), hdr, [[r.get(h, "") for h in hdr] for r in rows])
    for m in sorted({r["method_id"] for r in rows}):
        v = [r for r in rows if r["method_id"] == m]
        print(f"{m}: n {len(v)} | sample 0 closest to K4 map 2 (ties allowed) {sum(r['s0_closest_is_2'] for r in v)}/{len(v)}, bitwise {sum(r['s0_bitwise_equal_K4_2'] for r in v)} | "
              f"sample 1 closest to K4 map 3 {sum(r['s1_closest_is_3'] for r in v)}/{len(v)}, bitwise {sum(r['s1_bitwise_equal_K4_3'] for r in v)} | "
              f"median max-rel-diff to map 2 / 3: {np.median([r['s0_max_rel_diff_to_K4_2'] for r in v]):.2e} / {np.median([r['s1_max_rel_diff_to_K4_3'] for r in v]):.2e} | "
              f"handshake offset 2, maps [2,3] {sum(r['handshake_offset_2_maps_23'] for r in v)}/{len(v)}")


if __name__ == "__main__":
    main()
