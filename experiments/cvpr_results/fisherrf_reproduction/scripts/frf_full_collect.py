"""FisherRF-full (Stage F1b) pilot collection: per-episode official-coverage metrics (closed loop curve == official replay,
fisherrf_out_cov.py), behaviour and timing from the planner's own step logs, per-scene means vs the MAGICIAN paper's FisherRF
row (Table 6 / 7 per-scene AUC / Final) and our D3a ablation / Table 1 MAG-B episodes of the same scenes and starts.
Usage: python frf_full_collect.py [--scenes dunnottar redeemer barts]   (writes results/pilot_full_*.{csv,json})"""
import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
FULL = os.path.join(REPO, "work_dirs", "cvpr_results", "fisherrf_reproduction", "full", "outdoor", "closed_loop_v3")
OUT = os.path.join(os.path.dirname(HERE), "results")
D3A_CSV = os.path.join(OUT, "outdoor_episodes.csv")
MAGB = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "g3", "metrics", "outdoor", "MAG-B", "MAG-B")
PAPER = {"dunnottar": (0.500, 0.809), "colosseum": (0.551, 0.757), "bannerman": (0.595, 0.801), "pantheon": (0.270, 0.444),
         "redeemer": (0.727, 0.876), "liberty": (0.553, 0.819), "pisa": (0.486, 0.776), "fushimi": (0.565, 0.814),
         "alhambra": (0.462, 0.615), "neuschwanstein": (0.375, 0.582), "eiffel": (0.616, 0.827), "bridge": (0.637, 0.877),
         "barts": (0.673, 0.878), "sestino_museum": (0.571, 0.842)}


def auc(c, T):
    return float(np.trapz(np.asarray(c[:T + 1], np.float64), dx=1.0) / T)


def episode(s, k):
    root = os.path.join(FULL, f"{s}_s{k}")
    ep_p = os.path.join(root, "episodes", f"ep_FisherRF_{s}_s{k}_r0.json")
    met_p = os.path.join(root, "metrics", "FisherRF", f"FisherRF_{s}_s{k}_r0.json")
    if not (os.path.exists(ep_p) and os.path.exists(met_p)):
        return None
    ep, met = json.load(open(ep_p)), json.load(open(met_p))
    c = met["cov_official"]
    assert met["official_equal_stored"] and c == ep["coverage"]
    rows = [json.loads(l) for l in open(glob.glob(os.path.join(root, "planner_logs", "*.steps.jsonl"))[0])]
    reps = [rp for r in rows for rp in r["replans"]]
    best_len = [rp["info"]["best_len"] for rp in reps if "best_len" in rp["info"]]
    fb = {}
    for r in rows:
        if r["fallback"]:
            fb[r["fallback"]] = fb.get(r["fallback"], 0) + 1
    tot = np.array([r["timing"]["total_s"] for r in rows])
    mp = np.array([r["timing"]["map_update_s"] for r in rows])
    rcost = np.array([sum(v for k2, v in rp["timing"].items() if not k2.endswith("h_train_s")) for rp in reps]) if reps else np.zeros(0)
    keys = [tuple(json.loads(x)) for x in ep["executed_keys"]]
    return {"scene": s, "start": k, "AUC@20": auc(c, 20), "AUC@100": auc(c, 100), "Cov@20": c[20], "Cov@50": c[50], "Final": c[100],
            "official_equal_stored": met["official_equal_stored"], "distinct_positions": len({x[:3] for x in keys}),
            "distinct_poses": len(set(keys)), "n_replans": len(reps),
            "replan_reasons": {x: sum(1 for rp in reps if rp["reason"] == x) for x in ("queue_empty", "next_not_legal")},
            "frontier_found": sum(1 for rp in reps if rp["info"]["frontier"].get("center_rt") is not None),
            "fallbacks": fb, "no_frontier_error": sum(1 for r in rows if r.get("no_frontier_error")),
            "illegal_first_move": sum(v for k2, v in fb.items() if "first_move_not_legal" in k2),
            "no_best_path": sum(v for k2, v in fb.items() if "no_best_path" in k2),
            "d13_skipped_frames": sum(1 for r in rows if r["skipped_no_valid_depth"]),
            "best_path_len_mean": float(np.mean(best_len)) if best_len else None, "blocked_edges_final": rows[-1]["blocked_edges_total"],
            "n_gauss_final": rows[-1]["map"].get("n_gauss"), "step_total_mean_s": float(tot.mean()), "step_total_median_s": float(np.median(tot)),
            "step_total_p95_s": float(np.percentile(tot, 95)), "map_update_mean_s": float(mp.mean()),
            "replan_cost_mean_s": float(rcost.mean()) if len(rcost) else None, "replan_cost_p95_s": float(np.percentile(rcost, 95)) if len(rcost) else None,
            "peak_alloc_gb": max(r["peak_alloc_gb"] for r in rows), "episode_wall_s": ep["seconds_this_process"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["dunnottar", "redeemer", "barts"])
    ap.add_argument("--root", default=None, help="closed-loop root (default full/outdoor/closed_loop_v3)")
    ap.add_argument("--tag", default="pilot_full")
    a = ap.parse_args()
    global FULL
    FULL = a.root or FULL
    eps = [e for s in a.scenes for k in range(5) for e in [episode(s, k)] if e is not None]
    d3a = {(r["scene"], int(r["start"])): r for r in csv.DictReader(open(D3A_CSV))}
    magb = {}
    for s in a.scenes:
        for k in range(5):
            p = os.path.join(MAGB, f"MAG-B_{s}_s{k}_r0.json")
            if os.path.exists(p):
                m = json.load(open(p))
                magb[(s, k)] = (auc(m["cov_official"], 100), m["cov_official"][100])
    os.makedirs(OUT, exist_ok=True)
    cols = ["scene", "start", "AUC@20", "AUC@100", "Cov@20", "Cov@50", "Final", "official_equal_stored", "distinct_positions", "distinct_poses", "n_replans", "illegal_first_move", "no_best_path",
            "frontier_found", "no_frontier_error", "d13_skipped_frames", "best_path_len_mean", "step_total_mean_s", "replan_cost_mean_s"]
    with open(os.path.join(OUT, f"{a.tag}_episodes.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(cols + ["D3a_AUC@100", "D3a_Final", "MAGB_AUC@100", "MAGB_Final"])
        for e in eps:
            dk = d3a.get((e["scene"], e["start"]), {})
            mk = magb.get((e["scene"], e["start"]), (None, None))
            w.writerow([e[c] if not isinstance(e[c], float) else round(e[c], 5) for c in cols] +
                       [dk.get("AUC@100"), dk.get("Final"), None if mk[0] is None else round(mk[0], 5), None if mk[1] is None else round(mk[1], 5)])
    per = {}
    for s in a.scenes:
        E = [e for e in eps if e["scene"] == s]
        if not E:
            continue
        D = [d3a[(s, k)] for k in range(5) if (s, k) in d3a]
        M = [magb[(s, k)] for k in range(5) if (s, k) in magb]
        per[s] = {"n": len(E), "AUC@100": float(np.mean([e["AUC@100"] for e in E])), "Final": float(np.mean([e["Final"] for e in E])),
                  "AUC@20": float(np.mean([e["AUC@20"] for e in E])), "paper_AUC": PAPER[s][0], "paper_Final": PAPER[s][1],
                  "D3a_AUC@100": float(np.mean([float(d["AUC@100"]) for d in D])) if D else None,
                  "D3a_Final": float(np.mean([float(d["Final"]) for d in D])) if D else None,
                  "MAGB_AUC@100": float(np.mean([m[0] for m in M])) if M else None, "MAGB_Final": float(np.mean([m[1] for m in M])) if M else None,
                  "distinct_positions": float(np.mean([e["distinct_positions"] for e in E])), "n_replans": float(np.mean([e["n_replans"] for e in E])),
                  "best_path_len_mean": float(np.mean([e["best_path_len_mean"] for e in E if e["best_path_len_mean"] is not None])),
                  "illegal_first_move": float(np.mean([e["illegal_first_move"] for e in E])), "AUC@20_mean": float(np.mean([e["AUC@20"] for e in E])),
                  "step_total_mean_s": float(np.mean([e["step_total_mean_s"] for e in E]))}
    summ = {"episodes": eps, "per_scene": per}
    json.dump(summ, open(os.path.join(OUT, f"{a.tag}_summary.json"), "w"), indent=1, default=float)
    print("scene        n  AUC@100 (paper)    Final (paper)    | D3a AUC/Final | MAG-B AUC/Final | pos  replans pathlen step_s")
    for s, p in per.items():
        f = lambda x: "  -  " if x is None else "%.3f" % x  # noqa: E731
        print("%-12s %d  %.3f (%.3f)    %.3f (%.3f)    | %s / %s   | %s / %s     | %4.1f %6.1f %6.1f %6.2f" % (
            s, p["n"], p["AUC@100"], p["paper_AUC"], p["Final"], p["paper_Final"], f(p["D3a_AUC@100"]), f(p["D3a_Final"]),
            f(p["MAGB_AUC@100"]), f(p["MAGB_Final"]), p["distinct_positions"], p["n_replans"], p["best_path_len_mean"], p["step_total_mean_s"]))


if __name__ == "__main__":
    main()
