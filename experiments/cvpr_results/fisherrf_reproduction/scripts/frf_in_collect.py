"""FisherRF-full (frozen variant f1c4) on MP3D-150 -- aggregation (CPU only).

Episodes: full/indoor/closed_loop_f1c4/FisherRF-full/<level>_S<start>/ of the 44 EXP7 OURS manifest rows; an episode counts only if
its validation.json (frf_in_validate.py: V1-V3, EXP7 collect_mp3d150.validate, V4 same start as EXP7 OURS, f1c4 provenance, guard,
RGB/zbuf) is valid. Metrics from the C_obs curve C[0..150]: AUC@T = trapz(C[0..T]) / T (T = 20, 50, 80, 100, 150), Cov@T = C[T]
(T = 20, 50, 100, 150), Final = C[150]. Starts -> level mean -> level-equal mean over ALL-22 / HO-12 / VALSEL-10.
References, recomputed from their curves with the same formulas: EXP7 canonical records OURS / MAG-B / RANDOM
(work_dirs/cvpr_results/exp7/episodes/mp3d150/<M>/<tag>.json, valid = True) and the D3a indoor ablation
(work_dirs/cvpr_results/fisherrf_reproduction/indoor/closed_loop/FisherRF/<tag>/episode.json). The EXP7 metrics shared with
mp3d150_summary.csv are cross-checked (|d| <= 1e-12).
Paired level bootstrap of FisherRF-full - X (10,000 resamples, seed 0, the EXP7 / frf_collect procedure) + #levels FisherRF-full > X.
Behaviour / cost from the planner's own step logs (planner_fisherrf_full.steps.jsonl) and episode.json.
Outputs: results/indoor_full_f1c4_episodes.csv, results/indoor_full_f1c4_summary.json.
Usage: python frf_in_collect.py [--root DIR] [--tag indoor_full_f1c4]"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
FULL = os.path.join(R, "work_dirs", "cvpr_results", "fisherrf_reproduction", "full", "indoor", "closed_loop_f1c4", "FisherRF-full")
D3A = os.path.join(R, "work_dirs", "cvpr_results", "fisherrf_reproduction", "indoor", "closed_loop", "FisherRF")
W7 = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "episodes", "mp3d150")
E7 = os.path.join(R, "experiments", "cvpr_results", "exp7")
OUT = os.path.join(os.path.dirname(HERE), "results")
K = ["AUC@20", "AUC@50", "AUC@80", "AUC@100", "AUC@150", "Cov@20", "Cov@50", "Cov@100", "Cov@150", "Final"]
REFS = ["OURS", "MAG-B", "RANDOM", "D3a"]


def metrics(c):
    c = np.asarray(c, np.float64)
    assert len(c) == 151, len(c)
    m = {f"AUC@{T}": float(np.trapz(c[:T + 1]) / T) for T in (20, 50, 80, 100, 150)}
    m.update({f"Cov@{T}": float(c[T]) for T in (20, 50, 100, 150)})
    m["Final"] = float(c[150])
    return m


def boot(d, n=10000):
    d = np.asarray(d, float)
    b = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci_lo": float(np.percentile(b, 2.5)), "ci_hi": float(np.percentile(b, 97.5)), "wins": int((d > 0).sum()), "n": int(len(d))}


def behaviour(ep_dir, ep):
    rows = [json.loads(l) for l in open(os.path.join(ep_dir, "planner_fisherrf_full.steps.jsonl"))]
    reps = [rp for r in rows for rp in r["replans"]]
    fb = {}
    for r in rows:
        if r["fallback"]:
            fb[r["fallback"]] = fb.get(r["fallback"], 0) + 1
    rf = [rp["info"]["cand"].get("reach_filter") for rp in reps if rp["info"].get("cand", {}).get("reach_filter")]
    tot = np.array([r["timing"]["total_s"] for r in rows])
    rcost = np.array([sum(v for k2, v in rp["timing"].items() if not k2.endswith("h_train_s")) for rp in reps]) if reps else np.zeros(0)
    ex = [tuple(x) for x in ep["executed"]]
    best_len = [rp["info"]["best_len"] for rp in reps if "best_len" in rp["info"]]
    return {"steps_logged": len(rows), "distinct_positions": len({x[:3] for x in ex}), "distinct_poses": len(set(ex)), "n_replans": len(reps),
            "replan_queue_empty": sum(1 for rp in reps if rp["reason"] == "queue_empty"),
            "replan_next_not_legal": sum(1 for rp in reps if rp["reason"] == "next_not_legal"),
            "fallback_steps": int(sum(fb.values())), "fallbacks": fb,
            "no_best_path": sum(v for k2, v in fb.items() if "no_best_path" in k2),
            "first_move_not_legal": sum(v for k2, v in fb.items() if "first_move_not_legal" in k2),
            "seeded_random_turn": sum(v for k2, v in fb.items() if "seeded_random" in k2 and k2 != "no_map_seeded_random"),
            "no_map_seeded_random": fb.get("no_map_seeded_random", 0), "hover_empty_legal": fb.get("hover_empty_legal", 0),
            "no_frontier_error": sum(1 for r in rows if r.get("no_frontier_error")),
            "d13_skipped_frames": sum(1 for r in rows if r["skipped_no_valid_depth"]),
            "reach_n_before_mean": float(np.mean([x["n_before"] for x in rf])) if rf else None,
            "reach_n_after_mean": float(np.mean([x["n_after"] for x in rf])) if rf else None,
            "reach_drop_executed": int(sum(x["drop_executed"] for x in rf)), "reach_drop_obstacle": int(sum(x["drop_obstacle"] for x in rf)),
            "reach_drop_unreachable": int(sum(x["drop_unreachable"] for x in rf)),
            "best_path_len_mean": float(np.mean(best_len)) if best_len else None,
            "blocked_edges_final": rows[-1]["blocked_edges_total"], "n_gauss_final": rows[-1]["map"].get("n_gauss"),
            "rt_max_err": max(r["rt_max_err"] for r in rows),
            "step_total_mean_s": float(tot.mean()), "step_total_median_s": float(np.median(tot)), "step_total_p95_s": float(np.percentile(tot, 95)),
            "map_update_mean_s": float(np.mean([r["timing"]["map_update_s"] for r in rows])),
            "replan_cost_mean_s": float(rcost.mean()) if len(rcost) else None, "replan_cost_p95_s": float(np.percentile(rcost, 95)) if len(rcost) else None,
            "peak_alloc_gb": max(r["peak_alloc_gb"] for r in rows), "episode_wall_s": ep["wall_s"],
            "decision_s_mean": float(np.mean([s["decision_s"] for s in ep["steps"]]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=FULL)
    ap.add_argument("--tag", default="indoor_full_f1c4")
    a = ap.parse_args()
    man = [r for r in csv.DictReader(open(os.path.join(E7, "manifests", "mp3d150_manifest.csv"))) if r["method"] == "OURS"]
    split = {r["scene_id"]: r["split"] for r in man}
    eps, bad, missing = {}, {}, []
    ref = {m: {} for m in REFS}
    for r in man:
        tag = f"{r['scene_id']}_S{r['start_id']}"
        for m in ("OURS", "MAG-B", "RANDOM"):
            x = json.load(open(os.path.join(W7, m, f"{tag}.json")))
            assert x["valid"], (m, tag)
            ref[m][tag] = metrics(x["coverage"])
        ref["D3a"][tag] = metrics(json.load(open(os.path.join(D3A, tag, "episode.json")))["coverage"])
        d = os.path.join(a.root, tag)
        vp = os.path.join(d, "validation.json")
        if not os.path.exists(vp):
            missing.append(tag)
            continue
        v = json.load(open(vp))
        if not v["valid"]:
            bad[tag] = v["checks"]
            continue
        ep = json.load(open(os.path.join(d, "episode.json")))
        m = metrics(ep["coverage"])
        assert all(abs(m[k] - v["metrics"][k]) < 1e-12 for k in v["metrics"] if k in m)
        eps[tag] = dict(m, scene=r["scene_id"], start=int(r["start_id"]), split=r["split"], **behaviour(d, ep))
    # cross-check the EXP7 reference means against mp3d150_summary.csv (shared metrics)
    summ7 = {(x["split"], x["method"]): x for x in csv.DictReader(open(os.path.join(E7, "results", "mp3d150_summary.csv")))}

    def level_means(per_ep):
        lv = {}
        for tag, v in per_ep.items():
            lv.setdefault(tag.rsplit("_S", 1)[0], []).append(v)
        return {s: {k: float(np.mean([x[k] for x in xs])) for k in K} for s, xs in lv.items() if len(xs) == 2}
    LV = {"FisherRF-full": level_means(eps), **{m: level_means(ref[m]) for m in REFS}}
    res = {"episodes_valid": len(eps), "episodes_invalid": bad, "episodes_missing": missing,
           "levels_complete": len(LV["FisherRF-full"]), "variant": "f1c4 (FRF_FRONTIER_MODE=magician, FRF_REACH_FILTER=1, D12 docstring init)",
           "splits": {}, "paired": {}, "exp7_crosscheck_max_abs_diff": 0.0}
    for name, levels in (("ALL-22", sorted(split)), ("HO-12", sorted(s for s in split if split[s] == "HO")),
                         ("VALSEL-10", sorted(s for s in split if split[s] == "VALSEL"))):
        sp = {}
        for meth, lv in LV.items():
            done = [s for s in levels if s in lv]
            sp[meth] = {"levels": f"{len(done)}/{len(levels)}", **({k: float(np.mean([lv[s][k] for s in done])) for k in K} if done else {})}
            if meth in ("OURS", "MAG-B", "RANDOM") and len(done) == len(levels):
                x7 = summ7[(name, meth)]
                for k in ("AUC@20", "AUC@50", "AUC@100", "AUC@150", "Cov@20", "Cov@50", "Cov@100", "Final"):
                    res["exp7_crosscheck_max_abs_diff"] = max(res["exp7_crosscheck_max_abs_diff"], abs(sp[meth][k] - float(x7[k])))
        res["splits"][name] = sp
        both = [s for s in levels if s in LV["FisherRF-full"]]
        if len(both) == len(levels):
            res["paired"][name] = {f"FisherRF-full - {m}": {k: boot([LV["FisherRF-full"][s][k] - LV[m][s][k] for s in both]) for k in K} for m in REFS}
    assert res["exp7_crosscheck_max_abs_diff"] <= 1e-12, res["exp7_crosscheck_max_abs_diff"]
    B = list(eps.values())
    if B:
        num = lambda k: [b[k] for b in B if b.get(k) is not None]                                          # noqa: E731
        res["behaviour"] = {k: {"mean": float(np.mean(num(k))), "min": float(np.min(num(k))), "max": float(np.max(num(k))), "sum": float(np.sum(num(k)))}
                            for k in ("distinct_positions", "distinct_poses", "n_replans", "replan_queue_empty", "replan_next_not_legal", "fallback_steps",
                                      "no_best_path", "first_move_not_legal", "seeded_random_turn", "no_map_seeded_random", "hover_empty_legal",
                                      "no_frontier_error", "d13_skipped_frames", "reach_n_before_mean", "reach_n_after_mean", "reach_drop_executed",
                                      "reach_drop_obstacle", "reach_drop_unreachable", "best_path_len_mean", "blocked_edges_final", "n_gauss_final",
                                      "rt_max_err", "step_total_mean_s", "step_total_median_s", "step_total_p95_s", "map_update_mean_s",
                                      "replan_cost_mean_s", "replan_cost_p95_s", "peak_alloc_gb", "episode_wall_s", "decision_s_mean") if num(k)}
        res["behaviour"]["fallbacks_total"] = {}
        for b in B:
            for k2, v2 in b["fallbacks"].items():
                res["behaviour"]["fallbacks_total"][k2] = res["behaviour"]["fallbacks_total"].get(k2, 0) + v2
    os.makedirs(OUT, exist_ok=True)
    cols = ["tag", "scene", "start", "split"] + K + ["D3a_AUC@150", "OURS_AUC@150", "MAGB_AUC@150", "RANDOM_AUC@150", "distinct_positions", "distinct_poses",
                                                     "n_replans", "fallback_steps", "no_best_path", "seeded_random_turn", "no_map_seeded_random",
                                                     "d13_skipped_frames", "reach_n_after_mean", "best_path_len_mean", "step_total_mean_s",
                                                     "replan_cost_mean_s", "peak_alloc_gb", "episode_wall_s"]
    with open(os.path.join(OUT, f"{a.tag}_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for tag, e in sorted(eps.items()):
            e = dict(e, tag=tag, **{"D3a_AUC@150": ref["D3a"][tag]["AUC@150"], "OURS_AUC@150": ref["OURS"][tag]["AUC@150"],
                                    "MAGB_AUC@150": ref["MAG-B"][tag]["AUC@150"], "RANDOM_AUC@150": ref["RANDOM"][tag]["AUC@150"]})
            w.writerow([round(e[c], 6) if isinstance(e[c], float) else e[c] for c in cols])
    res["episodes"] = {t: {k: v for k, v in e.items()} for t, e in sorted(eps.items())}
    json.dump(res, open(os.path.join(OUT, f"{a.tag}_summary.json"), "w"), indent=1, default=float)
    print(f"valid {len(eps)} / 44, invalid {len(bad)}, missing {len(missing)}, levels complete {res['levels_complete']}, exp7 cross-check {res['exp7_crosscheck_max_abs_diff']:.1e}")
    for name, sp in res["splits"].items():
        print(name)
        for meth, v in sp.items():
            print("  %-14s %-6s " % (meth, v["levels"]) + " ".join("%s %.3f" % (k, v[k]) for k in K if k in v))
    for name, P in res["paired"].items():
        print("paired", name)
        for pair, d in P.items():
            print("  %-26s " % pair + "  ".join("%s %+.3f [%+.3f, %+.3f] %d/%d" % (k, d[k]["mean"], d[k]["ci_lo"], d[k]["ci_hi"], d[k]["wins"], d[k]["n"])
                                              for k in ("AUC@20", "AUC@100", "AUC@150", "Final")))


if __name__ == "__main__":
    main()
