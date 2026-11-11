"""EXP8 -- aggregate the replay measurements (CPU, seen2scene env). Reads work_dirs/cvpr_results/efficiency/runs/<METHOD>/<src>_<scene>_s<k>/
(driver steps.jsonl / summary.json; for RAYSCOUT also planner_logs/pl_*.steps.jsonl with the planner's synchronised stage timers) and
writes experiments/cvpr_results/efficiency/{tables,raw}/.

Timing definitions (identical for every RAYSCOUT configuration):
  decision      = planner decision_s - p1_s : from 'map state available' to the selected action (planning geometry, GeoAnchor refresh
                  at t % 10 == 0 -- counted in the step where it happens, so a window of whole decades amortises it exactly --,
                  first-layer generation, CDGR / one-step scoring, choice)
  cycle         = planner decision_s       : decision + online map update (record load + P1 fusion)
  roundtrip     = driver wall clock of the JSON request -> answer (cycle + inter-process I/O)
MAGICIAN beam:
  decision      = occ_s + novelty_s + beam_s (official plan(): occupancy field / imagined Gaussians, novelty from the history, beam
                  search); cycle = planner decision_s (observe = map update + plan); roundtrip as above
Main window: t = 10..99 (90 decisions per trajectory, exactly 9 GeoAnchor refreshes, warm process). Phases: early t = 1..20, middle
21..50, late 51..99. Statistics over all decisions of the window: mean, median, p95, std, n; plus per-trajectory medians."""
import csv
import glob
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
RUNS = os.path.join(R, "work_dirs", "cvpr_results", "efficiency", "runs")
E = os.path.join(R, "experiments", "cvpr_results", "efficiency")
TAB, RAW = os.path.join(E, "tables"), os.path.join(E, "raw")
PHASES = {"main_t10_99": (10, 99), "early_t1_20": (1, 20), "middle_t21_50": (21, 50), "late_t51_99": (51, 99)}
LABEL = {"OURS-K4D2": "RAYSCOUT (K=4, CDGR/D1)", "OURS-K4D1": "K=4, one-step", "OURS-K1D2": "K=1, CDGR/D1", "OURS-K1D1": "K=1, one-step",
         "OURS-K4D2-PROF": "RAYSCOUT profiled (component timers)", "MAG-B": "MAGICIAN beam (10x10)"}


def stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max())}


def load_run(d):
    m = d.split(os.sep)[-2]
    summ = json.load(open(os.path.join(d, "summary.json")))
    drv = {r["t"]: r for r in (json.loads(l) for l in open(os.path.join(d, "steps.jsonl")))}
    rows = []
    if m == "MAG-B":
        for t, r in sorted(drv.items()):
            g = r["diag"] or {}
            plan = g.get("occ_s", 0) + g.get("novelty_s", 0) + g.get("beam_s", 0)
            rows.append({"t": t, "decision": plan, "cycle": r["decision_s"], "roundtrip": r["roundtrip_s"], "map_update": r["decision_s"] - plan,
                         "occupancy": g.get("occ_s"), "novelty": g.get("novelty_s"), "beam": g.get("beam_s"), "n_legal": r["n_legal"],
                         "evaluated_views": g.get("expanded"), "n_gaussians": g.get("n_gaussians"), "pick_equal": r["pick"] == r["recorded_pick"] if r["recorded_pick"] is not None else None})
        peak = (summ.get("bye") or {}).get("peak_mem_gb")
    else:
        pl = glob.glob(os.path.join(d, "planner_logs", "pl_*.steps.jsonl"))[0]
        peak = 0.0
        for line in open(pl):
            s = json.loads(line)
            t = s["t"]
            tm = s.get("timing") or {}
            enc = (s.get("refresh") or {}).get("encode_s", 0.0)
            dd = (s.get("all_scores") or {}).get("d") or {}
            per = dd.get("per_root") or []
            peak = max(peak, s.get("peak_mem_gb") or 0.0)
            row = {"t": t, "decision": s["decision_s"] - s["p1_s"], "cycle": s["decision_s"], "roundtrip": drv[t]["roundtrip_s"], "map_update": s["p1_s"],
                   "geometry": s["geometry_s"], "geoanchor_refresh": enc, "l1_projection": tm.get("projection_s"), "l1_conditioning": tm.get("cond_s"),
                   "l1_flow_sampling": tm.get("sampling_s"), "l1_decode_and_distp1": tm.get("decode_score_s"), "l1_generation_total": tm.get("frozen_score_state_s"),
                   "policy": tm.get("policy_s"), "n_legal": s["n"], "l2_predicted_views": dd.get("predicted_views"), "l2_blocked": dd.get("blocked"),
                   "search_nodes": sum(r.get("nodes", 0) for r in per) if per else None, "pick_equal": drv[t]["pick"] == drv[t]["recorded_pick"] if drv[t]["recorded_pick"] is not None else None}
            row["other"] = row["decision"] - sum(row[k] or 0.0 for k in ("geometry", "geoanchor_refresh", "l1_generation_total", "policy"))
            eff = tm.get("eff")
            if eff:
                for k, v in eff.items():
                    row["eff." + k] = v
            rows.append(row)
    return m, summ, rows, peak


def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    runs = sorted(d for d in glob.glob(os.path.join(RUNS, "*", "*")) if os.path.exists(os.path.join(d, "summary.json")))
    per_run, all_rows = [], {}
    for d in runs:
        m, summ, rows, peak = load_run(d)
        key = (m, summ["src"])
        all_rows.setdefault(key, []).extend(dict(r, scene=summ["scene"], start=summ["start"]) for r in rows)
        win = [r for r in rows if 10 <= r["t"] <= 99]
        pe = [r["pick_equal"] for r in rows if r["pick_equal"] is not None]
        per_run.append({"method": m, "src": summ["src"], "scene": summ["scene"], "start": summ["start"], "gpu": summ["gpu"], "host": summ["host"],
                        "boot_s": summ["boot_s"], "peak_mem_gb": peak, "picks_equal_recorded": f"{sum(pe)}/{len(pe)}",
                        "decision_median_s": float(np.median([r["decision"] for r in win])) if win else None,
                        "cycle_median_s": float(np.median([r["cycle"] for r in win])) if win else None,
                        "roundtrip_median_s": float(np.median([r["roundtrip"] for r in win])) if win else None,
                        "n_legal_mean": float(np.mean([r["n_legal"] for r in win])) if win else None})
        with open(os.path.join(RAW, f"{m}__{summ['src']}_{summ['scene']}_s{summ['start']}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
    with open(os.path.join(TAB, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0]))
        w.writeheader()
        w.writerows(per_run)
    out = []
    for (m, src), rows in sorted(all_rows.items()):
        for ph, (a, b) in PHASES.items():
            sel = [r for r in rows if a <= r["t"] <= b]
            for metric in ("decision", "cycle", "roundtrip", "map_update", "n_legal", "evaluated_views", "l2_predicted_views"):
                vals = [r[metric] for r in sel if r.get(metric) is not None]
                if vals:
                    out.append({"method": m, "label": LABEL[m], "states_from": src, "phase": ph, "metric": metric,
                                "trajectories": len({(r["scene"], r["start"]) for r in sel}), **stats(vals)})
    with open(os.path.join(TAB, "latency_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "label", "states_from", "phase", "metric", "trajectories", "n", "mean", "median", "p95", "std", "min", "max"])
        w.writeheader()
        w.writerows(out)
    # component means (main window), RAYSCOUT runs
    comp = []
    for (m, src), rows in sorted(all_rows.items()):
        sel = [r for r in rows if 10 <= r["t"] <= 99]
        keys = sorted({k for r in sel for k in r if k not in ("t", "scene", "start", "pick_equal") and isinstance(r.get(k), (int, float))})
        for k in keys:
            vals = [r[k] for r in sel if isinstance(r.get(k), (int, float))]
            comp.append({"method": m, "states_from": src, "component": k, **stats(vals)})
    with open(os.path.join(TAB, "component_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "states_from", "component", "n", "mean", "median", "p95", "std", "min", "max"])
        w.writeheader()
        w.writerows(comp)
    # paired RAYSCOUT vs MAGICIAN on identical states (same trajectory and step)
    a = {(r["scene"], r["start"], r["t"]): r for r in all_rows.get(("OURS-K4D2", "V3-T0-D1"), [])}
    b = {(r["scene"], r["start"], r["t"]): r for r in all_rows.get(("MAG-B", "V3-T0-D1"), [])}
    common = sorted(k for k in a if k in b and 10 <= k[2] <= 99)
    pair = {}
    if common:
        for metric in ("decision", "cycle", "roundtrip"):
            ra = np.asarray([a[k][metric] for k in common]); rb = np.asarray([b[k][metric] for k in common])
            traj = sorted({k[:2] for k in common})
            med_ratio = [float(np.median([b[k][metric] for k in common if k[:2] == tr]) / np.median([a[k][metric] for k in common if k[:2] == tr])) for tr in traj]
            pair[metric] = {"n_states": len(common), "trajectories": len(traj), "rayscout_median": float(np.median(ra)), "magician_median": float(np.median(rb)),
                            "rayscout_mean": float(ra.mean()), "magician_mean": float(rb.mean()), "ratio_of_medians": float(np.median(rb) / np.median(ra)),
                            "ratio_of_means": float(rb.mean() / ra.mean()), "per_trajectory_median_ratio": dict(zip([f"{s}_s{k}" for s, k in traj], med_ratio)),
                            "states_rayscout_faster": int((ra < rb).sum())}
    json.dump(pair, open(os.path.join(TAB, "paired_latency.json"), "w"), indent=1)
    print(json.dumps(pair, indent=1))
    for r in per_run:
        print(r)


if __name__ == "__main__":
    main()
