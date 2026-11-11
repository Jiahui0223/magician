"""EXP8-MP3D -- aggregate the indoor replay runs (CPU) and build the Indoor / Outdoor Table 2 draft.
Indoor timing definitions (planner answers of work_dirs/cvpr_results/efficiency/runs_in/<METHOD>/<tag>/steps.jsonl):
  RAYSCOUT  cycle = planner decision_s (step start -> choice: map update incl. the region rebuild at t % 10 == 0, geometry, GeoAnchor
            encode at refreshes, both generation layers, CDGR, choice); map_update = eff.map_update_s (all P1 integrate_record calls of
            the step); decision = cycle - map_update
  MAGICIAN  cycle = decision_s (observe + plan); map_update = observe_s; decision = occ_s + novelty_s + beam_s
Main window t = 10..149 (140 decisions per trajectory, exactly 14 region refreshes); phases early 1..20, middle 21..50, late 51..149.
Outdoor numbers are read from tables/latency_stats.csv (EXP8 outdoor, unchanged). Outputs: tables/in_latency_stats.csv,
tables/in_component_stats.csv, tables/in_paired_latency.json, tables/T2_indoor_outdoor.{csv,md,tex}, raw_in/*.jsonl."""
import csv
import glob
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
RUNS = os.path.join(R, "work_dirs", "cvpr_results", "efficiency", "runs_in")
E = os.path.join(R, "experiments", "cvpr_results", "efficiency")
TAB, RAW = os.path.join(E, "tables"), os.path.join(E, "raw_in")
PH = {"main_t10_149": (10, 149), "early_t1_20": (1, 20), "middle_t21_50": (21, 50), "late_t51_149": (51, 149)}


def st(x):
    x = np.asarray([v for v in x if v is not None], float)
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "std": float(x.std(ddof=1))} if len(x) else {"n": 0}


def rows_of(d, m):
    out = []
    for line in open(os.path.join(d, "steps.jsonl")):
        r = json.loads(line)
        tm = r["timing"] or {}
        if m == "MAG-B":
            dec = tm["occ_s"] + tm["novelty_s"] + tm["beam_s"]
            row = {"t": r["t"], "cycle": tm["decision_s"], "map_update": tm["observe_s"], "decision": dec, "roundtrip": r["roundtrip_s"],
                   "occupancy": tm["occ_s"], "novelty": tm["novelty_s"], "beam": tm["beam_s"], "views": (r["diag"] or {}).get("expanded"), "n_legal": r["n_legal"]}
        else:
            eff = tm.get("eff", {})
            mu = eff.get("map_update_s", None)
            row = {"t": r["t"], "cycle": tm["decision_s"], "map_update": mu, "decision": tm["decision_s"] - mu if mu is not None else None, "roundtrip": r["roundtrip_s"],
                   "refresh": r["refresh"], "n_legal": r["n_legal"], "views": eff.get("n_gen_l1_views", 0) + eff.get("n_gen_l2_views", 0),
                   "l2_views": eff.get("n_gen_l2_views"), **{"eff." + k: v for k, v in eff.items()}}
        row["pick_equal"] = r["pick"] == r["recorded_pick"] if r["recorded_pick"] is not None else None
        out.append(row)
    return out


def main():
    os.makedirs(TAB, exist_ok=True); os.makedirs(RAW, exist_ok=True)
    data, runs = {}, []
    for d in sorted(glob.glob(os.path.join(RUNS, "*", "*"))):
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        m, tag = d.split(os.sep)[-2:]
        s = json.load(open(os.path.join(d, "summary.json")))
        rr = rows_of(d, "MAG-B" if m == "MAG-B" else "OURS")
        data.setdefault(m, {})[tag] = rr
        with open(os.path.join(RAW, f"{m}__{tag}.jsonl"), "w") as f:
            for r in rr:
                f.write(json.dumps(r, default=float) + "\n")
        win = [r for r in rr if 10 <= r["t"] <= 149]
        runs.append({"method": m, "tag": tag, "gpu": s["gpu"], "host": s["host"], "picks_equal_recorded": s["picks_equal_recorded"], "peak_mem_gb": s["peak_mem_gb"],
                     "decision_median_s": st([r["decision"] for r in win]).get("median"), "cycle_median_s": st([r["cycle"] for r in win]).get("median")})
    with open(os.path.join(TAB, "in_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0])); w.writeheader(); w.writerows(runs)
    out = []
    for m, by in data.items():
        rows = [dict(r, tag=t) for t, rr in by.items() for r in rr]
        for ph, (a, b) in PH.items():
            sel = [r for r in rows if a <= r["t"] <= b]
            for metric in ("decision", "cycle", "roundtrip", "map_update", "views", "n_legal"):
                v = [r.get(metric) for r in sel]
                if any(x is not None for x in v):
                    out.append({"method": m, "phase": ph, "metric": metric, "trajectories": len({r["tag"] for r in sel}), **st(v)})
    with open(os.path.join(TAB, "in_latency_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "phase", "metric", "trajectories", "n", "mean", "median", "p95", "std"]); w.writeheader(); w.writerows(out)
    comp = []
    for m, by in data.items():
        sel = [r for rr in by.values() for r in rr if 10 <= r["t"] <= 149]
        for k in sorted({k for r in sel for k in r if isinstance(r.get(k), (int, float)) and k not in ("t",)}):
            comp.append({"method": m, "component": k, **st([r.get(k) for r in sel if isinstance(r.get(k), (int, float))])})
    with open(os.path.join(TAB, "in_component_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "component", "n", "mean", "median", "p95", "std"]); w.writeheader(); w.writerows(comp)
    # paired on identical states
    A = {(t, r["t"]): r for t, rr in data.get("OURS-PROF", {}).items() for r in rr if 10 <= r["t"] <= 149}
    B = {(t, r["t"]): r for t, rr in data.get("MAG-B", {}).items() for r in rr if 10 <= r["t"] <= 149}
    common = sorted(k for k in A if k in B)
    pair = {}
    for metric in ("decision", "cycle", "roundtrip"):
        if not common:
            break
        a = np.asarray([A[k][metric] for k in common]); b = np.asarray([B[k][metric] for k in common])
        trajs = sorted({k[0] for k in common})
        pair[metric] = {"n_states": len(common), "trajectories": len(trajs), "rayscout_median": float(np.median(a)), "magician_median": float(np.median(b)),
                        "ratio_magician_over_rayscout_medians": float(np.median(b) / np.median(a)), "states_rayscout_faster": int((a < b).sum()),
                        "per_trajectory_ratio": {t: float(np.median([B[k][metric] for k in common if k[0] == t]) / np.median([A[k][metric] for k in common if k[0] == t])) for t in trajs}}
    json.dump(pair, open(os.path.join(TAB, "in_paired_latency.json"), "w"), indent=1)
    # overhead check: plain OURS vs OURS-PROF on the same trajectories
    ov = {}
    for t, rr in data.get("OURS", {}).items():
        p = data.get("OURS-PROF", {}).get(t)
        if p:
            ov[t] = {"plain_cycle_median": float(np.median([r["cycle"] for r in rr if 10 <= r["t"] <= 149])),
                     "prof_cycle_median": float(np.median([r["cycle"] for r in p if 10 <= r["t"] <= 149])),
                     "picks_equal": sum(x["pick_equal"] for x in rr if x["pick_equal"] is not None)}
    json.dump(ov, open(os.path.join(TAB, "in_timer_overhead.json"), "w"), indent=1)
    # Table 2: indoor / outdoor
    outd = {(r["method"], r["phase"], r["metric"]): r for r in csv.DictReader(open(os.path.join(TAB, "latency_stats.csv"))) if r["states_from"] == "V3-T0-D1"}
    ind = {(r["method"], r["phase"], r["metric"]): r for r in out}
    oruns = list(csv.DictReader(open(os.path.join(TAB, "runs.csv"))))
    pk = lambda m: [float(r["peak_mem_gb"]) for r in oruns if r["method"] == m and r["src"] == "V3-T0-D1"]
    T2 = []
    for env, src, mo, mm, ph, n_tr in (("Outdoor (Macarons++)", outd, "OURS-K4D2", "MAG-B", "main_t10_99", 7), ("Indoor (MP3D)", ind, "OURS-PROF", "MAG-B", "main_t10_149", None)):
        for m, lab in ((mm, "MAGICIAN beam"), (mo, "RAYSCOUT")):
            d, c = src.get((m, ph, "decision")), src.get((m, ph, "cycle"))
            if not d:
                continue
            if env.startswith("Outdoor"):
                peaks = pk(m)
            else:
                peaks = [r["peak_mem_gb"] for r in runs if r["method"] == m]
            mu = src.get((m, ph, "map_update"))
            T2.append([env, lab, d["trajectories"], d["n"], f"{float(d['median']):.2f}", f"{float(d['p95']):.2f}", f"{float(d['mean']):.2f} ± {float(d['std']):.2f}",
                       f"{float(c['median']):.2f}", f"{float(c['mean']):.2f}", f"{float(c['p95']):.2f}", f"{float(mu['mean']):.2f}" if mu else "-",
                       f"{np.median(peaks):.1f} ({min(peaks):.1f}–{max(peaks):.1f})"])
    hdr = ["Environment", "Method", "Trajectories", "Decisions", "Decision median (s)", "Decision P95 (s)", "Decision mean ± std (s)", "Cycle incl. map update, median (s)",
           "Cycle mean (s)", "Cycle P95 (s)", "Map update mean (s)", "Peak GPU mem GB (median, range)"]
    with open(os.path.join(TAB, "T2_indoor_outdoor.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(T2)
    md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in T2)
    open(os.path.join(TAB, "T2_indoor_outdoor.md"), "w").write(md)
    tex = [r"% generated by experiments/cvpr_results/efficiency/scripts/eff_in_analyze.py -- draft", r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}llcccc@{}}", r"\toprule", r" & & \multicolumn{2}{c}{Decision (s) $\downarrow$} & Cycle (s) $\downarrow$ & Peak mem. \\", r"\cmidrule(lr){3-4}",
           r"Env. & Method & median & P95 & mean & (GB) \\", r"\midrule"]
    for i, r in enumerate(T2):
        if i and T2[i - 1][0] != r[0]:
            tex.append(r"\midrule")
        env = r[0].split(" (")[0] if (i == 0 or T2[i - 1][0] != r[0]) else ""
        name = r"\textbf{Ours}" if r[1] == "RAYSCOUT" else "MAGICIAN"
        tex.append(f"{env} & {name} & {r[4]} & {r[5]} & {r[8]} & {r[11].split(' ')[0]} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\caption{Per-step decision latency (map state available $\rightarrow$ next action; online map update excluded) on identical real "
            r"planning states, one dedicated NVIDIA L40S per method; Cycle = decision + online map update (mean; the indoor RAYSCOUT planner rebuilds its region map every 10 steps). Outdoor: Macarons++ (7 trajectories, steps 10--99); Indoor: MP3D (8 trajectories, steps 10--149).}",
            r"\label{tab:runtime}", r"\end{table}"]
    open(os.path.join(TAB, "T2_indoor_outdoor.tex"), "w").write("\n".join(tex) + "\n")
    print(md)
    print(json.dumps(pair, indent=1))
    print(json.dumps(ov, indent=1))


if __name__ == "__main__":
    main()
