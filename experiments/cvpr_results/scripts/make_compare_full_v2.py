"""Complete comparison v2 (2026-11-14): v1 (make_compare_full.py, fig/compare_full/, unchanged) + SCONE (EXP9-SCONE, outdoor + indoor)
+ FisherRF-full indoor (f1c4, MP3D-150). Metrics per the SCONE task book: outdoor AUC@20/50/80/100 + Cov@20/50/80/100; indoor
AUC@20/50/80/100/150 + Cov@20/50/100/150. Outputs fig/compare_full_v2/ (new folder; v1 files are not touched) and SCONE result files
in experiments/cvpr_results/scone_baseline/results/.

New sources (each record validated here, nothing typed by hand):
  SCONE outdoor  work_dirs/cvpr_results/scone_baseline/outdoor/closed_loop/episodes/ep_scone_<scene>_s<k>_r0.json, official coverage
                 replay metrics/scone/scone_<scene>_s<k>_r0.json must have official_equal_stored = True (70/70)
  SCONE indoor   work_dirs/cvpr_results/scone_baseline/indoor/closed_loop/SCONE/<tag>/{episode.json, validation.json}: valid = True
                 (EXP7 V1-V3 + same start as MAG-B + provenance + guard), stored curve == validation curve (44/44)
  FisherRF-full indoor  work_dirs/cvpr_results/fisherrf_reproduction/full/indoor/closed_loop_f1c4/FisherRF-full/<tag>/episode.json, 44
                 rows of fisherrf_reproduction/results/indoor_full_f1c4_episodes.csv (validated by frf_in_validate.py); curve AUC@150
                 must match that CSV
Everything else (Random, MACARONS, MAGICIAN, Ours, FisherRF one-step / full outdoor) is loaded exactly as in v1. Scene- / level-equal
means; paired scene / level bootstrap 10,000, seed 0. Usage: python make_compare_full_v2.py (seen2scene env, CPU)."""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402
import make_compare_full as V1  # noqa: E402
import render as RD  # noqa: E402

plt = RD.plt
R = V1.R
OUT = os.path.join(L.FIG, "compare_full_v2")
SC_W = os.path.join(R, "work_dirs", "cvpr_results", "scone_baseline")
SC_RES = os.path.join(L.CV, "scone_baseline", "results")
FRF = V1.FRF
ROWS = ["RANDOM", "MACARONS", "SCONE", "FRF1", "FRF", "MAG-B", "OURS"]
NAME = dict(V1.NAME, SCONE="SCONE†")
STY = dict(V1.STY, SCONE=dict(color="#009E73", ls="--", marker="v", lw=1.4))
OUT_K = [("AUC", 20), ("AUC", 50), ("AUC", 80), ("AUC", 100), ("Cov", 20), ("Cov", 50), ("Cov", 80), ("Cov", 100)]
IN_K = [("AUC", 20), ("AUC", 50), ("AUC", 80), ("AUC", 100), ("AUC", 150), ("Cov", 20), ("Cov", 50), ("Cov", 100), ("Cov", 150)]
lab, met = V1.lab, V1.met


def scone_outdoor():
    d = {}
    for s in L.SCENE_ORDER:
        for k in range(5):
            tag = f"scone_{s}_s{k}_r0"
            ep = json.load(open(os.path.join(SC_W, "outdoor", "closed_loop", "episodes", f"ep_{tag}.json")))
            mt = json.load(open(os.path.join(SC_W, "outdoor", "metrics", "scone", f"{tag}.json")))
            assert mt.get("official_equal_stored") is True and len(ep["coverage"]) == 101 and ep["scene"] == s and int(ep["start"]) == k, tag
            d[(s, k)] = np.asarray(ep["coverage"], float)
    return d


def indoor_extra(keys):
    sc, ff = {}, {}
    frf_rows = {r["tag"]: r for r in csv.DictReader(open(os.path.join(L.CV, "fisherrf_reproduction", "results", "indoor_full_f1c4_episodes.csv")))}
    for (lv, st) in keys:
        tag = f"{lv}_S{st}"
        d = os.path.join(SC_W, "indoor", "closed_loop", "SCONE", tag)
        v = json.load(open(os.path.join(d, "validation.json")))
        c = np.asarray(json.load(open(os.path.join(d, "episode.json")))["coverage"], float)
        assert v.get("valid") is True and len(c) == 151 and np.array_equal(c, np.asarray(v["coverage"], float)), tag
        sc[(lv, st)] = c
        assert tag in frf_rows, tag
        c2 = np.asarray(json.load(open(os.path.join(FRF, "full", "indoor", "closed_loop_f1c4", "FisherRF-full", tag, "episode.json")))["coverage"], float)
        assert len(c2) == 151 and abs(np.trapz(c2) / 150 - float(frf_rows[tag]["AUC@150"])) < 1e-5, tag            # CSV stores 6 decimals
        ff[(lv, st)] = c2
    return sc, ff


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SC_RES, exist_ok=True)
    cu_o, un_o, uo = V1.outdoor()
    cu_o["SCONE"] = scone_outdoor()
    un_o["SCONE"] = {s: np.mean([cu_o["SCONE"][(s, k)] for k in range(5)], 0) for s in uo}
    cu_i, un_i, ui = V1.indoor()
    sc_i, ff_i = indoor_extra(sorted(cu_i["OURS"]))
    cu_i["SCONE"], cu_i["FRF"] = sc_i, ff_i
    for m in ("SCONE", "FRF"):
        un_i[m] = {lv: np.mean([c for (s, _), c in cu_i[m].items() if s == lv], 0) for lv in ui}
    T_out, T_in = V1.table(un_o, uo, OUT_K), V1.table(un_i, ui, IN_K)
    V1.crosscheck({m: dict(T_out[m], **V1.table({m: un_o[m]}, uo, [("Cov", 100)])[m]) for m in T_out},
                  {m: dict(T_in[m]) for m in T_in})                           # v1 rows must be unchanged
    ff_sum = json.load(open(os.path.join(L.CV, "fisherrf_reproduction", "results", "indoor_full_f1c4_summary.json")))
    V1.ROWS, V1.NAME, V1.OUT = ROWS, NAME, OUT
    V1.write_table("compare_outdoor", T_out, OUT_K, "Outdoor: Macarons++ (MAGICIAN benchmark), 100 steps",
                   "14 scenes x 5 official starts, official coverage, scene-equal means; Cov@100 = final coverage. † adapted to this protocol by us "
                   "(SCONE: official MACARONS SCONE code + pretrained occupancy / visibility weights, zero-shot; FisherRF: our reproduction, "
                   "full = frontier targets + Fisher path evaluation, one-step = Fisher score over the legal neighbours).")
    V1.write_table("compare_indoor", T_in, IN_K, "Indoor: MP3D (EXP7 MP3D-150), 150 steps",
                   "22 val levels x 2 starts, level-scale surface coverage, level-equal means; Cov@150 = final coverage. "
                   "— : MACARONS not run on MP3D.")
    V1.STY = STY
    V1.curves(un_o, un_i, uo, ui)
    pr = []
    pairs = [("OURS", m) for m in ROWS if m != "OURS"] + [("SCONE", m) for m in ("RANDOM", "MACARONS", "FRF", "MAG-B")]
    for ds, un, units, keys in (("outdoor", un_o, uo, OUT_K), ("indoor", un_i, ui, IN_K)):
        for a, b in pairs:
            if a not in un or b not in un:
                continue
            for k in keys:
                mm, lo, hi, nh, n = V1.boot([met(un[a][u], [k])[lab(k)] - met(un[b][u], [k])[lab(k)] for u in units])
                pr.append([ds, f"{NAME[a]} - {NAME[b]}", lab(k), f"{mm:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}", f"{nh}/{n}"])
    L.write_csv(os.path.join(OUT, "compare_paired.csv"), ["dataset", "comparison", "metric", "mean", "ci95_lo", "ci95_hi", "units_first_higher"], pr)
    # SCONE result files (per episode curves, per scene / level, summary)
    for ds, cu, keys, units, T in (("outdoor", cu_o, OUT_K, uo, T_out), ("indoor", cu_i, IN_K, ui, T_in)):
        rows = [[s, k] + [f"{v:.6f}" for v in c] for (s, k), c in sorted(cu["SCONE"].items())]
        L.write_csv(os.path.join(SC_RES, f"scone_{ds}_curves.csv"), ["scene", "start"] + [f"c{t}" for t in range(len(rows[0]) - 2)], rows)
        per = [[u] + [f"{met(un_o['SCONE'][u] if ds == 'outdoor' else un_i['SCONE'][u], [k])[lab(k)]:.5f}" for k in keys] for u in units]
        L.write_csv(os.path.join(SC_RES, f"scone_{ds}_per_unit.csv"), ["scene" if ds == "outdoor" else "level"] + [lab(k) for k in keys], per)
    summ = {"outdoor_All-14": T_out["SCONE"], "indoor_ALL-22": T_in["SCONE"], "fisherrf_full_indoor_ALL-22": T_in["FRF"],
            "fisherrf_full_indoor_summary_crosscheck": ff_sum.get("splits", {}).get("ALL-22") if isinstance(ff_sum.get("splits"), dict) else None,
            "episodes": {"outdoor": len(cu_o["SCONE"]), "indoor": len(cu_i["SCONE"])}}
    json.dump(summ, open(os.path.join(SC_RES, "scone_summary.json"), "w"), indent=1, default=float)
    for name in ("compare_outdoor", "compare_indoor"):
        print(open(os.path.join(OUT, f"{name}.csv")).read())


if __name__ == "__main__":
    main()
