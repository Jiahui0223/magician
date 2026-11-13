"""Complete comparison incl. our FisherRF reproductions (user request 2026-11-13): outdoor (Macarons++ All-14, 100 steps) and indoor
(MP3D-150 ALL-22, 150 steps) tables with AUC@20/50/80/100(/150) and Cov@20/50/100(/150), coverage curves, paired differences.

Methods and sources (every curve is a validated record; nothing typed by hand, no paper-reported numbers):
  outdoor  Random Walk / MACARONS / MAGICIAN / Ours: manifest/episodes.csv COMPLETE rows (cvpr_lib.load_manifest; official coverage,
           replay-verified when the manifest was built)
           FisherRF (full, f1c4): work_dirs/cvpr_results/fisherrf_reproduction/full/outdoor/closed_loop_f1c4/<scene>_s<k>/
           FisherRF (one-step, D3a): work_dirs/cvpr_results/fisherrf_reproduction/outdoor/closed_loop/<scene>_s<k>/
           (both: official coverage replay must equal the stored curve, official_equal_stored)
  indoor   Random Walk / MAGICIAN / Ours: exp7/manifests/mp3d150_manifest.csv output_path, valid = 1 in exp7/results/mp3d150_episodes.csv
           FisherRF (one-step, D3a): work_dirs/cvpr_results/fisherrf_reproduction/indoor/closed_loop/FisherRF/<tag>/episode.json (44/44 valid
           in fisherrf_reproduction/results/indoor_episodes.csv); FisherRF (full) and MACARONS were not run indoors -> dash.
Metrics on C[0..T]: AUC@t = trapz(C[0..t]) / t, Cov@t = C[t]; outdoor scene-equal (5 starts per scene), indoor level-equal (2 starts per
level). Cross-checked against Table 1 / EXP7 / FisherRF summaries. Paired: Ours - baseline, scene / level bootstrap 10,000, seed 0.
Outputs fig/compare_full/: compare_{outdoor,indoor}.{csv,tex,pdf,png}, compare_curves.{pdf,png}, compare_curves_data.csv,
compare_paired.csv. Usage: python make_compare_full.py (seen2scene env, CPU)."""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402
import render as RD  # noqa: E402

plt = RD.plt
R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
OUT = os.path.join(L.FIG, "compare_full")
FRF = os.path.join(R, "work_dirs", "cvpr_results", "fisherrf_reproduction")
EXP7 = os.path.join(L.CV, "exp7")
ROWS = ["RANDOM", "MACARONS", "FRF1", "FRF", "MAG-B", "OURS"]
NAME = {"RANDOM": "Random Walk†", "MACARONS": "MACARONS†", "FRF1": "FisherRF† (one-step)", "FRF": "FisherRF† (full)", "MAG-B": "MAGICIAN", "OURS": "Ours"}
STY = {"RANDOM": RD.STYLE["RANDOM"], "MACARONS": RD.STYLE["MACARONS"], "MAG-B": RD.STYLE["MAG-B"], "OURS": RD.STYLE["V3-T0-D1"],
       "FRF": dict(color="#CC79A7", ls="-", marker="D", lw=1.4), "FRF1": dict(color="#CC79A7", ls=":", marker="d", lw=1.2)}
OUT_K = [("AUC", 20), ("AUC", 50), ("AUC", 80), ("AUC", 100), ("Cov", 20), ("Cov", 50), ("Cov", 100)]
IN_K = [("AUC", 20), ("AUC", 50), ("AUC", 80), ("AUC", 100), ("AUC", 150), ("Cov", 20), ("Cov", 50), ("Cov", 100), ("Cov", 150)]
lab = lambda k: f"{k[0]}@{k[1]}"


def met(c, keys):
    c = np.asarray(c, float)
    return {lab(k): (np.trapz(c[:k[1] + 1]) / k[1] if k[0] == "AUC" else c[k[1]]) for k in keys}


def frf_out(root):
    d = {}
    for s in L.SCENE_ORDER:
        for k in range(5):
            u = f"{s}_s{k}"
            ep = json.load(open(os.path.join(root, u, "episodes", f"ep_FisherRF_{u}_r0.json")))
            mt = json.load(open(os.path.join(root, u, "metrics", "FisherRF", f"FisherRF_{u}_r0.json")))
            assert mt.get("official_equal_stored") is True and len(ep["coverage"]) == 101, u
            d[(s, k)] = np.asarray(ep["coverage"], float)
    return d


def outdoor():
    cur = {"FRF": frf_out(os.path.join(FRF, "full", "outdoor", "closed_loop_f1c4")), "FRF1": frf_out(os.path.join(FRF, "outdoor", "closed_loop"))}
    for m, mid in (("RANDOM", "RANDOM"), ("MACARONS", "MACARONS"), ("MAG-B", "MAG-B"), ("OURS", "V3-T0-D1")):
        E = L.load_manifest(mid)
        assert len(E) == 70, (m, len(E))
        cur[m] = {key: np.asarray(r["coverage_official"], float) for key, r in E.items()}
    unit = {m: {s: np.mean([cur[m][(s, k)] for k in range(5)], 0) for s in L.SCENE_ORDER} for m in cur}
    return cur, unit, L.SCENE_ORDER


def indoor():
    valid = {(r["method"], r["scene_id"], r["start_id"]) for r in csv.DictReader(open(os.path.join(EXP7, "results", "mp3d150_episodes.csv"))) if r["valid"] == "1"}
    frf_ok = {r["tag"] for r in csv.DictReader(open(os.path.join(L.CV, "fisherrf_reproduction", "results", "indoor_episodes.csv")))}
    cur = {m: {} for m in ("RANDOM", "MAG-B", "OURS", "FRF1")}
    for r in csv.DictReader(open(os.path.join(EXP7, "manifests", "mp3d150_manifest.csv"))):
        key = (r["scene_id"], r["start_id"])
        if r["method"] in cur and r["method"] != "FRF1":
            assert (r["method"], *key) in valid, (r["method"], key)
            cur[r["method"]][key] = np.asarray(json.load(open(os.path.join(R, r["output_path"])))["coverage"], float)
        if r["method"] == "OURS":
            tag = f"{r['scene_id']}_S{r['start_id']}"
            assert tag in frf_ok, tag
            cur["FRF1"][key] = np.asarray(json.load(open(os.path.join(FRF, "indoor", "closed_loop", "FisherRF", tag, "episode.json")))["coverage"], float)
    levels = sorted({k[0] for k in cur["OURS"]})
    assert len(levels) == 22 and all(len(v) == 44 and all(len(c) == 151 for c in v.values()) for v in cur.values())
    unit = {m: {lv: np.mean([c for (s, _), c in cur[m].items() if s == lv], 0) for lv in levels} for m in cur}
    return cur, unit, levels


def table(unit, units, keys):
    return {m: {lab(k): float(np.mean([met(unit[m][u], [k])[lab(k)] for u in units])) for k in keys} for m in unit}


def crosscheck(T_out, T_in):
    t1 = {r["method_id"]: r for r in csv.DictReader(open(os.path.join(L.FIG, "table1", "table1_all14_full.csv")))}
    for m, mid in (("RANDOM", "RANDOM"), ("MACARONS", "MACARONS"), ("MAG-B", "MAG-B"), ("OURS", "V3-T0-D1")):
        for a, b in (("AUC@20", "AUC@20"), ("AUC@100", "AUC@100"), ("Cov@100", "Final")):
            assert abs(T_out[m][a] - float(t1[mid][b])) < 5e-4, (m, a, T_out[m][a], t1[mid][b])
    ff = json.load(open(os.path.join(L.CV, "fisherrf_reproduction", "results", "full14_f1c4_table.json")))["rows"]
    assert abs(T_out["FRF"]["AUC@100"] - ff["FisherRF-full"]["AUC@100"]) < 1e-9 and abs(T_out["FRF1"]["AUC@100"] - ff["FisherRF-score-1step (D3a)"]["AUC@100"]) < 1e-9
    s7 = {r["method"]: r for r in csv.DictReader(open(os.path.join(EXP7, "results", "mp3d150_summary.csv"))) if r["split"] == "ALL-22"}
    for m in ("RANDOM", "MAG-B", "OURS"):
        for a in ("AUC@20", "AUC@150", "Cov@100"):
            assert abs(T_in[m][a] - float(s7[m][a])) < 5e-4, (m, a, T_in[m][a], s7[m][a])
    fi = json.load(open(os.path.join(L.CV, "fisherrf_reproduction", "results", "summary.json")))["indoor"]["ALL-22"]["FisherRF"]
    assert abs(T_in["FRF1"]["AUC@150"] - fi["AUC@150"]) < 5e-4, (T_in["FRF1"]["AUC@150"], fi["AUC@150"])


def boot(d):
    d = np.asarray(d, float)
    b = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int((d > 0).sum()), len(d)


def write_table(name, T, keys, title, note):
    rows_m = [m for m in ROWS]
    have = [m for m in rows_m if m in T]
    body, bold = [], set()
    for i, m in enumerate(rows_m):
        body.append([NAME[m]] + ([f"{T[m][lab(k)]:.3f}" for k in keys] if m in T else ["—"] * len(keys)))
    for j, k in enumerate(keys):
        best = max(round(T[m][lab(k)], 3) for m in have)
        bold |= {(i, j + 1) for i, m in enumerate(rows_m) if m in T and round(T[m][lab(k)], 3) == best}
    hdr = ["Method"] + [lab(k) + " ↑" for k in keys]
    L.write_csv(os.path.join(OUT, f"{name}.csv"), ["Method"] + [lab(k) for k in keys], body)
    RD.table(os.path.join(OUT, name), hdr, body, bold=bold, notes=[note], title=title, rules_after={len(rows_m) - 2})
    RD.tex(os.path.join(OUT, f"{name}.tex"), ["Method"] + [lab(k) + r" $\uparrow$" for k in keys], body, bold=bold, caption=title + ". " + note,
           label=f"tab:{name}", rules_after={len(rows_m) - 2},
           notes=["generated by experiments/cvpr_results/scripts/make_compare_full.py -- do not edit by hand (needs booktabs)"])


def curves(cu_out, cu_in, units_out, units_in):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    rows = []
    for ax, cu, units, T, title in ((axs[0], cu_out, units_out, 100, "Outdoor · Macarons++ (14 scenes × 5 starts)"),
                                    (axs[1], cu_in, units_in, 150, "Indoor · MP3D (22 levels × 2 starts)")):
        x = np.arange(T + 1)
        for m in ROWS:
            if m not in cu:
                continue
            Y = np.stack([cu[m][u] for u in units])
            mu = Y.mean(0)
            b = Y[np.random.default_rng(0).integers(0, len(units), (2000, len(units)))].mean(1)
            lo, hi = np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)
            st = STY[m]
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12, lw=0)
            ax.plot(x, mu, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], markevery=max(1, T // 10), ms=3.2, label=NAME[m])
            rows += [[title.split(" ·")[0], NAME[m], t, f"{mu[t]:.5f}", f"{lo[t]:.5f}", f"{hi[t]:.5f}"] for t in range(T + 1)]
        ax.set_xlim(0, T); ax.set_ylim(0, 1.0)
        ax.set_xlabel("step"); ax.set_title(title, fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("coverage")
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=6, fontsize=7.2, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    RD.save(fig, os.path.join(OUT, "compare_curves"))
    L.write_csv(os.path.join(OUT, "compare_curves_data.csv"), ["dataset", "method", "step", "mean", "ci95_lo", "ci95_hi"], rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    cu_o, un_o, uo = outdoor()
    cu_i, un_i, ui = indoor()
    T_out, T_in = table(un_o, uo, OUT_K), table(un_i, ui, IN_K)
    crosscheck(T_out, T_in)
    write_table("compare_outdoor", T_out, OUT_K, "Outdoor: Macarons++ (MAGICIAN benchmark), 100 steps",
                "14 scenes x 5 official starts, official coverage, scene-equal means; Cov@100 = final coverage. † adapted to this protocol by us "
                "(FisherRF: our reproduction; full = frontier targets + Fisher path evaluation, one-step = Fisher score over the legal neighbours).")
    write_table("compare_indoor", T_in, IN_K, "Indoor: MP3D (EXP7 MP3D-150), 150 steps",
                "22 val levels x 2 starts, level-scale surface coverage, level-equal means; Cov@150 = final coverage. "
                "— : not run indoors (FisherRF full indoor version not implemented; MACARONS not run on MP3D).")
    curves(un_o, un_i, uo, ui)
    pr = []
    for ds, un, units, keys in (("outdoor", un_o, uo, OUT_K), ("indoor", un_i, ui, IN_K)):
        for m in ROWS:
            if m == "OURS" or m not in un:
                continue
            for k in keys:
                mm, lo, hi, nh, n = boot([met(un["OURS"][u], [k])[lab(k)] - met(un[m][u], [k])[lab(k)] for u in units])
                pr.append([ds, f"Ours - {NAME[m]}", lab(k), f"{mm:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}", f"{nh}/{n}"])
    L.write_csv(os.path.join(OUT, "compare_paired.csv"), ["dataset", "comparison", "metric", "mean", "ci95_lo", "ci95_hi", "units_ours_higher"], pr)
    for name in ("compare_outdoor", "compare_indoor"):
        print(open(os.path.join(OUT, f"{name}.csv")).read())


if __name__ == "__main__":
    main()
