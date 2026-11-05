"""CVPR protocol audit -- AUC / Final / coverage-definition audit and paper-vs-reproduced comparison (read-only; magician env
for LMDB access, CPU).
Part 1 (formula tests): the Table 1 metric function (experiments/cvpr_results/table1/scripts/aggregate.py:metrics, imported
  unchanged) and MAGICIAN's evaluation_lmdb.compute_auc (official e697009, re-implemented verbatim: np.trapz(c[:100],
  x=arange(100)), plus its 'Max Cov' = max(c[:100])) on synthetic curves with known integrals (constant, linear ramp, late
  step, start-only, end-only) -> endpoints / normalisation.
Part 2 (Table 1 data): for every model, Final = c[100] vs max(c[0..100]) vs max(c[0..99]); share of non-monotone curves;
  official vs monotone coverage; scene-first vs episode-pooled averaging (re-derived from the canonical episodes and
  compared with results/summary.json).
Part 3 (paper vs reproduced): per scene, MAGICIAN paper Table 6 / 7 (transcribed from the PDF text, checked against
  table1/SCENE_REGISTRY.yaml), MAG-B reproduced under the frozen Table 1 protocol, and our upstream-loop LMDB
  reproduction, under several AUC / Final definitions; All-15 (paper) and All-14 means; per-scene start std.
Outputs: protocol_audit/SCENE_BY_SCENE_PAPER_VS_REPRODUCED.csv, protocol_audit/results/metric_audit.json."""
import csv
import glob
import json
import os
import pickle
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy"
S2S = f"{R}/seen2scene"
T1 = f"{S2S}/experiments/cvpr_results/table1"
EP = f"{S2S}/work_dirs/cvpr_results/table1/episodes"
OUT = f"{S2S}/experiments/cvpr_results/protocol_audit"
LMDB = f"{R}/MAGICIAN/results/scene_exploration/magician_lmdb"
sys.path.insert(0, os.path.join(T1, "scripts"))
from aggregate import metrics as t1_metrics  # noqa: E402

PAPER_NAME = {"dunnottar": "Dunnottar Castle", "colosseum": "Colosseum", "bannerman": "Bannerman Castle", "pantheon": "Pantheon",
              "redeemer": "Christ the Redeemer", "liberty": "Statue of Liberty", "pisa": "Pisa Cathedral", "fushimi": "Fushimi Castle",
              "alhambra": "Alhambra Palace", "neuschwanstein": "Neuschwanstein Castle", "eiffel": "Eiffel Tower", "bridge": "Manhattan Bridge",
              "sofia_church": "St. Sofia Church", "barts": "Barts", "sestino_museum": "Sestino Museum"}
ORDER = list(PAPER_NAME)
METHODS = ["random_walk", "scone", "macarons", "fisherrf", "magician"]
# MAGICIAN paper (arXiv 2603.22650, local MAGICIAN/Magician.pdf, appendix B.2, page 12): Table 6 (AUC) and Table 7 (Final),
# columns Rand. Walk / SCONE / MACARONS / FisherRF / MAGICIAN. Transcribed from `pdftotext -layout` output; cross-checked below
# against table1/SCENE_REGISTRY.yaml (magician_paper fields, transcribed independently on 2026-11-03).
PAPER_AUC = {"dunnottar": [0.149, 0.366, 0.618, 0.500, 0.745], "colosseum": [0.219, 0.589, 0.656, 0.551, 0.704], "bannerman": [0.192, 0.559, 0.575, 0.595, 0.761],
             "pantheon": [0.198, 0.465, 0.601, 0.270, 0.644], "redeemer": [0.439, 0.772, 0.859, 0.727, 0.793], "liberty": [0.323, 0.632, 0.711, 0.553, 0.797],
             "pisa": [0.290, 0.486, 0.678, 0.486, 0.723], "fushimi": [0.279, 0.689, 0.718, 0.565, 0.766], "alhambra": [0.126, 0.369, 0.567, 0.462, 0.631],
             "neuschwanstein": [0.184, 0.325, 0.452, 0.375, 0.608], "eiffel": [0.333, 0.683, 0.709, 0.616, 0.754], "bridge": [0.258, 0.632, 0.750, 0.637, 0.705],
             "sofia_church": [0.280, 0.532, 0.621, 0.608, 0.710], "barts": [0.214, 0.551, 0.660, 0.673, 0.831], "sestino_museum": [0.132, 0.367, 0.537, 0.571, 0.637]}
PAPER_AUC_AVG = [0.241, 0.534, 0.647, 0.546, 0.721]
PAPER_FIN = {"dunnottar": [0.225, 0.527, 0.820, 0.809, 0.975], "colosseum": [0.272, 0.755, 0.794, 0.757, 0.872], "bannerman": [0.240, 0.722, 0.834, 0.801, 0.917],
             "pantheon": [0.309, 0.610, 0.796, 0.444, 0.842], "redeemer": [0.581, 0.924, 0.967, 0.876, 0.973], "liberty": [0.443, 0.819, 0.909, 0.819, 0.947],
             "pisa": [0.353, 0.566, 0.865, 0.776, 0.941], "fushimi": [0.449, 0.844, 0.853, 0.814, 0.931], "alhambra": [0.162, 0.473, 0.775, 0.615, 0.852],
             "neuschwanstein": [0.223, 0.444, 0.551, 0.582, 0.848], "eiffel": [0.541, 0.856, 0.915, 0.827, 0.923], "bridge": [0.356, 0.781, 0.924, 0.877, 0.955],
             "sofia_church": [0.331, 0.619, 0.795, 0.865, 0.891], "barts": [0.240, 0.677, 0.768, 0.878, 0.996], "sestino_museum": [0.141, 0.430, 0.713, 0.842, 0.924]}
PAPER_FIN_AVG = [0.324, 0.670, 0.819, 0.786, 0.919]


def auc_official_lmdb(c):                      # evaluation_lmdb.py compute_auc on coverage[:100] (official e697009, verbatim)
    y = np.asarray(c, float)[:100]
    return float(np.trapz(y, np.arange(len(y))))


def defs(c):
    c = np.asarray(c, float)
    return {"A_trapz101_div100": float(np.trapz(c, dx=1.0) / 100), "B_mean101": float(c.mean()), "C_lmdb_raw": auc_official_lmdb(c),
            "C_div99": auc_official_lmdb(c) / 99, "C_div100": auc_official_lmdb(c) / 100, "mean_c1_100": float(c[1:].mean()),
            "final_c100": float(c[100]), "max_c0_100": float(c.max()), "max_c0_99": float(c[:100].max())}


def tests():
    T = {"constant 1": np.ones(101), "ramp t/100": np.arange(101) / 100.0, "late step (1 from t=50)": (np.arange(101) >= 50).astype(float),
         "start only (c0=1)": (np.arange(101) == 0).astype(float), "end only (c100=1)": (np.arange(101) == 100).astype(float)}
    out = {}
    for n, c in T.items():
        m = t1_metrics(c)
        d = defs(c)
        out[n] = {"table1_auc100": m["auc100"], "table1_auc20": m["auc20"], "table1_auc50": m["auc50"], "table1_final": m["final"], **d}
    # expected closed forms for the Table 1 rule A = trapz(c[0..k]) / k
    exp = {"constant 1": 1.0, "ramp t/100": 0.5, "late step (1 from t=50)": 0.505, "start only (c0=1)": 0.005, "end only (c100=1)": 0.005}
    ok = {n: abs(out[n]["table1_auc100"] - exp[n]) < 1e-12 for n in exp}
    return out, exp, ok


def episodes():
    E = {}
    for f in glob.glob(os.path.join(EP, "*", "*.json")):
        r = json.load(open(f))
        E[(r["model"], r["scene"], int(r["start"]))] = r
    return E


def lmdb_curves():
    cache = os.path.join(OUT, "results", "lmdb_coverage_cache.npz")
    try:
        import lmdb
    except ImportError:                       # seen2scene env: use the cache written by the magician env (same keys / values)
        if not os.path.exists(cache):
            return {}
        z = np.load(cache)
        return {(k.rsplit("_s", 1)[0], int(k.rsplit("_s", 1)[1])): z[k] for k in z.files}
    env = lmdb.open(LMDB, readonly=True, lock=False)
    out = {}
    with env.begin() as txn:
        for key, val in txn.cursor():
            k = key.decode()
            if "/point_cloud" in k or k.count("/") != 1:
                continue
            d = pickle.loads(val)
            if "coverage" in d:
                s, i = k.split("/")
                out[(s, int(i))] = np.asarray(d["coverage"], float)
    env.close()
    np.savez_compressed(cache, **{f"{s}_s{i}": v for (s, i), v in out.items()})
    return out


def main():
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    res = {}
    res["formula_tests"], res["formula_expected"], res["formula_ok"] = tests()
    # registry cross-check of the paper transcription
    import yaml
    reg = yaml.safe_load(open(os.path.join(T1, "SCENE_REGISTRY.yaml")))
    mism = []
    by_id = {it["local_id"]: it for it in reg["scenes"]}
    for s in ORDER:
        mp = by_id.get(s, {}).get("magician_paper")
        if mp:
            for j, m in enumerate(METHODS):
                for key, tab in (("auc", PAPER_AUC), ("final", PAPER_FIN)):
                    v = mp[key][m]
                    if abs(float(v) - tab[s][j]) > 1e-9:
                        mism.append((s, key, m, v, tab[s][j]))
        else:
            mism.append((s, "missing in registry"))
    res["paper_transcription_vs_registry_mismatches"] = mism
    for j in range(5):                                                    # paper averages are the 15-scene means
        res.setdefault("paper_avg_check", {})[METHODS[j]] = {"auc_mean15": float(np.mean([PAPER_AUC[s][j] for s in ORDER])), "auc_reported": PAPER_AUC_AVG[j],
                                                              "final_mean15": float(np.mean([PAPER_FIN[s][j] for s in ORDER])), "final_reported": PAPER_FIN_AVG[j]}
    E = episodes()
    models = sorted({m for m, _, _ in E})
    # Part 2: final definitions, monotonicity, averaging
    p2 = {}
    S = json.load(open(os.path.join(T1, "results", "summary.json")))
    for m in models:
        rows = [(s, k, np.asarray(r["coverage_official"], float), np.asarray(r["coverage_mono"], float)) for (mm, s, k), r in E.items() if mm == m]
        if not rows:
            continue
        nonmono = [float(np.max(np.maximum.accumulate(c) - c)) for _, _, c, _ in rows]
        dmax = [float(c.max() - c[100]) for _, _, c, _ in rows]
        scenes = sorted({s for s, _, _, _ in rows})
        full = [s for s in scenes if sum(1 for x in rows if x[0] == s) == 5 and s != "sofia_church"]
        sm = lambda f: float(np.mean([np.mean([f(c, cm) for ss, _, c, cm in rows if ss == s]) for s in full])) if full else None  # noqa: E731
        ep = lambda f: float(np.mean([f(c, cm) for ss, _, c, cm in rows if ss in full])) if full else None  # noqa: E731
        p2[m] = {"episodes": len(rows), "complete_scenes": len(full), "share_nonmonotone": float(np.mean([x > 1e-12 for x in nonmono])),
                 "max_drop_mean": float(np.mean(nonmono)), "final_vs_max_mean_gap": float(np.mean(dmax)), "share_final_below_max": float(np.mean([x > 1e-12 for x in dmax])),
                 "scene_first": {"auc100": sm(lambda c, cm: np.trapz(c) / 100), "final": sm(lambda c, cm: c[100]), "final_max": sm(lambda c, cm: c.max()),
                                 "auc100_mono": sm(lambda c, cm: np.trapz(cm) / 100), "final_mono": sm(lambda c, cm: cm[100])},
                 "episode_pooled": {"auc100": ep(lambda c, cm: np.trapz(c) / 100), "final": ep(lambda c, cm: c[100])}}
    res["table1_definitions"] = p2
    # Part 3: paper vs reproduced
    L = lmdb_curves()
    rows = []
    for s in ORDER:
        r = {"scene": s, "paper_name": PAPER_NAME[s]}
        for j, m in enumerate(METHODS):
            r[f"paper_auc_{m}"] = PAPER_AUC[s][j]
            r[f"paper_final_{m}"] = PAPER_FIN[s][j]
        mb = [np.asarray(E[("MAG-B", s, k)]["coverage_official"], float) for k in range(5) if ("MAG-B", s, k) in E]
        lm = [L[(s, k)] for k in range(5) if (s, k) in L]
        for tag, cs in (("rep_magb", mb), ("rep_lmdb", lm)):
            if len(cs) == 5:
                D = [defs(c) for c in cs]
                for q in ("A_trapz101_div100", "B_mean101", "C_div99", "C_div100", "final_c100", "max_c0_99", "max_c0_100"):
                    r[f"{tag}_{q}"] = float(np.mean([d[q] for d in D]))
                r[f"{tag}_auc_start_std"] = float(np.std([d["A_trapz101_div100"] for d in D], ddof=1))
                r[f"{tag}_final_start_std"] = float(np.std([d["final_c100"] for d in D], ddof=1))
                r[f"{tag}_n"] = 5
            else:
                r[f"{tag}_n"] = len(cs)
        if r.get("rep_magb_n") == 5:
            r["diff_auc_rep_magb_minus_paper"] = r["rep_magb_A_trapz101_div100"] - PAPER_AUC[s][4]
            r["diff_final_rep_magb_minus_paper"] = r["rep_magb_final_c100"] - PAPER_FIN[s][4]
        if r.get("rep_lmdb_n") == 5:
            r["diff_auc_rep_lmdb_minus_paper"] = r["rep_lmdb_A_trapz101_div100"] - PAPER_AUC[s][4]
            r["diff_final_rep_lmdb_minus_paper"] = r["rep_lmdb_final_c100"] - PAPER_FIN[s][4]
        rows.append(r)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(os.path.join(OUT, "SCENE_BY_SCENE_PAPER_VS_REPRODUCED.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    s14 = [r for r in rows if r["scene"] != "sofia_church"]
    agg = {"paper_all15": {"auc": float(np.mean([PAPER_AUC[s][4] for s in ORDER])), "final": float(np.mean([PAPER_FIN[s][4] for s in ORDER]))},
           "paper_all14": {"auc": float(np.mean([r["paper_auc_magician"] for r in s14])), "final": float(np.mean([r["paper_final_magician"] for r in s14]))}}
    for tag in ("rep_magb", "rep_lmdb"):
        agg[tag] = {q: float(np.mean([r[f"{tag}_{q}"] for r in s14])) for q in ("A_trapz101_div100", "B_mean101", "C_div99", "C_div100", "final_c100", "max_c0_99", "max_c0_100")}
        d_auc = np.asarray([r[f"{tag}_A_trapz101_div100"] - r["paper_auc_magician"] for r in s14])
        d_fin = np.asarray([r[f"{tag}_final_c100"] - r["paper_final_magician"] for r in s14])
        rng = np.random.default_rng(0)
        bi = rng.integers(0, 14, (10000, 14))
        agg[tag]["diff_auc_mean"], agg[tag]["diff_auc_ci"] = float(d_auc.mean()), [float(np.quantile(d_auc[bi].mean(1), q)) for q in (0.025, 0.975)]
        agg[tag]["diff_final_mean"], agg[tag]["diff_final_ci"] = float(d_fin.mean()), [float(np.quantile(d_fin[bi].mean(1), q)) for q in (0.025, 0.975)]
        agg[tag]["scenes_rep_below_paper_auc"] = int((d_auc < 0).sum())
        agg[tag]["scenes_rep_below_paper_final"] = int((d_fin < 0).sum())
        # per-scene: is |diff| within 2 standard errors of the start mean (start std / sqrt 5)?
        agg[tag]["scenes_auc_within_2se"] = int(sum(abs(r[f"{tag}_A_trapz101_div100"] - r["paper_auc_magician"]) <= 2 * r[f"{tag}_auc_start_std"] / np.sqrt(5) for r in s14))
        agg[tag]["scenes_final_within_2se"] = int(sum(abs(r[f"{tag}_final_c100"] - r["paper_final_magician"]) <= 2 * r[f"{tag}_final_start_std"] / np.sqrt(5) for r in s14))
        agg[tag]["corr_auc_with_paper"] = float(np.corrcoef([r[f"{tag}_A_trapz101_div100"] for r in s14], [r["paper_auc_magician"] for r in s14])[0, 1])
        agg[tag]["corr_final_with_paper"] = float(np.corrcoef([r[f"{tag}_final_c100"] for r in s14], [r["paper_final_magician"] for r in s14])[0, 1])
    agg["magb_vs_lmdb"] = {"auc_mean_diff": float(np.mean([r["rep_magb_A_trapz101_div100"] - r["rep_lmdb_A_trapz101_div100"] for r in s14])),
                           "final_mean_diff": float(np.mean([r["rep_magb_final_c100"] - r["rep_lmdb_final_c100"] for r in s14]))}
    res["paper_vs_reproduced"] = agg
    json.dump(res, open(os.path.join(OUT, "results", "metric_audit.json"), "w"), indent=1, default=float)
    print(json.dumps({"formula_ok": res["formula_ok"], "registry_mismatch": len(mism), "paper_avg_check": res["paper_avg_check"], "agg": agg}, indent=1))


if __name__ == "__main__":
    main()
