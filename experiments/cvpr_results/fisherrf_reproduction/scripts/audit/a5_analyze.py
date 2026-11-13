"""FisherRF audit A5 (CPU): analysis of the A2 planner-side probes + A3 GT gains (map health, scorer validity at fixed states).

Scorer validity per state and candidate set (legal one-step neighbours = D3a decision set; uniform whole-grid poses = f1c4
candidate distribution; all): Spearman rho(score, true gain_mono); Top-1 gain of argmax score vs random pick (mean gain) vs
oracle (max gain), in coverage percentage points; 'captured' = (Top-1 - random) / (oracle - random).
Scorers: eig (official pose_eval, lambda 0.1), eig_clean (phantoms removed), unk_true_fov (never-observed voxels in FOV, a
MAGICIAN-style frontier score), unk_plan_fov (planner 'unknown' labels in FOV, the f1c4 filter quantity), novelty (rendered
silhouette < 0.5 fraction), and random.
Usage: python a5_analyze.py --probe DIR   -> DIR/a5_summary.json + printed tables
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

SCORERS = ["eig", "eig_clean", "unk_true_fov", "unk_plan_fov", "novelty"]
EXTRA = ["eig_lambda1e-6", "eig_no_train", "eig_clean_nobig"]                                  # present in the H-statistics rerun only
COMBO_K = (5, 10, 20)                                                       # 'frontier first, then EIG': argmax EIG among top-K by unk_true_fov


def top1(score, gain, rng):
    s = np.asarray(score, float)
    best = np.flatnonzero(s == s.max())
    return float(np.mean(gain[best]))                                   # ties: expected gain of a uniform tie-break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    a = ap.parse_args()
    rows, health, eigprops = [], [], []
    for f in sorted(glob.glob(os.path.join(a.probe, "*_gt.json"))):
        tag = os.path.basename(f)[:-8]
        P = json.load(open(os.path.join(a.probe, f"{tag}.json")))
        Gt = json.load(open(f))
        gts = {st["t"]: st for st in Gt["states"]}
        for st in P["states"]:
            t = st["t"]
            g = {c["key"]: c for c in gts[t]["candidates"]}
            C = st["candidates"]
            re = [x for x in st["render_executed"] if "psnr" in x]
            old = [x for x in re if x["t"] < t]
            cur = [x for x in re if x["t"] == t]
            pn = st.get("predict_next", {})
            occ = st["occ"]
            health.append({"ep": tag, "t": t, "n_gauss": st["n_gauss"], "n_phantom": st["n_phantom_neginf_scale"],
                           "frac_in_box": st["frac_means_in_scene_box"],
                           "psnr_cur": cur[0]["psnr"] if cur else None,
                           "psnr_old_mean": float(np.mean([x["psnr"] for x in old])) if old else None,
                           "depth_rel_old_median": float(np.median([x["depth_rel_median"] for x in old if "depth_rel_median" in x])) if old else None,
                           "depth_l1_old_rt": float(np.mean([x["depth_l1_rt"] for x in old if "depth_l1_rt" in x])) if old else None,
                           "sil_cov_old": float(np.mean([x["sil_cov_valid"] for x in old])) if old else None,
                           "floater_old": float(np.mean([x["floater_frac_invalid"] for x in old if "floater_frac_invalid" in x])) if old else None,
                           "pred_sil_cov": pn.get("sil_cov_valid"), "pred_depth_rel_median": pn.get("depth_rel_median"), "pred_psnr": pn.get("psnr"),
                           "n_frames_rendered": len(re),
                           "unknown_observed_frac": occ["n_unknown_but_observed"] / max(occ["n_unknown"], 1),
                           "observed_once_frac": occ["n_observed_once"] / max(occ["n_observed"], 1),
                           "observed_vox_labelled_unknown_frac": occ["n_unknown_but_observed"] / max(occ["n_observed"], 1)})
            if not C or C[0].get("eig") is None:
                continue
            gain = np.array([g[c["key"]]["gain_mono"] for c in C]) * 100.0          # percentage points
            occf = np.array([g[c["key"]]["occupied"] for c in C])
            grp = np.array([c["group"] for c in C])
            eig = np.array([c["eig"] for c in C])
            for c in C:
                eigprops.append({"ep": tag, "t": t, "group": c["group"], "eig": c["eig"], "gain_pp": 100 * g[c["key"]]["gain_mono"],
                                 "eig_phantom_share": c["eig_phantom"] / c["eig"] if c["eig"] > 0 else 0.0,
                                 "eig_unseen_share": c["eig_unseen"] / c["eig"] if c["eig"] > 0 else 0.0,
                                 "n_vis_gauss": c["n_vis_gauss"], "novelty": c["novelty"], "occupied": bool(g[c["key"]]["occupied"])})
            for subset in ("legal", "uniform", "all"):
                m = np.ones(len(C), bool) if subset == "all" else (grp == subset)
                m &= ~occf if subset == "uniform" else m                            # far poses inside the mesh are unreachable
                if m.sum() < 3 or np.ptp(gain[m]) <= 0:
                    continue
                gm = gain[m]
                r = {"ep": tag, "t": t, "subset": subset, "n": int(m.sum()), "oracle": float(gm.max()), "random": float(gm.mean()),
                     "executed_gain": float(gain[[c["executed_next"] for c in C]].mean()) if any(c["executed_next"] for c in C) and subset == "legal" else None}
                for sname in SCORERS + [x for x in EXTRA if x in C[0]]:
                    sv = np.array([c[sname] for c in C], float)[m]
                    r[f"top1_{sname}"] = top1(sv, gm, None)
                    r[f"rho_{sname}"] = float(spearmanr(sv, gm).correlation) if np.ptp(sv) > 0 else 0.0
                uk = np.array([c["unk_true_fov"] for c in C], float)[m]
                ev = eig[m]
                order = np.argsort(-uk, kind="stable")
                for K in COMBO_K:
                    top = order[:K]
                    r[f"top1_frontier{K}_then_eig"] = float(gm[top[np.argmax(ev[top])]])
                    r[f"top1_frontier{K}_random"] = float(gm[top].mean())
                # phantom share of the EIG of the argmax-EIG candidate (the decision)
                ci = np.flatnonzero(m)[np.argmax(ev)]
                r["phantom_share_of_decision"] = C[ci]["eig_phantom"] / C[ci]["eig"] if C[ci]["eig"] > 0 else 0.0
                if "eig_big" in C[ci]:
                    r["big_share_of_decision"] = C[ci]["eig_big"] / C[ci]["eig"] if C[ci]["eig"] > 0 else 0.0
                # fraction of candidates with EIG ~ 0 relative to the per-state max
                r["frac_eig_zero"] = float((ev <= 1e-3 * max(ev.max(), 1e-12)).mean())
                rows.append(r)
    out = {"health": health, "scorer_rows": rows}
    # ---- summaries
    print("=== map health (per episode, mean over probe states) ===")
    eps = sorted({h["ep"] for h in health})
    for e in eps:
        H = [h for h in health if h["ep"] == e]
        f = lambda k: np.nanmean([h[k] for h in H if h[k] is not None]) if any(h[k] is not None for h in H) else float("nan")  # noqa
        print("%-18s gauss %7.0f phantom %5.0f inbox %.3f | PSNR cur %.1f old %.1f | depth rel med old %.4f l1 %.3f rt | sil cov old %.3f floater %.3f | "
              "pred: sil %.3f rel %.4f | unk-observed %.3f obs-once %.3f obs->unknown %.3f" % (
                  e, f("n_gauss"), f("n_phantom"), f("frac_in_box"), f("psnr_cur"), f("psnr_old_mean"), f("depth_rel_old_median"),
                  f("depth_l1_old_rt"), f("sil_cov_old"), f("floater_old"), f("pred_sil_cov"), f("pred_depth_rel_median"),
                  f("unknown_observed_frac"), f("observed_once_frac"), f("observed_vox_labelled_unknown_frac")))
    print("\n=== scorer validity (mean over states; gains in coverage pp; captured = (top1-random)/(oracle-random)) ===")
    summ = {}
    for subset in ("legal", "uniform", "all"):
        R = [r for r in rows if r["subset"] == subset]
        if not R:
            continue
        line = {"n_states": len(R), "oracle": float(np.mean([r["oracle"] for r in R])), "random": float(np.mean([r["random"] for r in R]))}
        print(f"[{subset}] states {len(R)}  oracle {line['oracle']:.2f}  random {line['random']:.2f}" +
              (f"  executed(f1c4) {np.mean([r['executed_gain'] for r in R if r['executed_gain'] is not None]):.2f}" if subset == "legal" else ""))
        if "big_share_of_decision" in R[0]:
            print(f"   big-Gaussian (scale > 3.5 rt) share of the argmax-EIG candidate: median {np.median([r['big_share_of_decision'] for r in R]):.2f}, "
                  f"states > 0.5: {np.mean([r['big_share_of_decision'] > 0.5 for r in R]):.2f}")
        names = SCORERS + [x for x in EXTRA if f"top1_{x}" in R[0]] + [f"frontier{K}_then_eig" for K in COMBO_K] + [f"frontier{K}_random" for K in COMBO_K]
        print(f"   EIG~0 (per-state) fraction {np.mean([r['frac_eig_zero'] for r in R]):.2f}; phantom share of the argmax-EIG candidate: "
              f"median {np.median([r['phantom_share_of_decision'] for r in R]):.2f}, states > 0.5: {np.mean([r['phantom_share_of_decision'] > 0.5 for r in R]):.2f}")
        for sname in names:
            t1 = np.array([r[f"top1_{sname}"] for r in R])
            rnd = np.array([r["random"] for r in R])
            orc = np.array([r["oracle"] for r in R])
            cap = (t1 - rnd) / np.maximum(orc - rnd, 1e-9)
            rho = np.array([r.get(f"rho_{sname}", np.nan) for r in R])
            # paired bootstrap over states of top1 - random
            rng = np.random.default_rng(0)
            d = t1 - rnd
            bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(5000)]
            line[sname] = {"top1": float(t1.mean()), "top1_minus_random": float(d.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                           "captured_mean": float(cap.mean()), "rho_mean": float(np.nanmean(rho)), "rho_median": float(np.nanmedian(rho)),
                           "frac_states_top1_ge_random": float((t1 >= rnd).mean())}
            print(f"   {sname:22s} top1 {t1.mean():6.2f}  (−random {d.mean():+6.2f} [{np.quantile(bs, 0.025):+.2f}, {np.quantile(bs, 0.975):+.2f}])  "
                  f"captured {cap.mean():+.2f}  rho mean {np.nanmean(rho):+.2f} median {np.nanmedian(rho):+.2f}  states top1>=random {(t1 >= rnd).mean():.2f}")
        summ[subset] = line
    out["scorer_summary"] = summ
    # ---- EIG properties
    E = eigprops
    for grp in ("legal", "uniform"):
        X = [x for x in E if x["group"] == grp]
        eig = np.array([x["eig"] for x in X])
        gain = np.array([x["gain_pp"] for x in X])
        nv = np.array([x["n_vis_gauss"] for x in X])
        nov = np.array([x["novelty"] for x in X])
        z = eig <= 1e-3 * np.maximum(eig.max(), 1e-9)
        print(f"\n[{grp}] candidates {len(X)}: EIG ~ 0 for {z.mean():.2f}; mean gain of EIG~0 candidates {gain[z].mean() if z.any() else float('nan'):.2f} pp vs others {gain[~z].mean():.2f} pp; "
              f"rho(EIG, n_visible_gauss) {spearmanr(eig, nv).correlation:+.2f}; rho(EIG, novelty) {spearmanr(eig, nov).correlation:+.2f}; "
              f"rho(EIG, gain) pooled {spearmanr(eig, gain).correlation:+.2f}; rho(novelty, gain) pooled {spearmanr(nov, gain).correlation:+.2f}")
        ps = np.array([x["eig_phantom_share"] for x in X])
        us = np.array([x["eig_unseen_share"] for x in X])
        print(f"   phantom share of EIG: median {np.median(ps):.3f}, p90 {np.quantile(ps, 0.9):.3f}, max {ps.max():.3f}; "
              f"share from Gaussians never seen by keyframes: median {np.median(us):.3f}, p90 {np.quantile(us, 0.9):.3f}")
    out["eig_props"] = eigprops
    json.dump(out, open(os.path.join(a.probe, "a5_summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
