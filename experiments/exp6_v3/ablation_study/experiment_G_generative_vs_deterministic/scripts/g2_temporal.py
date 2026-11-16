"""Experiment G2 item 3 -- closed-loop temporal analysis from the existing replay-verified official coverage curves (protocol G2-v1 section 3;
CPU only). D0 = closed_loop/D0 (g_collect.load_new: replay == stored, 101 values), G1 = V3-T0-K1D2, G4 = V3-T0-D1 (cvpr_lib.load_manifest).
Per step t = 0..100: scene-equal mean of the per-scene (5-start mean) difference D0 - G4 and D0 - G1, pointwise 95 % band from the scene
bootstrap (10,000, seed 0); crossover = smallest t with mean diff <= 0 for every t' >= t (None if never), plus the last sign change;
late gain = C[100] - C[50] per episode -> scene mean -> paired difference with CI. Groups All-14, Dev+Test-8, Unseen-6.
Writes results/g2/temporal/{temporal_curves.csv, temporal_summary.csv, temporal_summary.md} and figures/g2/temporal_diff.{pdf,png}."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g_collect as GC  # noqa: E402

L = GC.L
GD = GC.GD
OUT = os.path.join(GD, "results/g2/temporal")
FIG = os.path.join(GD, "figures/g2")
GROUPS = {"All-14": list(L.SCENE_ORDER), "Dev+Test-8": [s for s in L.SCENE_ORDER if L.SPLIT[s] != "train"],
          "Unseen-6": ["alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"]}


def scene_curves(E):
    return {s: np.mean([E[(s, k)]["coverage_official"] for k in range(L.N_START)], 0) for s in L.SCENE_ORDER if all((s, k) in E for k in range(L.N_START))}


def boot_rows(D, n=10000, seed=0):
    """D [S, T] per-scene differences -> mean [T], lo [T], hi [T] (scene bootstrap, same resampling indices for every t)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, D.shape[0], (n, D.shape[0]))
    B = D[idx].mean(1)
    return D.mean(0), np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)


def main():
    E = {"D0": GC.load_new("D0")[0], "G1": L.load_manifest("V3-T0-K1D2"), "G4": L.load_manifest("V3-T0-D1")}
    Cs = {k: scene_curves(v) for k, v in E.items()}
    late = {k: {s: float(np.mean([E[k][(s, i)]["coverage_official"][100] - E[k][(s, i)]["coverage_official"][50] for i in range(L.N_START)]))
                for s in Cs[k]} for k in E}
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    curve_rows, summ, md = [], [], ["# G2 item 3 — closed-loop temporal analysis (protocol G2-v1 §3)", "",
                                    "ΔC(t) = scene-equal mean coverage difference (official), 95 % pointwise band = scene bootstrap (10,000, seed 0).", ""]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for gi, (g, sc) in enumerate(GROUPS.items()):
        md += [f"## {g} ({len(sc)} scenes)", "", "| Pair | crossover t (ΔC ≤ 0 from t on) | last sign change | ΔC@20 | ΔC@50 | ΔC@100 | max ΔC (t) | min ΔC (t) | late gain C[100]−C[50] diff [CI] |",
               "|---|---|---|---|---|---|---|---|---|"]
        for ref, col in (("G4", "#c0392b"), ("G1", "#2471a3")):
            ss = [s for s in sc if s in Cs["D0"] and s in Cs[ref]]
            Dm = np.stack([Cs["D0"][s] - Cs[ref][s] for s in ss])
            m, lo, hi = boot_rows(Dm)
            for t in range(101):
                curve_rows.append([g, f"D0-{ref}", t, m[t], lo[t], hi[t], len(ss)])
            cross = next((t for t in range(101) if np.all(m[t:] <= 0)), None)
            sgn = np.sign(m)
            ch = [t for t in range(1, 101) if sgn[t] != 0 and sgn[t - 1] != 0 and sgn[t] != sgn[t - 1]]
            ld = [late["D0"][s] - late[ref][s] for s in ss]
            lm_, llo, lhi = L.boot_ci(ld)
            row = {"group": g, "pair": f"D0-{ref}", "n_scenes": len(ss), "crossover_t": cross, "last_sign_change_t": ch[-1] if ch else None,
                   "dC@20": m[20], "dC@50": m[50], "dC@100": m[100], "max_dC": float(m.max()), "argmax_t": int(m.argmax()), "min_dC": float(m.min()), "argmin_t": int(m.argmin()),
                   "late_gain_D0": float(np.mean([late["D0"][s] for s in ss])), f"late_gain_{ref}": float(np.mean([late[ref][s] for s in ss])),
                   "late_gain_diff": lm_, "late_gain_ci_lo": llo, "late_gain_ci_hi": lhi, "band_excludes0_steps": int(((lo > 0) | (hi < 0)).sum()),
                   "band_above0_steps": int((lo > 0).sum()), "band_below0_steps": int((hi < 0).sum())}
            summ.append(row)
            md.append(f"| D0 − {ref} | {cross} | {row['last_sign_change_t']} | {m[20]:+.4f} | {m[50]:+.4f} | {m[100]:+.4f} | {m.max():+.4f} ({m.argmax()}) | "
                      f"{m.min():+.4f} ({m.argmin()}) | {lm_:+.4f} [{llo:+.4f}, {lhi:+.4f}] |")
            ax = axes[gi]
            ax.fill_between(range(101), lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(range(101), m, color=col, lw=1.6, label=f"D0 − {ref}")
        md += ["", f"Steps with the band above 0 / below 0 (D0 − G4): {summ[-2]['band_above0_steps']} / {summ[-2]['band_below0_steps']}; "
               f"(D0 − G1): {summ[-1]['band_above0_steps']} / {summ[-1]['band_below0_steps']}.", ""]
        ax = axes[gi]
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(f"{g}", fontsize=10)
        ax.set_xlabel("step")
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("coverage difference (D0 − G)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"temporal_diff.{ext}"), dpi=200)
    L.write_csv(os.path.join(OUT, "temporal_curves.csv"), ["group", "pair", "t", "mean_diff", "ci_lo", "ci_hi", "n_scenes"], curve_rows)
    L.write_csv(os.path.join(OUT, "temporal_summary.csv"), list(summ[0]), [list(r.values()) for r in summ])
    open(os.path.join(OUT, "temporal_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
