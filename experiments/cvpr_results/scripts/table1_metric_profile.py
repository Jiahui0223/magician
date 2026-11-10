"""Table 1 metric profile (CPU): for every horizon T = 1..100, scene-equal AUC@T and Cov@T of every complete Table 1 method and the
scene-paired difference Ours - baseline (scene bootstrap 95 % CI, 10,000 draws, seed 0; wins / losses over the 14 scenes).
Reads manifest/episodes.csv only. Output: fig/table1/metric_profile.csv + metric_profile.{pdf,png}.
Purpose: show the full profile so that the table columns can be justified; it is NOT used to pick the horizon where Ours looks best."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402
import render as RD  # noqa: E402

plt = RD.plt
OURS, BASE = "V3-T0-D1", ["MAG-B", "MACARONS", "RANDOM"]
OUT = os.path.join(L.FIG, "table1")


def per_scene(m):
    E = L.load_manifest(m)
    C = {s: np.mean([E[(s, k)]["coverage_official"] for k in range(5)], 0) for s in L.SCENE_ORDER}
    A = {s: np.array([L.auc(C[s], T) for T in range(1, 101)]) for s in L.SCENE_ORDER}         # [100] AUC@1..100
    return np.stack([A[s] for s in L.SCENE_ORDER]), np.stack([C[s][1:] for s in L.SCENE_ORDER])   # [14, 100] each


rng = np.random.default_rng(0)
idx = rng.integers(0, 14, (10000, 14))
P = {m: per_scene(m) for m in [OURS] + BASE}
rows = []
prof = {}
for b in BASE:
    for k, name in ((0, "AUC"), (1, "Cov")):
        d = P[OURS][k] - P[b][k]                                   # [14, 100]
        mean = d.mean(0); bs = d[idx].mean(1)
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        wins, losses = (d > 0).sum(0), (d < 0).sum(0)
        rel = mean / np.maximum(P[b][k].mean(0), 1e-9)
        prof[(b, name)] = (mean, lo, hi, wins, rel)
        for T in range(1, 101):
            rows.append([b, f"{name}@T", T, repr(float(P[OURS][k].mean(0)[T - 1])), repr(float(P[b][k].mean(0)[T - 1])), repr(float(mean[T - 1])), repr(float(lo[T - 1])),
                         repr(float(hi[T - 1])), int(wins[T - 1]), int(losses[T - 1]), repr(float(rel[T - 1]))])
L.write_csv(os.path.join(OUT, "metric_profile.csv"), ["baseline", "metric", "T", "ours", "baseline_value", "diff", "ci95_lo", "ci95_hi", "scene_wins", "scene_losses", "relative_diff"], rows)
for b in BASE:
    for name in ("AUC", "Cov"):
        mean, lo, hi, wins, rel = prof[(b, name)]
        T = np.arange(1, 101)
        sig = T[lo > 0]
        print(f"Ours - {b:8s} {name}@T: max diff {mean.max():+.3f} at T={int(T[mean.argmax()])} (CI [{lo[mean.argmax()]:+.3f}, {hi[mean.argmax()]:+.3f}], wins {wins[mean.argmax()]}/14) | "
              f"max relative {100 * rel.max():+.0f}% at T={int(T[rel.argmax()])} | CI > 0 for T in {('%d..%d' % (sig.min(), sig.max())) if len(sig) else 'none'} ({len(sig)} of 100) | "
              + " ".join(f"T{t}:{mean[t - 1]:+.3f}({wins[t - 1]})" for t in (10, 20, 30, 50, 75, 100)))
# figure: paired difference vs horizon
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.28})
for ax, name, ttl in ((axs[0], "AUC", "(a) $\\Delta$AUC@$T$ (Ours $-$ baseline)"), (axs[1], "Cov", "(b) $\\Delta$Coverage@$T$ (Ours $-$ baseline)")):
    for b in ("MAG-B", "MACARONS"):
        mean, lo, hi, _, _ = prof[(b, name)]
        st = RD.STYLE[b]
        ax.fill_between(np.arange(1, 101), lo, hi, color=st["color"], alpha=0.15, lw=0)
        ax.plot(np.arange(1, 101), mean, color=st["color"], ls=st["ls"], lw=1.5, label=f"vs {st['label']}")
    ax.axhline(0, color="k", lw=0.6)
    for t in (20, 50, 100):
        ax.axvline(t, color="#BBBBBB", lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(1, 100); ax.set_xlabel("Horizon $T$ (steps)"); ax.set_title(ttl, fontsize=9); ax.grid(color="#E3E3E3", lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel("Scene-paired difference")
axs[1].legend(fontsize=7.5, loc="upper right")
RD.save(fig, os.path.join(OUT, "metric_profile"))
