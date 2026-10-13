"""Aggregate NBV episode logs (exp1-task7 onwards).

For every <planner>_N<N> directory under --root and every T in --T:
- AUC_T = (1/T) * sum_{t=1..T} (c_{t-1} + c_t) / 2, with c the surface coverage (step 0 = initial obs);
- final coverage c_T, and the steps needed to reach 80% coverage (T+1 if never reached);
- remaining gain c_Tmax - c_T, which measures saturation;
- decision latency: mean seconds per step to generate and score the candidates (execution excluded).
It also reports paired bootstrap 95% CIs (10k resamples, episodes paired by scene) for the
differences between planners with the same N, e.g. oracle - random.

Usage: python -m seen2scene.tools.nbv_eval --root work_dirs/exp1_infra/task7_calibration --T 5 10 20
"""
import argparse
import glob
import json
import os

import numpy as np


def load(root):
    runs, latency = {}, {}
    for d in sorted(glob.glob(os.path.join(root, "*_N*"))):
        eps, lat = {}, []
        for f in glob.glob(os.path.join(d, "*.json")):
            j = json.load(open(f))
            eps[j["scene"]] = np.array([s["surface_coverage"] for s in j["log"]])
            lat += [s["t_candidates"] + s["t_scoring"] for s in j["log"][1:] if "t_scoring" in s]
        runs[os.path.basename(d)] = eps
        latency[os.path.basename(d)] = float(np.mean(lat)) if lat else float("nan")
    return runs, latency


def metrics(c, T):
    c = c[: T + 1]
    auc = float(np.mean((c[:-1] + c[1:]) / 2))
    reach = np.nonzero(c[1:] >= 0.8)[0]
    return {"auc": auc, "final": float(c[-1]), "steps_to_80": float(reach[0] + 1 if len(reach) else T + 1)}


def boot(x, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--T", nargs="+", type=int, default=[5, 10, 20])
    args = ap.parse_args()
    runs, latency = load(args.root)
    tmax = max(args.T)
    summary = {}
    for name, eps in runs.items():
        for T in args.T:
            ms = [metrics(c, T) for c in eps.values() if len(c) > T]
            rem = [float(c[tmax] - c[T]) for c in eps.values() if len(c) > tmax]
            summary[f"{name}/T{T}"] = {"n": len(ms), **{k: boot(np.array([m[k] for m in ms])) for k in ms[0]},
                                       "remaining_gain_to_T%d" % tmax: boot(np.array(rem)) if rem else None,
                                       "decision_latency_s": latency[name]}
    # paired differences between planners with the same N
    Ns = sorted({k.split("_N")[1] for k in runs})
    for n in Ns:
        names = [k for k in runs if k.endswith(f"_N{n}")]
        for a in names:
            for b in names:
                if a >= b:
                    continue
                common = sorted(set(runs[a]) & set(runs[b]))
                for T in args.T:
                    for key in ("auc", "final"):
                        d = np.array([metrics(runs[a][s], T)[key] - metrics(runs[b][s], T)[key] for s in common])
                        summary[f"diff/{a}-{b}/T{T}/{key}"] = boot(d)
    curves = {name: np.mean(np.stack([c for c in eps.values() if len(c) == tmax + 1]), 0).round(4).tolist()
              for name, eps in runs.items()}
    json.dump({"summary": summary, "mean_curves": curves}, open(os.path.join(args.root, "summary.json"), "w"), indent=1)
    for k, v in summary.items():
        if k.startswith("diff/"):
            print(f"{k:45s} mean {v[0]:+.4f}  95% CI [{v[1]:+.4f}, {v[2]:+.4f}]")
        else:
            print(f"{k:22s} n={v['n']:3d}  AUC {v['auc'][0]:.4f} [{v['auc'][1]:.4f},{v['auc'][2]:.4f}]  final {v['final'][0]:.4f}  "
                  f"steps->80% {v['steps_to_80'][0]:.2f}  remaining->T{tmax} {v['remaining_gain_to_T%d' % tmax][0] if v['remaining_gain_to_T%d' % tmax] else float('nan'):.4f}  "
                  f"latency {v['decision_latency_s']:.2f}s")
    for name, c in curves.items():
        print(f"curve {name}: {c}")


if __name__ == "__main__":
    main()
