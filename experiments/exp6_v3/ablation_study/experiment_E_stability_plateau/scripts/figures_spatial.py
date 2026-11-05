"""EXP6 V3 / ablation E -- spatial figures of a typical success (H0) and failure (H3 low plateau) on the same colosseum start
(CPU; read-only: e_oracle / e_predict outputs, service records, pose table). Default start 0, plateau state t = 30.
FIGURES/colosseum_spatial_s<k>.{png,pdf}:
 (a)/(b) top view: GT evaluation points covered (grey) / not covered (red) at t = 100, camera trajectory (colour = step)
 (c)/(d) top / side view of the failure at step t: GT points (light), the agent's observed S10 points (blue), the predicted
         novel cells of the executed candidate split into hit (green: within 2.0 of a true novel cell), seen (orange: near GT
         points, not novel in the true view) and empty (magenta: no GT evaluation point within 2.0), camera + legal candidates
 (e) side view: GT points covered by H0 but not by H3 at t = 100 (the plateau's missing region)
 (f) per-step predicted value of the executed action vs the GT-surface part of the true value, H3 and H0 (H0: value_cov
     recomputed by e_predict on the H0 states), with the plateau onset."""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(EX))))
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_E_stability_plateau")
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
PT = np.load(os.path.join(R, "work_dirs", "exp6_v3", "H", "pose_tables", "colosseum.npz"))
LUT = {tuple(int(x) for x in r): j for j, r in enumerate(PT["idx"])}
M20 = (1 << 20) - 1


def dec(ids):
    return np.stack([(ids >> 40) & M20, (ids >> 20) & M20, ids & M20], 1).astype(np.float64) - (1 << 19) + 0.5


def centre(k):
    return PT["C"][LUT[tuple(json.loads(k))]]


def onset(c):
    return next((t for t in range(101) if c[min(t + 20, 100)] - c[t] < 0.02), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--t", type=int, default=30)
    ap.add_argument("--fail", default="V3-H3")
    a = ap.parse_args()
    k, T = a.start, a.t
    o0, z0 = json.load(open(f"{W}/oracle/V3/colosseum_s{k}_r0.json")), np.load(f"{W}/oracle/V3/colosseum_s{k}_r0.npz")
    o3, z3 = json.load(open(f"{W}/oracle/{a.fail}/colosseum_s{k}_r0.json")), np.load(f"{W}/oracle/{a.fail}/colosseum_s{k}_r0.npz")
    gp = z0["gt_points"].astype(np.float64)
    N = len(gp)
    cov0 = np.unpackbits(z0["covered_t100"])[:N].astype(bool)
    cov3 = np.unpackbits(z3["covered_t100"])[:N].astype(bool)
    e0, e3 = json.load(open(f"{EP}/V3/colosseum_s{k}.json")), json.load(open(f"{EP}/{a.fail}/colosseum_s{k}.json"))
    C0 = np.asarray([centre(x) for x in e0["executed_keys"]])
    C3 = np.asarray([centre(x) for x in e3["executed_keys"]])
    fig, ax = plt.subplots(2, 3, figsize=(19, 12))
    fn = a.fail.replace("V3-", "")
    cat = [(cov0 & cov3, "#c8c8c8", "covered by both"), (~cov0 & ~cov3, "#efefef", "covered by neither"),
           (cov3 & ~cov0, "#1f77b4", f"covered only by {fn}"), (cov0 & ~cov3, "#d62728", f"covered only by H0 (missing in {fn})")]
    for A, C, e, nm in ((ax[0, 0], C0, e0, "H0 (success)"), (ax[0, 1], C3, e3, f"{fn} (low plateau)")):
        for m, col, lab in cat:
            A.scatter(gp[m][::3, 0], gp[m][::3, 2], s=0.5, color=col, label=f"{lab} ({m.mean():.3f})")
        A.scatter(C[:, 0], C[:, 2], c=np.arange(len(C)), cmap="viridis", s=14, zorder=3)
        A.plot(C[:, 0], C[:, 2], color="#444", lw=0.5)
        A.set_title(f"(a/b) start {k} {nm}: Final {e['coverage_official'][100]:.3f}\nGT points at t=100 by who covered them; dots = camera (colour = step)", fontsize=9)
        A.set_xlim(-70, 70); A.set_ylim(-70, 70); A.set_aspect("equal"); A.set_xlabel("x"); A.set_ylabel("z")
        A.legend(fontsize=7, markerscale=8, loc="lower left")
    # state t of the failure
    pz = np.load(f"{W}/oracle/{a.fail}/colosseum_s{k}_r0_predict_cells.npz")
    idx = {(int(p), int(q)): (int(r), int(s)) for p, q, r, s in pz["index"]}
    row = next(r for r in o3["steps"] if r["t"] == T)
    ex = row["executed"]
    off, n = idx[(T, ex)]
    cells = dec(pz["ids"][off:off + n])
    c = row["cands"][ex]
    true_ids = z3["cell_ids"][c["cell_off"]:c["cell_off"] + c["gt_cells"]]
    tt = cKDTree(dec(true_ids)) if len(true_ids) else None
    hit = (tt.query(cells, k=1, distance_upper_bound=2.0)[0] < np.inf) if tt is not None else np.zeros(len(cells), bool)
    surf = cKDTree(gp).query(cells, k=1, distance_upper_bound=2.0)[0] < np.inf
    ep = json.load(open(os.path.join(R, e3["episode_file"])))
    svc = os.path.join(os.path.dirname(os.path.dirname(os.path.join(R, e3["episode_file"]))), "svc", ep["tag"])
    obs = np.concatenate([np.load(f"{svc}/svc_colosseum_s{k}_t{t}.npz")["sample_points"] for t in range(T + 1)])
    predv = next(json.loads(l) for l in open(f"{W}/oracle/{a.fail}/colosseum_s{k}_r0_predict.jsonl") if json.loads(l)["t"] == T)["cands"][ex]["pred_value"]
    cam = centre(row["pose_key"])
    cands = np.asarray([centre(x["key"]) for x in row["cands"]])
    for A, (i, j), lab in ((ax[0, 2], (0, 2), "top view x-z"), (ax[1, 0], (2, 1), "side view z-y")):
        A.scatter(gp[::6, i], gp[::6, j], s=0.3, color="#dddddd")
        A.scatter(obs[::3, i], obs[::3, j], s=0.3, color="#4a90d9", alpha=0.5, label="observed S10 points (t<=%d)" % T)
        for m, col, nm in ((hit, "#2ca02c", "pred novel: hit"), (~hit & surf, "#ff7f0e", "pred novel: seen"), (~surf, "#d62ad6", "pred novel: empty")):
            A.scatter(cells[m, i], cells[m, j], s=3, color=col, label=f"{nm} ({m.sum()})")
        A.scatter(cands[:, i], cands[:, j], marker="^", s=40, color="k")
        A.scatter([cam[i]], [cam[j]], marker="*", s=200, color="gold", edgecolor="k")
        A.set_title(f"(c/d) {a.fail.replace('V3-', '')} s{k} at t={T}: predicted novel cells of the executed view, {lab}\n"
                    f"predicted value {predv:.0f}, true value (GT-surface part) {c.get('gt_value_rel', float('nan')):.0f}; true one-step gain {c['dcov_pp']:.2f} pp", fontsize=9)
        A.legend(fontsize=7, markerscale=4, loc="lower left", title="star = camera, triangles = legal candidates", title_fontsize=7)
        A.set_aspect("equal")
    ax[0, 2].set_xlim(-70, 70); ax[0, 2].set_ylim(-70, 70)
    ax[1, 0].set_xlim(-70, 70); ax[1, 0].set_ylim(-15, 40); ax[1, 0].set_xlabel("z"); ax[1, 0].set_ylabel("y (up)")
    miss = cov0 & ~cov3
    A = ax[1, 1]
    A.scatter(gp[::6, 2], gp[::6, 1], s=0.3, color="#dddddd")
    A.scatter(gp[miss][::2, 2], gp[miss][::2, 1], s=0.6, color="#d62728", label=f"covered by H0, not by {a.fail.replace('V3-', '')} ({miss.mean():.3f} of GT)")
    A.scatter(C3[:, 2], C3[:, 1], s=10, color="#e67e22", label=f"{a.fail.replace('V3-', '')} cameras")
    A.scatter(C0[:, 2], C0[:, 1], s=10, color="#555", marker="x", label="H0 cameras")
    A.set_title("(e) side view z-y: missing region of the plateau", fontsize=9)
    A.set_xlim(-70, 70); A.set_ylim(-15, 40); A.set_aspect("equal"); A.set_xlabel("z"); A.set_ylabel("y (up)"); A.legend(fontsize=7, markerscale=1.5)
    A = ax[1, 2]
    for mid, col, nm in (("V3", "#555", "H0"), (a.fail, "#e67e22", a.fail.replace("V3-", ""))):
        pr = [json.loads(l) for l in open(f"{W}/oracle/{mid}/colosseum_s{k}_r0_predict.jsonl")]
        ts = [r["t"] for r in pr]
        A.plot(ts, [r["cands"][r["executed"]]["pred_value"] for r in pr], color=col, lw=1.4, label=f"{nm}: predicted value of executed view")
        orr = {r["t"]: r for r in (o0 if mid == "V3" else o3)["steps"]}
        A.plot(ts, [orr[t]["cands"][orr[t]["executed"]].get("gt_value_rel", np.nan) for t in ts], color=col, lw=1.0, ls="--", label=f"{nm}: true value (GT-surface part)")
        on = onset((e0 if mid == "V3" else e3)["coverage_official"])
        A.axvline(on, color=col, lw=0.6, ls=":")
    A.set_xlabel("step"); A.set_ylabel("value (weighted novel cells)"); A.set_title("(f) predicted vs true value of the executed view; dotted = plateau onset", fontsize=9)
    A.legend(fontsize=7); A.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(EX, "FIGURES", f"colosseum_spatial_s{k}.{ext}"), dpi=140)
    print("ok", k, T)


if __name__ == "__main__":
    main()
