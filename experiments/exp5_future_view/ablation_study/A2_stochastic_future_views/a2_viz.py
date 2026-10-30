"""EXP5 Ablation A2 figures (seen2scene env, CPU). Real model outputs only (v2.npz / cases/*.npz), real GT (t*).

Per case of results/cases.json and model (V2M, V2O), candidates = the model's pick, the oracle best and (if different)
the GTD pick:
  depth_<dataset>_<case>_<model>.png  2D: GT future depth and the 4 decoded samples on ONE depth scale; colour = depth of
                                       DistP1-new hits, light gray = hit on known surface / projection outcome, dark gray =
                                       UNKNOWN hit inside the dilated BAND (not new), white = no hit; + agreement map
  3d_<dataset>_<case>_<model>.png     3D: observed points (gray) + GT new surface (black) / each sample's new surface
                                       (orange) / union coloured by the number of samples within 2 voxel radii
  orbit_<dataset>_<case>_V2M.gif       short orbit of the union panel (success and stochastic cases)
  fig_quality_diversity.png            B1 summary: per-sample quality vs diversity, sample influence, K ablation
Annotations -> figures/annotations.json
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ABL, "common"))
import ab_common as A  # noqa: E402
import ab_style as ST  # noqa: E402

import numpy as np  # noqa: E402

plt = ST.plt
FIG, RES = os.path.join(HERE, "figures"), os.path.join(HERE, "results")
GH, GW = 64, 114
UNIT = {"indoor": 7.0, "outdoor": 1.0}
UNAME = {"indoor": "m", "outdoor": "scene units"}
ANN = {}


def cat_image(t, new, t_known, hit_u, valid, scale):
    """RGBA image of one depth hypothesis: depth colour on new hits, gray elsewhere."""
    img = np.ones((len(t), 4))
    fin = np.isfinite(t)
    known = fin & ~hit_u
    img[known] = plt.matplotlib.colors.to_rgba(ST.KNOWN_GRAY)
    img[fin & hit_u & ~new] = plt.matplotlib.colors.to_rgba(ST.UNK_NOTNEW)
    img[new] = ST.SEQ(np.clip((t[new] - scale[0]) / (scale[1] - scale[0]), 0, 1))
    img[~valid] = plt.matplotlib.colors.to_rgba("#f6d5d5")
    return img.reshape(GH, GW, 4)


def load_case(dataset, sid):
    d = A.state_dir(dataset, sid)
    return (np.load(os.path.join(d, "v2.npz")), json.load(open(os.path.join(d, "v2.json"))), np.load(os.path.join(d, "oracle.npz")),
            np.load(os.path.join(A.W, "cases", f"{dataset}_{sid}.npz")))


def picks_of(dataset, sid):
    import ab_load as AL
    st = next(s for s in A.load_manifest(dataset)["states"] if s["sid"] == sid)
    d = A.state_dir(dataset, sid)
    r = dict(st)
    v = json.load(open(os.path.join(d, "v2.json")))
    rec = {k: np.asarray(x) for k, x in np.load(os.path.join(A.REPO, st["sources"][0]["svc"])).items()
           if k in ("legal_idx", "executed_pose_idx_history")}
    sc = {m: np.asarray(v["models"][m]["distp1"]) for m in ("V2O", "V2M")}
    sc["GTD"] = np.asarray(v["gtd_distp1"])
    return {m: AL.pick_of(dataset, r, sc[m], rec, m)[0] for m in sc}, st


def fig_depth(dataset, nm, sid, model):
    z, v2, orc, cz = load_case(dataset, sid)
    pk, st = picks_of(dataset, sid)
    C = np.asarray(st["cand"])
    g = orc["gain"] * 100
    best = int(C[np.argmax(g[C])])
    rows = [(pk[model], f"{model} pick")]
    if best != pk[model]:
        rows.append((best, "oracle best"))
    if pk["GTD"] not in (pk[model], best):
        rows.append((pk["GTD"], "GTD pick"))
    u = UNIT[dataset]
    k_M = float(z["k_M"])
    depths = []
    for j, _ in rows:
        ts = orc["tstar"][j]
        depths.append(ts[cz[f"gt_new_c{j}"]] * k_M / u)
        T = z[f"{model}_t"][j]
        depths.append((T[cz[f"{model}_new"][j]] * k_M / u))
    allv = np.concatenate([x for x in depths if len(x)]) if any(len(x) for x in depths) else np.asarray([0.0, 1.0])
    scale = (float(np.quantile(allv, 0.02)), float(np.quantile(allv, 0.98)) + 1e-6)
    fig, axs = plt.subplots(len(rows), 6, figsize=(17, 2.15 * len(rows) + 1.5), squeeze=False, layout="constrained")
    sk = np.asarray(v2["models"][model]["distp1_k"])
    for i, (j, lab) in enumerate(rows):
        ts = orc["tstar"][j]
        valid = z["valid"][j].astype(bool)
        tk = z["t_known"][j]
        gt_new = cz[f"gt_new_c{j}"]
        cat = z["lab_cat"][j]                       # futureview.geometry.CAT: 1 NOHIT (incl. beyond tmax), 2 KNOWN, 3 HIT
        tt = np.where(np.isin(cat, (2, 3, 4, 5)), ts, np.inf)
        gt_hit_u = cat == 3
        axs[i, 0].imshow(cat_image(tt * k_M / u, gt_new, tk, gt_hit_u, valid, scale), interpolation="nearest")
        axs[i, 0].set_title(f"GT future depth\nGTD = {v2['gtd_distp1'][j]:.1f}, gain = {g[j]:.3f} pp", loc="left", fontsize=8.5)
        new = cz[f"{model}_new"][j]
        agree = new.sum(0)
        for k in range(4):
            T = z[f"{model}_t"][j][k] * k_M / u
            axs[i, 1 + k].imshow(cat_image(T, new[k], tk, z[f"{model}_hit_u"][j][k].astype(bool), valid, scale), interpolation="nearest")
            axs[i, 1 + k].set_title(f"sample {k + 1}: DistP1 = {sk[j, k]:.1f}", loc="left", fontsize=8.5)
            ANN[f"depth|{dataset}|{nm}|{model}|c{j}|k{k}"] = float(sk[j, k])
        im = axs[i, 5].imshow(agree.reshape(GH, GW), cmap=plt.matplotlib.colors.ListedColormap(["white"] + ST.BLUE[2::3][:4]),
                              vmin=-0.5, vmax=4.5, interpolation="nearest")
        axs[i, 5].contour(gt_new.reshape(GH, GW).astype(float), levels=[0.5], colors=ST.INK, linewidths=0.8)
        axs[i, 5].set_title("# samples with a new hit\n(black contour: GT new surface)", loc="left", fontsize=8.5)
        axs[i, 0].set_ylabel(f"j={j}: {lab}\nmean DistP1 {sk[j].mean():.1f}", fontsize=8.5, color=ST.INK)
        ANN[f"depth|{dataset}|{nm}|{model}|c{j}|gtd"] = float(v2["gtd_distp1"][j])
        ANN[f"depth|{dataset}|{nm}|{model}|c{j}|gain"] = float(g[j])
        for a in axs[i]:
            a.set_xticks([])
            a.set_yticks([])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(*scale), cmap=ST.SEQ), ax=axs[-1, :5], orientation="horizontal",
                      fraction=0.08, aspect=60)
    cb.set_label(f"depth of DistP1-new hits ({UNAME[dataset]}); one scale for GT and all samples")
    fig.colorbar(im, ax=axs[-1, 5], orientation="horizontal", fraction=0.08, ticks=range(5), label="# samples")
    regret = g[C].max() - g[pk[model]]
    fig.suptitle(f"{dataset} case '{nm}' — {st['scene']} s{st['start']} t={st['t']} — {model}: 4 stochastic future view hypotheses "
                 f"(regret of the pick {regret:.3f} pp)\ncolour = depth of hits DistP1 counts as new; light gray = hit on known surface; "
                 f"dark gray = UNKNOWN hit inside the dilated band (not counted); white = no hit; pink = invalid ray", x=0.005, ha="left", fontsize=9)
    fig.savefig(os.path.join(FIG, f"depth_{dataset}_{nm}_{model}.png"))
    plt.close(fig)
    return rows


def crop(P, lo, hi):
    return P[np.all((P >= lo) & (P <= hi), 1)] if len(P) else P


def view_of(cam, pts):
    f = (pts.mean(0) - cam) if len(pts) else np.asarray([1.0, 0, 0])
    return float(np.degrees(np.arctan2(-f[1], -f[0]))) + 180.0


def fig_3d(dataset, nm, sid, model, rows, gif=False):
    from scipy.spatial import cKDTree
    z, v2, orc, cz = load_case(dataset, sid)
    u = UNIT[dataset]
    vs = 0.011 * float(z["k_M"]) / u                          # one P1 voxel in figure units
    sk = np.asarray(v2["models"][model]["distp1_k"])
    g = orc["gain"] * 100
    fig = plt.figure(figsize=(16, 2.9 * len(rows) + 0.5))
    obs_all = cz["obs_pts"] / u
    for i, (j, lab) in enumerate(rows):
        gt = cz[f"gt_pts_c{j}"] / u
        sp = [cz[f"{model}_pts_c{j}_k{k}"] / u for k in range(4)]
        cam = cz[f"cam_c{j}"] / u
        allp = np.concatenate([gt] + sp + [cam[None]])
        lo, hi = np.quantile(allp, 0.01, 0), np.quantile(allp, 0.99, 0)
        pad = 0.15 * (hi - lo) + 3 * vs
        lo, hi = np.minimum(lo - pad, cam), np.maximum(hi + pad, cam)
        obs = crop(obs_all, lo, hi)
        if len(obs) > 40000:
            obs = obs[np.random.default_rng(0).choice(len(obs), 40000, replace=False)]
        az = view_of(cam, np.concatenate(sp + [gt]) if len(gt) or any(len(x) for x in sp) else cam[None])
        union = np.concatenate([x for x in sp if len(x)]) if any(len(x) for x in sp) else np.zeros((0, 3))
        if len(union):
            trees = [cKDTree(x) if len(x) else None for x in sp]
            cnt = sum((tr.query(union, k=1, distance_upper_bound=8 * vs)[0] < np.inf).astype(int) if tr is not None else 0 for tr in trees)
        panels = [("GT new surface (black)", [(gt, ST.INK, 1.2)])] + \
                 [(f"sample {k + 1}: DistP1 {sk[j, k]:.1f}", [(sp[k], ST.COL["V2M"] if model == "V2M" else ST.COL["V2O"], 1.0),
                                                            (gt, ST.INK, 0.25)]) for k in range(4)]
        for p, (title, layers) in enumerate(panels + [("union: # samples within 8 voxels", None)]):
            ax = fig.add_subplot(len(rows), 6, 6 * i + p + 1, projection="3d")
            ax.scatter(*obs.T, s=0.3, color=ST.KNOWN_GRAY, alpha=0.5, lw=0, rasterized=True)
            if layers is None:
                if len(union):
                    sc = ax.scatter(*union.T, s=1.2, c=cnt, cmap=plt.matplotlib.colors.ListedColormap(ST.BLUE[2::3][:4]), vmin=0.5, vmax=4.5,
                                    lw=0, rasterized=True)
            else:
                for P, col, s in layers:
                    if len(P):
                        ax.scatter(*P.T, s=s, color=col, lw=0, rasterized=True)
            ax.scatter(*cam, s=30, color=ST.INK, marker="^")
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(hi - lo)
            ax.view_init(elev=28, azim=az)
            ax.set_axis_off()
            ax.set_title(title, fontsize=8.5, loc="left")
        fig.text(0.005, 1 - (i + 0.5) / len(rows), f"j={j}\n{lab}\ngain {g[j]:.3f} pp\nGTD {v2['gtd_distp1'][j]:.1f}", fontsize=8.5,
                 va="center", color=ST.INK, transform=fig.transFigure)
        ANN[f"3d|{dataset}|{nm}|{model}|c{j}|n_pts"] = [int(len(gt))] + [int(len(x)) for x in sp]
        if gif and i == 0 and len(union):
            orbit(dataset, nm, model, obs, union, cnt, gt, cam, lo, hi, az)
    fig.suptitle(f"{dataset} case '{nm}' — {model}: 3D meaning of the 4 hypotheses (gray = observed points up to t, triangle = candidate camera; "
                 f"units {UNAME[dataset]})", x=0.01, ha="left", fontsize=9.5)
    fig.subplots_adjust(left=0.06, right=0.995, wspace=0.0, hspace=0.05, top=0.88, bottom=0.0)
    fig.savefig(os.path.join(FIG, f"3d_{dataset}_{nm}_{model}.png"), bbox_inches="tight")
    plt.close(fig)


def orbit(dataset, nm, model, obs, union, cnt, gt, cam, lo, hi, az):
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*obs.T, s=0.3, color=ST.KNOWN_GRAY, alpha=0.5, lw=0)
    ax.scatter(*union.T, s=1.4, c=cnt, cmap=plt.matplotlib.colors.ListedColormap(ST.BLUE[2::3][:4]), vmin=0.5, vmax=4.5, lw=0)
    if len(gt):
        ax.scatter(*gt.T, s=0.4, color=ST.INK, lw=0)
    ax.scatter(*cam, s=30, color=ST.INK, marker="^")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_axis_off()
    ax.set_title(f"{dataset} '{nm}' {model}: union of 4 samples\n(light→dark blue = 1→4 samples within 8 voxels, black = GT new)", fontsize=8)

    def upd(f):
        ax.view_init(elev=28, azim=az + 6 * f - 60)
        return []
    FuncAnimation(fig, upd, frames=21, interval=180).save(os.path.join(FIG, f"orbit_{dataset}_{nm}_{model}.gif"), writer=PillowWriter(fps=6))
    plt.close(fig)


def fig_summary():
    S = json.load(open(os.path.join(RES, "a2_summary.json")))
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.4))
    for i, d in enumerate(("indoor", "outdoor")):
        x = S[d]
        rows = list(csv.DictReader(open(os.path.join(RES, f"a2_states_{d}.csv"))))
        ax = axs[i, 0]
        for m in ("V2O", "V2M"):
            xs = [float(r["surface_chamfer_div"]) for r in rows if r["model"] == m and r["surface_chamfer_div"] not in ("", "None")
                  and r["chamfer_to_gt_new"] not in ("", "None")]
            ys = [float(r["chamfer_to_gt_new"]) for r in rows if r["model"] == m and r["surface_chamfer_div"] not in ("", "None")
                  and r["chamfer_to_gt_new"] not in ("", "None")]
            ax.scatter(xs, ys, s=7, color=ST.COL[m], alpha=0.5, lw=0, label=ST.LAB[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"diversity: Chamfer between samples ({UNAME[d]})")
        ax.set_ylabel(f"error: Chamfer to GT new surface ({UNAME[d]})")
        ax.legend(fontsize=7.5, loc="upper left")
        ST.clean(ax, None)
        ax.set_title(f"{d}: diversity vs error (per state)", loc="left")
        ax = axs[i, 1]
        labs = ("k1_agree", "loo_pivotal", "top1_k4", "top1_k1_mean")
        names = ("K=1 pick = K=4 pick", "state has a pivotal sample", "Top-1, K=4", "Top-1, K=1 (mean)")
        for o, m in enumerate(("V2O", "V2M")):
            vals = [x["models"][m][k]["mean"] for k in labs]
            ax.barh(np.arange(4) + 0.38 * o, vals, height=0.36, color=ST.COL[m], label=ST.LAB[m])
            for y, v in enumerate(vals):
                ax.text(v + 0.01, y + 0.38 * o, f"{v:.2f}", va="center", fontsize=7.5, color=ST.INK2)
                ANN[f"summary|{d}|{m}|{labs[y]}"] = v
        ax.set_yticks(np.arange(4) + 0.19)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ST.clean(ax, "x")
        ax.set_title(f"{d}: influence of single samples", loc="left")
        for c, key in ((2, "top1"), (3, "regret")):
            ax = axs[i, c]
            for m in ("V2O", "V2M"):
                ks = [K for K in A.K_ABL if f"{m}_K{K}" in x["kabl"]]
                if not ks:
                    continue
                ys = [x["kabl"][f"{m}_K{K}"][key]["mean"] for K in ks]
                ax.plot(range(len(ks)), ys, color=ST.COL[m], marker="o", lw=2, label=ST.LAB[m])
                ANN[f"summary|{d}|{m}|kabl|{key}"] = ys
            ax.set_xticks(range(len(A.K_ABL)))
            ax.set_xticklabels([f"K={K}" for K in A.K_ABL])
            if key == "top1":
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, None)
            ST.clean(ax)
            ax.set_title(f"{d}: {'Top-1 ↑' if key == 'top1' else 'regret (pp) ↓'} vs K", loc="left")
    fig.suptitle("A2: stochastic future view hypotheses — diversity vs error, single-sample influence, number of samples K", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_quality_diversity.png"))
    plt.close(fig)


def main():
    ST.setup()
    os.makedirs(FIG, exist_ok=True)
    # the per-case figures of this script were replaced by a2v2_viz.py (user redesign 2026-10-28); only the summary remains
    fig_summary()
    json.dump(ANN, open(os.path.join(FIG, "annotations.json"), "w"), indent=1)
    print("A2 figures:", len(os.listdir(FIG)))


if __name__ == "__main__":
    main()
