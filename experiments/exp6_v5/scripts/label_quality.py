"""EXP6 V5 -- label quality of the Train-6 multi-surface cache (magician env, 1 GPU for the example re-peel).
Aggregates the per-trajectory label files (work_dirs/exp6_v5/labels/v1): surfaces per ray inside the supervised range
(all 7,296 rays of every view), share of rays with >= 2 surfaces, layer-1 == T1.1 t* agreement, K saturation, gap
between the first and second surface, supervised range, stored-subset composition. Figures (experiments/exp6_v5/figures):
  label_surfaces_per_ray.png  share of rays with 0 / 1 / 2 / 3 / 4 / 5 / 6+ surfaces per scene
  label_gap21.png             distribution of t2 - t1 (voxels, log axis) with Delta = 8 VS
  label_example_views.png     re-peeled example views: first / second surface depth and number of surfaces per ray
Output experiments/exp6_v5/reports/label_quality.json. Usage: python label_quality.py"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_multisurface_labels as B  # noqa: E402

np, torch, L11, RY = B.np, B.torch, B.L11, B.RY
LAB = os.path.join(B.R, "work_dirs", "exp6_v5", "labels", "v1")
FIG = os.path.join(B.R, "experiments", "exp6_v5", "figures")
REP = os.path.join(B.R, "experiments", "exp6_v5", "reports")
SCENES = ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {"scenes": {}, "files": 0}
    gaps = {}
    allh = np.zeros(7, np.int64)
    for s in SCENES:
        fs = sorted(glob.glob(os.path.join(LAB, s, "*.npz")))
        h = np.zeros(7, np.int64)
        feq, fmr, sat, views, sec, ranges, sub2, held = [], [], [], 0, 0.0, [], [], 0
        g = []
        for f in fs:
            z = np.load(f)
            m = json.loads(str(z["meta"]))
            sm = m["summary"]
            h += np.asarray(sm["hist_all_rays"], np.int64)
            feq.append(sm["first_eq_min"]); fmr.append(sm["first_maxrel_max"]); sat.append(sm["saturated_mean"])
            views += m["rows"]; sec += m["seconds"]; held += int(m["heldout"])
            gg = z["gap21"].astype(np.float32) / B.VS
            g.append(gg[:: max(1, len(gg) // 20000)])
            ok = z["ray_idx"] >= 0
            ranges.append(float((z["t_hi"][ok].astype(np.float32) > z["t_lo"][ok].astype(np.float32)).mean()))
            sub2.append(float((z["n_surf"][ok] >= 2).mean()))
        out["files"] += len(fs)
        gaps[s] = np.concatenate(g) if g else np.zeros(0)
        allh += h
        out["scenes"][s] = {"trajectories": len(fs), "heldout_trajectories": held, "views": views, "rays": int(h.sum()), "hist_surfaces_per_ray_0_to_6plus": h.tolist(),
                            "share_ge1": float(h[1:].sum() / max(1, h.sum())), "share_ge2": float(h[2:].sum() / max(1, h.sum())),
                            "share_ge2_among_ge1": float(h[2:].sum() / max(1, h[1:].sum())), "first_eq_tstar_min": float(min(feq)) if feq else None,
                            "first_maxrel_max": float(max(fmr)) if fmr else None, "K16_saturated_share": float(np.mean(sat)) if sat else None,
                            "rays_with_supervised_range_in_subset": float(np.mean(ranges)) if ranges else None, "subset_share_ge2": float(np.mean(sub2)) if sub2 else None,
                            "gap21_vox_quantiles_5_25_50_75_95": np.quantile(gaps[s], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist() if len(gaps[s]) else None,
                            "gap21_share_ge8vox": float((gaps[s] >= 8).mean()) if len(gaps[s]) else None,
                            "gap21_share_ge32vox": float((gaps[s] >= 32).mean()) if len(gaps[s]) else None, "gpu_seconds": sec}
    ga = np.concatenate(list(gaps.values()))
    out["all"] = {"rays": int(allh.sum()), "hist": allh.tolist(), "share_ge2": float(allh[2:].sum() / allh.sum()), "share_ge2_among_ge1": float(allh[2:].sum() / allh[1:].sum()),
                  "gap21_vox_quantiles_5_25_50_75_95": np.quantile(ga, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(), "gap21_share_ge8vox": float((ga >= 8).mean()),
                  "gap21_share_ge32vox": float((ga >= 32).mean())}
    # ---- figure 1: surfaces per ray
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cols = ["#d9d9d9", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
    left = np.zeros(len(SCENES) + 1)
    rows = [np.asarray(out["scenes"][s]["hist_surfaces_per_ray_0_to_6plus"], float) for s in SCENES] + [allh.astype(float)]
    rows = np.stack([r / max(1, r.sum()) for r in rows])
    labels = SCENES + ["Train-6"]
    for k in range(7):
        ax.barh(labels, rows[:, k], left=left, color=cols[k], edgecolor="white", linewidth=1.5, label=f"{k}" if k < 6 else "6+")
        left += rows[:, k]
    ax.set_xlim(0, 1); ax.set_xlabel("share of rays (all 7,296 rays of every training view)")
    ax.legend(title="surfaces in range", ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=8)
    ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Train-6 mesh labels: surfaces per ray inside the supervised range")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "label_surfaces_per_ray.png"), dpi=150); plt.close(fig)
    # ---- figure 2: gap21
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.logspace(-1, 3.2, 60)
    ax.hist(np.clip(ga, 0.1, None), bins=bins, color="#2171b5", alpha=0.85)
    ax.set_xscale("log"); ax.axvline(8, color="#cb181d", lw=1.5, ls="--"); ax.text(8.5, ax.get_ylim()[1] * 0.9, "Delta = 8 VS", color="#cb181d", fontsize=8)
    ax.axvline(0.5, color="#636363", lw=1, ls=":"); ax.text(0.53, ax.get_ylim()[1] * 0.8, "merge tol 0.5 VS", color="#636363", fontsize=8)
    ax.set_xlabel("gap between the first and second surface (voxels, VS = 0.011 model units)"); ax.set_ylabel("rays (subsampled)")
    ax.set_title("Gap between first and second surface (Train-6, all labelled views)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "label_gap21.png"), dpi=150); plt.close(fig)
    # ---- figure 3: example views re-peeled
    dev = torch.device("cuda")
    ex = [("pisa", "cg-A-0001", 300), ("liberty", "cg-A-0001", 200), ("redeemer", "cg-B-0001", 400)]
    fig, axs = plt.subplots(len(ex), 3, figsize=(11, 2.4 * len(ex)))
    for r, (s, tj, row) in enumerate(ex):
        z = np.load(os.path.join(B.SHARDS, s, f"{tj}.npz"))
        mesh, _ = L11.load_mesh(s, dev)
        peel = B.Peeler(mesh, dev, 16, batch=1)
        t = peel.layers([z["R"][row]], [z["T"][row]], [z["jitter"][row]], float(z["k_M"]))[0]
        tm, cnt = B.merge_dups(t, 0.5 * B.VS)
        fr = L11.frame_of(s)
        o, d, tcap, dz = B.view_geometry(fr, z["R"][row:row + 1], z["T"][row:row + 1], z["jitter"][row:row + 1], float(z["k_M"]))
        hi = np.minimum(tcap[0], B.ray_box_exit(o[0], d[0], np.asarray(z["dims"], np.float64) * B.VS))
        inr = np.isfinite(tm) & (tm <= hi[:, None])
        n = inr.sum(1)
        t1 = np.where(inr[:, 0], tm[:, 0], np.nan)
        t2 = np.where(n >= 2, np.where(inr, tm, np.inf)[:, 1], np.nan)
        vmax = np.nanmax(np.concatenate([t1, t2])) if np.isfinite(np.nanmax(t1)) else 1
        for c, (img, ttl, cm) in enumerate(((t1, "first surface depth", "viridis"), (t2, "second surface depth (if any)", "viridis"), (n, "surfaces per ray", "Blues"))):
            im = axs[r, c].imshow(np.asarray(img, float).reshape(RY.GH, RY.GW), cmap=cm, vmin=0, vmax=(vmax if c < 2 else 6))
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            axs[r, c].set_title(f"{s} {tj} view {row}: {ttl}" if c == 0 else ttl, fontsize=8)
            fig.colorbar(im, ax=axs[r, c], fraction=0.03)
        del mesh, peel
        torch.cuda.empty_cache()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "label_example_views.png"), dpi=150); plt.close(fig)
    json.dump(out, open(os.path.join(REP, "label_quality.json"), "w"), indent=1)
    print(json.dumps(out["all"], indent=1)); print({s: (v["trajectories"], round(v["share_ge2"], 3), v["first_eq_tstar_min"], round(v["K16_saturated_share"], 4)) for s, v in out["scenes"].items()})


if __name__ == "__main__":
    main()
