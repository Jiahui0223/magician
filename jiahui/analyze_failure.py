"""
Analyze a MAGICIAN failure case from the saved LMDB (no GPU needed).
For a scene, plots:
  (A) coverage-vs-step for all trajectories (failure traj highlighted)
  (B) 3D camera path of the failure traj vs a good traj, over the real reconstructed point cloud
      -> if the failure camera flies into a region with little/no real surface, that supports the
         "lured by imagined (hallucinated) structure" hypothesis.
Also prints quantitative failure signatures.

Usage: python jiahui/analyze_failure.py --scene pantheon --bad 2 --good 0
Output: work_dir/vis/<scene>_failure_analysis.png
"""
import os, argparse, pickle
import numpy as np
import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "work_dir", "vis")
LMDB = os.path.join(ROOT, "results", "scene_exploration", "magician_lmdb")


def load(scene, traj):
    env = lmdb.open(LMDB, readonly=True, lock=False)
    with env.begin() as t:
        raw = t.get(f"{scene}/{traj}".encode())
    env.close()
    return pickle.loads(raw) if raw else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="pantheon")
    ap.add_argument("--bad", type=int, default=2)
    ap.add_argument("--good", type=int, default=0)
    ap.add_argument("--trajs", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    data = {tr: load(args.scene, tr) for tr in args.trajs}
    data = {k: v for k, v in data.items() if v is not None}

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f"MAGICIAN failure analysis — {args.scene}", fontsize=14)

    # ---- (A) coverage curves ----
    ax = fig.add_subplot(1, 3, 1)
    for tr, d in data.items():
        cov = d["coverage"]
        lw, alpha = (2.8, 1.0) if tr == args.bad else (1.2, 0.6)
        ax.plot(cov, lw=lw, alpha=alpha, label=f"traj{tr} (max={max(cov):.2f})")
    ax.set_title("Coverage vs step")
    ax.set_xlabel("step"); ax.set_ylabel("coverage"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # real reconstructed surface (use the GOOD traj's point cloud = most complete)
    pts_good = data[args.good]["points"]
    sub = pts_good[np.random.RandomState(0).choice(len(pts_good), min(15000, len(pts_good)), replace=False)]

    # ---- (B) 3D path of BAD traj over real surface ----
    def path_panel(idx, tr, title):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, c="lightgray", alpha=0.25)
        X = data[tr]["X_cam_history"]
        steps = np.arange(len(X))
        p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=steps, cmap="plasma", s=14)
        ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=0.7, c="k", alpha=0.4)
        ax.scatter(X[0, 0], X[0, 1], X[0, 2], c="lime", s=80, marker="^", label="start")
        ax.set_title(title); ax.legend(fontsize=8)
        fig.colorbar(p, ax=ax, shrink=0.5, label="step")

    path_panel(2, args.bad, f"BAD traj{args.bad} path over real surface")
    path_panel(3, args.good, f"GOOD traj{args.good} path over real surface")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, f"{args.scene}_failure_analysis.png")
    fig.savefig(out, dpi=120)
    print(f"-> {out}\n")

    # ---- quantitative failure signatures ----
    print("=== quantitative signatures ===")
    surf_min, surf_max = pts_good.min(0), pts_good.max(0)
    surf_center = pts_good.mean(0)
    for tr, d in data.items():
        cov = np.array(d["coverage"]); X = np.array(d["X_cam_history"])
        # plateau step: first step reaching 95% of its own final coverage
        final = cov[-1]
        plateau = int(np.argmax(cov >= 0.95 * final)) if final > 0 else -1
        # camera path spread (bbox diagonal) -> small = stuck in a region
        spread = np.linalg.norm(X.max(0) - X.min(0))
        # mean distance of cameras to surface centroid
        d2c = np.linalg.norm(X - surf_center, axis=1).mean()
        # fraction of camera positions INSIDE the surface bbox (flew "into" structure/void)
        inside = np.mean(np.all((X >= surf_min) & (X <= surf_max), axis=1))
        tag = "  <-- BAD" if tr == args.bad else ""
        print(f" traj{tr}: final_cov={final:.3f} maxcov={cov.max():.3f} plateau@step={plateau:3d} "
              f"path_spread={spread:6.1f} mean_dist2center={d2c:6.1f} frac_inside_bbox={inside:.2f}{tag}")


if __name__ == "__main__":
    main()
