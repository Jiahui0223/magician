"""Camera Pose C2 -- camera-pose distribution figure (magician env, CPU; evaluation-side visualisation only).
Top-down (z-up metres) view of chosen VALSEL states: GT mesh vertices within +-0.4 m of the camera height (obstacles, grey),
original lattice positions (light grid), the current pose (black), C0 candidates (squares) and new C2 candidates (circles)
with yaw arrows, coloured by their true one-step gain (pp), the oracle-best C0 and overall candidates (stars).
States: the two largest C2c oracle improvements ("success") and two states where C2c adds nothing although many new
candidates are legal ("no gain"). Output C2/reports/figures/pose_distribution.png. Usage: python c2_viz_poses.py"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C2 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(C2)))
EXP5 = os.path.join(REPO, "experiments", "exp5_future_view")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import c2_poses as CP  # noqa: E402
import t2_common as C  # noqa: E402
import t21_traj as TJ  # noqa: E402
import t21_levels as LV  # noqa: E402

S, GE = TJ.S, TJ.GE
ORC = os.path.join(REPO, "work_dirs", "exp_camera_pose", "C2", "oracle")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    A = json.load(open(os.path.join(C2, "reports", "stage_a.json")))
    rows = A["rows"]
    d = sorted(rows, key=lambda r: r["C2c"]["best"] - r["C0"]["best"], reverse=True)
    succ = d[:2]
    nog = [r for r in rows if abs(r["C2c"]["best"] - r["C0"]["best"]) < 1e-12 and r["C2c"]["n_legal"] >= 20 and r["C0"]["best"] > 0.2][:2]
    sel = [("success", r) for r in succ] + [("no gain", r) for r in nog]
    fig, axes = plt.subplots(1, len(sel), figsize=(5.0 * len(sel), 5.2), squeeze=False)
    cmap = plt.get_cmap("viridis")
    mesh_cache = {}
    for ax, (kind, r) in zip(axes[0], sel):
        s = json.load(open(os.path.join(ORC, f"{r['sid']}.json")))
        man = json.load(open(os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2", "valsel", "manifests", "states_valsel.json")))["states"]
        st = next(x for x in man if x["sid"] == r["sid"])
        card = C.load_card(st["scene"])
        g = GE.grid_def(card)
        if st["scan"] not in mesh_cache:
            mesh_cache = {st["scan"]: LV.mesh_zup(st["scan"])}
        v, _ = mesh_cache[st["scan"]]
        p0 = np.asarray(S.rt_to_zup_m(CP.position(g, s["f0"])))
        near = v[(np.abs(v[:, 2] - p0[2]) < 0.4) & (np.linalg.norm(v[:, :2] - p0[:2], axis=1) < 3.0)]
        if len(near) > 20000:
            near = near[np.random.default_rng(0).choice(len(near), 20000, replace=False)]
        ax.scatter(near[:, 0], near[:, 1], s=0.3, color="#adb5bd", zorder=1)
        gains = np.asarray([c["gain"] * 100 for c in s["candidates"]])
        norm = Normalize(vmin=0, vmax=max(gains.max(), 1e-6))
        for c in sorted(s["candidates"], key=lambda c: c["gain"]):
            p = np.asarray(S.rt_to_zup_m(CP.position(g, c["f"])))
            e, az = CP.angles(g, c["f"])
            u = np.array([np.sin(np.deg2rad(az)), -np.cos(np.deg2rad(az))]) * 0.22 * np.cos(np.deg2rad(e))
            col = cmap(norm(c["gain"] * 100))
            jit = 0.0 if tuple(c["f"][:3]) != tuple(s["f0"][:3]) else 0.0
            ax.arrow(p[0] + jit, p[1], u[0], u[1], width=0.006, head_width=0.05, color=col, alpha=0.9, zorder=3, length_includes_head=True)
            ax.scatter([p[0]], [p[1]], marker="s" if c["family"] == "C0" else "o", s=34 if c["family"] == "C0" else 18, facecolor=col, edgecolor="#212529" if c["family"] == "C0" else "none", lw=0.8, zorder=4)
        e0, a0 = CP.angles(g, s["f0"])
        u0 = np.array([np.sin(np.deg2rad(a0)), -np.cos(np.deg2rad(a0))]) * 0.3
        ax.scatter([p0[0]], [p0[1]], marker="X", s=90, color="#212529", zorder=6, label="current pose")
        ax.arrow(p0[0], p0[1], u0[0], u0[1], width=0.01, head_width=0.07, color="#212529", zorder=6)
        b0 = max([c for c in s["candidates"] if c["family"] == "C0"], key=lambda c: c["gain"])
        ba = max(s["candidates"], key=lambda c: c["gain"])
        for b, col, lab in ((b0, "#e8590c", f"C0 oracle best {b0['gain'] * 100:.2f} pp"), (ba, "#c92a2a", f"overall best {ba['gain'] * 100:.2f} pp ({ba['family']})")):
            p = np.asarray(S.rt_to_zup_m(CP.position(g, b["f"])))
            ax.scatter([p[0]], [p[1]], marker="*", s=240, facecolor="none", edgecolor=col, lw=1.6, zorder=7, label=lab)
        ax.set_aspect("equal"); ax.set_xlim(p0[0] - 1.6, p0[0] + 1.6); ax.set_ylim(p0[1] - 1.6, p0[1] + 1.6)
        ax.set_title(f"{kind}: {r['tag'].split('_ogr')[0].split('_rnv')[0]} t={r['t']}  (C2c Δ {r['C2c']['best'] - r['C0']['best']:+.2f} pp)", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="lower left"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label="true one-step gain (pp)")
    fig.suptitle("C2 candidate poses (top view): squares = original C0 candidates, dots = new C2 candidates, arrows = yaw; grey = GT mesh near camera height (visualisation only)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.join(C2, "reports", "figures"), exist_ok=True)
    fig.savefig(os.path.join(C2, "reports", "figures", "pose_distribution.png"), dpi=130)
    print([(k, r["sid"], r["t"]) for k, r in sel])


if __name__ == "__main__":
    main()
