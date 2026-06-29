"""
Plot MACARONS training-loss curves for a run.
- Total train_loss: from losses_data_<model>.json (the only loss saved to disk).
- Depth / Occupancy / Coverage component losses: regex-extracted from the SLURM .out log
  (they are only printed there, and the log is flooded by pytorch3d warnings — we just grep them out).

Usage:
  python jiahui/plot_train_curves.py --model jiahui_macarons_ddp --log jiahui/log/train_3031204.out
Output: work_dir/vis/<model>_train_curves.png
"""
import os, re, json, argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "work_dir", "vis")


def rolling(xs, w):
    if len(xs) < w or w <= 1:
        return list(range(len(xs))), xs
    out, idx = [], []
    s = sum(xs[:w])
    for i in range(w - 1, len(xs)):
        if i >= w:
            s += xs[i] - xs[i - w]
        out.append(s / w)
        idx.append(i)
    return idx, out


def grep_floats(log_text, label):
    return [float(x) for x in re.findall(label + r" loss: tensor\(([0-9.]+)", log_text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jiahui_macarons_ddp")
    ap.add_argument("--log", default=None, help="path to the .out log (for component losses)")
    ap.add_argument("--smooth", type=int, default=25, help="rolling-average window for per-pose curves")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # --- total loss from json ---
    jpath = os.path.join(ROOT, f"losses_data_{args.model}.json")
    total = json.load(open(jpath))["train_loss"] if os.path.exists(jpath) else []

    # --- component losses from log ---
    depth = occ = cov = []
    if args.log and os.path.exists(args.log):
        txt = open(args.log, "r", errors="ignore").read()
        depth = grep_floats(txt, "Depth")
        occ = grep_floats(txt, "Occupancy")
        cov = grep_floats(txt, "Coverage")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"MACARONS training curves — {args.model}", fontsize=14)

    def panel(ax, raw, title, color, trained):
        if not raw:
            ax.set_title(title + "  (no data)"); return
        ax.plot(raw, color=color, alpha=0.25, lw=0.6, label="raw")
        ix, sm = rolling(raw, args.smooth)
        if sm:
            ax.plot(ix, sm, color=color, lw=2.0, label=f"rolling({args.smooth})")
        tag = "TRAINED" if trained else "FROZEN (monitor only)"
        ax.set_title(f"{title}  [{tag}]  n={len(raw)}")
        ax.set_xlabel("logged step"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)

    # total loss (per scene, ~2/epoch) -> also label epochs
    ax = axes[0, 0]
    if total:
        ax.plot(total, "o-", color="black", ms=3, lw=1, label="train_loss (per scene)")
        ax.set_title(f"Total train_loss  n={len(total)}  (~2/epoch -> ~{len(total)//2} epochs)")
        ax.set_xlabel("scene step"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.set_title("Total train_loss (no json)")

    panel(axes[0, 1], occ, "Occupancy loss (SconeOcc)", "tab:blue", trained=True)
    panel(axes[1, 0], cov, "Coverage loss (SconeVis)", "tab:green", trained=True)
    panel(axes[1, 1], depth, "Depth loss (ManyDepth)", "tab:red", trained=False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT, f"{args.model}_train_curves.png")
    fig.savefig(out, dpi=120)
    print(f"-> {out}")
    print(f"   total={len(total)}  depth={len(depth)}  occ={len(occ)}  cov={len(cov)}")
    if total:
        print(f"   total_loss: first={total[0]:.3f}  last={total[-1]:.3f}")
    if occ:
        print(f"   occ_loss:   first={occ[0]:.3f}  last={occ[-1]:.3f}")
    if cov:
        print(f"   cov_loss:   first={cov[0]:.3f}  last={cov[-1]:.3f}")


if __name__ == "__main__":
    main()
