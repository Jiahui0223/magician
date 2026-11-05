"""Experiment F -- figures from results/*.json (CPU, read-only).
  figures/F1_gamma.png        official one-step regret / geometric two-step regret / selected-path true union vs gamma, per layer and phase
  figures/F2_soft.png         the same metrics vs sigma (dist weight and area weight), plus the offline diffusion diagnostic
  figures/F3_K.png            regret and flip rate vs K, within-sample / between-camera variance by phase, adaptive vs equal-budget controls
  figures/F_summary.png       candidate - D-v2 on both layers (scene-clustered 95 % CI) + F4 Soft x K interaction"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.dirname(HERE)
RES = os.environ.get("F_RES") or os.path.join(FD, "results")
FIG = os.environ.get("F_FIG") or os.path.join(FD, "figures")
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
PH = [("all", "all states", "#222222"), ("early", "early t < 20", "#0072B2"), ("middle", "middle 20 ≤ t < 50", "#E69F00"), ("late", "late 50 ≤ t < 100", "#009E73")]
LAY = [("explore", "exploration layer (395 states, 4 scenes)"), ("confirm", "confirmation layer (360 states, 6 new scenes)")]
MET = [("off_regret", "regret, official one-step gain (pp)"), ("geo2_regret", "regret, geometric two-step coverage (pp)"), ("path_union", "selected path: true union (pp)")]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(alpha=0.6, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def J(p):
    p = os.path.join(RES, p)
    return json.load(open(p)) if os.path.exists(p) else None


def val(row, m, ph):
    return row.get(m if ph == "all" else f"{m}@{ph}")


def fig_f1():
    g = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2))
    for r, (lay, title) in enumerate(LAY):
        d = J(f"F1_horizon_discount/F1_{lay}.json")
        for c, (m, lab) in enumerate(MET):
            ax = axs[r, c]
            if d is None:
                ax.set_axis_off()
                continue
            for ph, phl, col in PH:
                y = [val(d["table"][f"gamma_{x:g}"], m, ph) for x in g]
                ax.plot(g, y, "-o", color=col, lw=2.2 if ph == "all" else 1.2, ms=4 if ph == "all" else 3, label=phl)
            ax.set_xticks(g)
            ax.set_xlabel("γ (discount of the second-step marginal)", fontsize=8, color=MUTED)
            ax.set_title(lab if r == 0 else "", fontsize=9, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(title, fontsize=8.5, color=INK)
            style(ax)
    axs[0, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("F1  horizon discount  J = V1 + γ·max_c2 ΔV2   (γ = 1 is D-v2)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG, "F1_gamma.png"), dpi=150)
    plt.close(fig)


def fig_f2():
    sig = [0.25, 0.5, 1.0, 2.0]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2))
    for r, (lay, title) in enumerate(LAY):
        d = J(f"F2_soft_coverage/F2_{lay}.json")
        if d is None:
            for ax in axs[r]:
                ax.set_axis_off()
            continue
        T = d["table"]
        for c, (m, lab) in enumerate(MET[:2]):
            ax = axs[r, c]
            for w, col, nm in (("w", "#0072B2", "soft, depth weight w(z)"), ("area", "#D55E00", "soft, area weight (1)")):
                ax.plot([0] + sig, [T[f"avg0_{w}"][m]] + [T[f"s{s:g}_{w}"][m] for s in sig], "-o", color=col, lw=1.8, ms=4, label=nm)
            ax.axhline(T["hard (D-v2)"][m], color=INK, lw=1.2, ls="--", label="hard (D-v2)")
            ax.axhline(T["hard_area"][m], color=MUTED, lw=1.0, ls=":", label="hard, area weight")
            ax.set_xticks([0] + sig)
            ax.set_xticklabels(["0\n(avg first)"] + [f"{s:g}" for s in sig])
            ax.set_xlabel("σ / τ_cov (τ_cov = 1.0 runtime unit = one scoring cell)", fontsize=8, color=MUTED)
            ax.set_title(lab if r == 0 else "", fontsize=9, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(title, fontsize=8.5, color=INK)
            style(ax)
        ax = axs[r, 2]
        D = d["diffusion"]
        for key, col, nm in (("near_gt_share", "#009E73", "mass on cells near the GT surface"), ("hard_cell_share", "#CC79A7", "mass on the hard cells")):
            ax.plot([0] + sig, [D["avg0_w"][key]] + [D[f"s{s:g}_w"][key] for s in sig], "-o", color=col, lw=1.8, ms=4, label=nm)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0] + sig)
        ax.set_xlabel("σ / τ_cov (depth weight)", fontsize=8, color=MUTED)
        ax.set_title("offline diffusion diagnostic (first-layer views, K = 4)" if r == 0 else "", fontsize=9, color=INK, loc="left")
        style(ax)
    axs[0, 0].legend(fontsize=7.5, frameon=False)
    axs[0, 2].legend(fontsize=7.5, frameon=False)
    fig.suptitle("F2  soft coverage scoring (Gaussian kernel per sample, mean over samples, max-union for two steps)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG, "F2_soft.png"), dpi=150)
    plt.close(fig)


def fig_f3():
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2))
    for r, (lay, title) in enumerate(LAY):
        d = J(f"F3_adaptive_K/F3_{lay}.json")
        if d is None:
            for ax in axs[r]:
                ax.set_axis_off()
            continue
        T, S, V, A = d["table"], d["subsets"], d["variance"], d["adaptive"]
        ax = axs[r, 0]
        Ks = [1, 2, 4, 8]
        ax.plot(Ks, [T[f"K8_first{k}"]["off_regret"] for k in Ks], "-o", color="#0072B2", lw=1.8, ms=4, label="first K samples")
        ax.plot([1, 2, 4], [S[f"K{k}"]["off_regret_subset_mean"] for k in (1, 2, 4)], "-s", color="#E69F00", lw=1.2, ms=4, label="mean over disjoint subsets")
        ad = T["adaptive 4->8"]["off_regret"]
        ax.plot([A["budget_samples_per_state"]], [ad], "D", color="#D55E00", ms=7, label=f"adaptive 4→8 ({A['upgrade_share'] * 100:.0f} % upgraded)")
        rk = "random upgrade (equal budget, 50 seeds)"
        ax.plot([A["budget_samples_per_state"]], [T[rk]["off_regret"]], "x", color=INK, ms=7, mew=1.8, label="random upgrade, same budget")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel("samples per candidate (K)", fontsize=8, color=MUTED)
        ax.set_ylabel(title, fontsize=8.5, color=INK)
        ax.set_title("regret, official one-step gain (pp)" if r == 0 else "", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False)
        style(ax)
        ax = axs[r, 1]
        ax.plot([1, 2, 4], [S[f"K{k}"]["flip_rate"] for k in (1, 2, 4)], "-o", color="#CC79A7", lw=1.8, ms=4)
        ax.set_ylim(0, 1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4])
        ax.set_xticklabels(["1", "2", "4"])
        ax.set_xlabel("samples per candidate (K)", fontsize=8, color=MUTED)
        ax.set_title("pick flip rate between disjoint sample subsets" if r == 0 else "", fontsize=9, color=INK, loc="left")
        style(ax)
        ax = axs[r, 2]
        x = np.arange(4)
        ax.bar(x, [V[p]["ratio_within_over_between"] for p, _, _ in PH], color=[c for _, _, c in PH], width=0.6, edgecolor="white")
        for i, (p, _, _) in enumerate(PH):
            ax.text(i, V[p]["ratio_within_over_between"], f"z<1: {V[p]['top2_z_lt1_share_K4'] * 100:.0f} %", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([lab.split(" ")[0] for _, lab, _ in PH], fontsize=8)
        ax.set_title("within-sample / between-camera variance (K = 8)" if r == 0 else "", fontsize=9, color=INK, loc="left")
        style(ax)
    fig.suptitle("F3  number of samples per candidate and adaptive K (hard D-v2 scoring)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG, "F3_K.png"), dpi=150)
    plt.close(fig)


def fig_summary():
    ex = {"F1": J("F1_horizon_discount/F1_explore.json"), "F2": J("F2_soft_coverage/F2_explore.json"), "F3": J("F3_adaptive_K/F3_explore.json")}
    cf = J("F4_interaction/F_confirm_gates.json")
    sel = J("selection.json")
    if sel is None:
        return
    rows = [("F1 γ = %g" % sel["F1_gamma_star"], ex["F1"]["table"][f"gamma_{sel['F1_gamma_star']:g}"], "F1 gamma*"),
            ("F2 %s" % sel["F2_variant_star"], ex["F2"]["table"].get(sel["F2_variant_star"], {}), "F2 soft*"),
            ("F3 adaptive 4→8", ex["F3"]["table"]["adaptive 4->8"], "F3 adaptive")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for c, (m, lab) in enumerate(MET[:2]):
        ax = axs[c]
        for i, (nm, er, key) in enumerate(rows):
            for j, (src, col, off) in enumerate(((er, "#56B4E9", -0.15), (cf["gate_table"].get(key, {}) if cf else {}, "#0072B2", 0.15))):
                d = src.get(f"d_{m}", {}).get("scene")
                if not d:
                    continue
                y = len(rows) - 1 - i + off
                ci = d.get("ci", [d["d"], d["d"]])
                ax.plot(ci, [y, y], color=col, lw=2)
                ax.plot([d["d"]], [y], "o", color=col, ms=6, label=("exploration" if j == 0 else "confirmation") if i == 0 else None)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
        ax.set_title(f"candidate − D-v2: {lab}\n(negative = better; scene-clustered 95 % CI)", fontsize=9, color=INK, loc="left")
        style(ax)
    axs[0].legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax = axs[2]
    for j, (lay, d, col) in enumerate((("exploration", J("F4_interaction/F4_explore.json"), "#56B4E9"), ("confirmation", cf, "#0072B2"))):
        if not d:
            continue
        I = d["interaction"] if lay == "exploration" else d.get("F4_interaction")
        if not I:
            continue
        for i, (k, nm) in enumerate((("off_regret|quads", "official"), ("geo2_regret|quads", "geometric two-step"))):
            for q, (kk, mk) in enumerate((("gain_K4_vs_K1_hard", "s"), ("gain_K4_vs_K1_soft", "^"), ("interaction", "o"))):
                x = I[k][kk]
                y = i * 4 + q + (j - 0.5) * 0.3
                ci = x.get("ci", [x["d"], x["d"]])
                ax.plot(ci, [y, y], color=col, lw=2)
                ax.plot([x["d"]], [y], mk, color=col, ms=6, label=lay if (i == 0 and q == 2) else None)
    ax.axvline(0, color=INK, lw=0.8)
    lab = []
    for nm in ("official", "geo. two-step"):
        lab += [f"{nm}: K4 − K1 gain, hard", f"{nm}: K4 − K1 gain, soft", f"{nm}: interaction", ""]
    ax.set_yticks(range(8))
    ax.set_yticklabels(lab, fontsize=7.5)
    ax.set_title("F4  K = 1 → 4 regret reduction (pp, > 0 = K4 better)\nand the Soft × K interaction", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F_summary.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_f1()
    fig_f2()
    fig_f3()
    fig_summary()
    print("figures written", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
