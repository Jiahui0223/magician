"""CVPR paper figures (display layer). Reads the canonical CVPR Table 1 episode records (official coverage curves, 101 values each)
and the validated aggregates (results/per_model.csv, per_scene.csv) to cross-check; never edits them.
  figures/coverage_curves_{all14,dev5,test3}.{png,pdf} + coverage_curves_3panel.{png,pdf}
      mean official coverage vs exploration step (0-100); per scene: mean of the 5 starts; split: scene-equal mean;
      shaded = scene-level bootstrap 95% CI (10k, seed 0) for the configured band models only.
  figures/improvement_over_magician.{png,pdf}
      paired per-scene differences vs MAGICIAN (re-evaluated, same starts): dAUC@20, dAUC@100, dFinal; marker = scene mean of
      the 5 paired start differences, bar = +-1 SE over those 5; bottom row = All-14 mean with scene-level bootstrap 95% CI.
  figures/exploratory/radar_all14.{png,pdf}   (exploratory only: identical 0-1 scale on every axis for every method)
Checks (written to MANIFEST.json): curve value at t = 20 / 50 / 100 and trapz / 100 equal the per_model.csv Cov@20 / Cov@50 /
Final / AUC@100 of each complete group; every curve model is READY and complete on the plotted scenes."""
import csv
import datetime
import glob
import hashlib
import json
import os
import subprocess

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(PT)))
CFG = yaml.safe_load(open(os.path.join(PT, "config.yaml")))
P = lambda k: os.path.join(REPO, CFG["sources"][k])  # noqa: E731
FIG = os.path.join(PT, "figures")
S14 = [s for s in CFG["scene_order"] if s != "sofia_church"]
GROUPS = {"all14": S14, "dev5": [s for s in S14 if CFG["split_of"][s] == "dev"], "test3": [s for s in S14 if CFG["split_of"][s] == "test"]}
GLAB = {"all14": "All-14", "dev5": "Dev-5", "test3": "Test-3"}
GKEY = {"all14": "all14", "dev5": "dev", "test3": "test"}
BAND = ["MAG-B", "V3-D1-H3-NORISK"]
CHECKS = []
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False})


def check(cond, msg):
    CHECKS.append({"ok": bool(cond), "check": msg})
    if not cond:
        raise AssertionError(msg)


def style(m):
    st = dict(CFG["style"][m])
    if isinstance(st["ls"], list):
        st["ls"] = (st["ls"][0], tuple(st["ls"][1]))
    return st


EXCLUDED = {}


def load_curves(models):
    """Only models that are READY and complete on all 14 scenes x 5 starts are plotted; others are excluded (recorded in
    MANIFEST.json as partial) -- never padded or partially averaged."""
    mreg = {m["id"]: m.get("status") for m in yaml.safe_load(open(P("model_registry")))["models"]}
    C = {}
    for m in models:
        if mreg.get(m) != "READY":
            EXCLUDED[m] = f"registry status {mreg.get(m)}"
            continue
        miss = [f"{s}_s{k}" for s in S14 for k in range(5) if not os.path.exists(os.path.join(REPO, CFG["sources"]["episodes"], m, f"{s}_s{k}.json"))]
        if miss:
            EXCLUDED[m] = f"partial: {len(miss)} of 70 episodes missing"
            continue
        for s in S14:
            cs = []
            for k in range(5):
                r = json.load(open(os.path.join(REPO, CFG["sources"]["episodes"], m, f"{s}_s{k}.json")))
                c = np.asarray(r["coverage_official"], float)
                check(len(c) == 101 and int(r["start"]) == k and r["scene"] == s, f"{m} {s} s{k}: 101-value official curve")
                cs.append(c)
            C[(m, s)] = np.stack(cs)
    check(CFG["improvement_reference"] not in EXCLUDED, "reference model complete")
    return C


def scene_mean(C, m, s):
    return C[(m, s)].mean(0)


def group_curve(C, m, scenes, boot=None):
    X = np.stack([scene_mean(C, m, s) for s in scenes])
    mu = X.mean(0)
    if boot is None:
        return mu, None, None
    rng = np.random.default_rng(CFG["bootstrap"]["seed"])
    idx = rng.integers(0, len(scenes), (CFG["bootstrap"]["n"], len(scenes)))
    B = X[idx].mean(1)
    return mu, np.quantile(B, 0.025, axis=0), np.quantile(B, 0.975, axis=0)


def save(fig, name, sub=None):
    d = os.path.join(FIG, sub) if sub else FIG
    os.makedirs(d, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=CFG["dpi"], bbox_inches="tight")
    plt.close(fig)


def curves(C, per_model):
    models = [m for m in CFG["curve_models"] if m not in EXCLUDED]
    fig3, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for gi, (g, scenes) in enumerate(GROUPS.items()):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for A in (ax, axs[gi]):
            for m in models:
                st = style(m)
                mu, lo, hi = group_curve(C, m, scenes, boot=True if m in BAND else None)
                ref = per_model.get((m, GKEY[g]))
                if A is ax and ref is not None and ref["complete"] == "True":
                    for k, q in ((20, "cov20"), (50, "cov50"), (100, "final")):
                        check(abs(mu[k] - float(ref[q])) < 1e-9, f"{m} {g}: curve[{k}] == per_model {q}")
                    check(abs(np.trapz(mu) / 100 - float(ref["auc100"])) < 1e-9, f"{m} {g}: trapz(curve)/100 == per_model auc100")
                if lo is not None:
                    A.fill_between(np.arange(101), lo, hi, color=st["color"], alpha=0.15, lw=0)
                A.plot(np.arange(101), mu, color=st["color"], ls=st["ls"], lw=1.6 if m == "V3-D1-H3-NORISK" else 1.1,
                       label=CFG["model_labels"][m])
            for x in (20, 50):
                A.axvline(x, color="#bbbbbb", lw=0.5, zorder=0)
            A.set_xlim(0, 100)
            A.set_ylim(0, 1.0)
            A.set_xticks([0, 20, 50, 100])
            A.set_xlabel("Exploration step")
            A.grid(axis="y", color="#e5e5e5", lw=0.5)
            A.set_title(f"{GLAB[g]} ({len(scenes)} scenes $\\times$ 5 starts)", fontsize=9)
        ax.set_ylabel("Surface coverage (official)")
        ax.legend(fontsize=7, loc="lower right")
        save(fig, f"coverage_curves_{g}")
    axs[0].set_ylabel("Surface coverage (official)")
    axs[-1].legend(fontsize=6.5, loc="lower right")
    fig3.tight_layout()
    save(fig3, "coverage_curves_3panel")


def improvement(C):
    ref = CFG["improvement_reference"]
    models = [m for m in CFG["improvement_models"] if m not in EXCLUDED]
    mets = [("auc20", r"$\Delta$AUC@20", lambda c: np.trapz(c[..., :21], axis=-1) / 20), ("auc100", r"$\Delta$AUC@100", lambda c: np.trapz(c, axis=-1) / 100),
            ("final", r"$\Delta$Final coverage", lambda c: c[..., 100])]
    labels = [f"{CFG['scene_labels'][s]} ({CFG['split_of'][s]})" for s in S14] + ["All-14 mean"]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 4.6), sharey=True)
    y = np.arange(len(labels))[::-1].astype(float)
    out = {}
    for ai, (q, lab, fn) in enumerate(mets):
        A = axs[ai]
        for mi, m in enumerate(models):
            st = style(m)
            off = (mi - (len(models) - 1) / 2) * 0.28
            d_sc, se = [], []
            for s in S14:
                d = fn(C[(m, s)]) - fn(C[(ref, s)])            # 5 paired start differences
                d_sc.append(d.mean())
                se.append(d.std(ddof=1) / np.sqrt(5))
            d_sc = np.asarray(d_sc)
            rng = np.random.default_rng(CFG["bootstrap"]["seed"])
            B = d_sc[rng.integers(0, 14, (CFG["bootstrap"]["n"], 14))].mean(1)
            lo, hi = np.quantile(B, [0.025, 0.975])
            out[f"{m}|{q}"] = {"mean": float(d_sc.mean()), "ci": [float(lo), float(hi)], "wins": int((d_sc > 0).sum()), "losses": int((d_sc < 0).sum()),
                                "per_scene": dict(zip(S14, map(float, d_sc)))}
            mk = st["marker"]
            face = st["color"] if m == "V3-D1-H3-NORISK" else "white"
            A.errorbar(d_sc, y[:-1] + off, xerr=se, fmt=mk, ms=4, mfc=face, mec=st["color"], ecolor=st["color"], elinewidth=0.7, capsize=0,
                       label=CFG["model_labels"][m] if ai == 0 else None)
            A.errorbar([d_sc.mean()], [y[-1] + off], xerr=[[d_sc.mean() - lo], [hi - d_sc.mean()]], fmt=mk, ms=5.5, mfc=face, mec=st["color"],
                       ecolor=st["color"], elinewidth=1.0, capsize=2)
        A.axvline(0, color="k", lw=0.6)
        A.axhline(y[-1] + 0.5, color="#999999", lw=0.5)
        A.set_xlabel(lab + " vs MAGICIAN")
        A.grid(axis="x", color="#e5e5e5", lw=0.5)
        A.set_ylim(-0.7, len(labels) - 0.3)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(labels, fontsize=7.5)
    axs[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=2)
    fig.tight_layout()
    save(fig, "improvement_over_magician")
    return out


def radar(per_model):
    keys = [("auc20", "AUC@20"), ("auc50", "AUC@50"), ("auc100", "AUC@100"), ("cov20", "Cov@20"), ("cov50", "Cov@50"), ("final", "Final")]
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False)
    fig = plt.figure(figsize=(3.6, 3.6))
    A = fig.add_subplot(111, polar=True)
    for m in [x for x in CFG["radar_models"] if x not in EXCLUDED]:
        r = per_model[(m, "all14")]
        v = [float(r[k]) for k, _ in keys]
        st = style(m)
        A.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], color=st["color"], ls=st["ls"], lw=1.2, label=CFG["model_labels"][m])
    A.set_xticks(ang)
    A.set_xticklabels([l for _, l in keys], fontsize=8)
    A.set_ylim(0, 1)                                       # identical absolute scale on every axis and for every method
    A.set_yticks([0.25, 0.5, 0.75, 1.0])
    A.tick_params(axis="y", labelsize=6)
    A.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    A.set_title("All-14 (exploratory; same 0-1 scale on all axes)", fontsize=8)
    save(fig, "radar_all14", sub="exploratory")


def main():
    per_model = {(r["model"], r["group"]): r for r in csv.DictReader(open(P("per_model")))}
    models = sorted(set(CFG["curve_models"]) | set(CFG["improvement_models"]) | {CFG["improvement_reference"]} | set(CFG["radar_models"]))
    C = load_curves(models)
    curves(C, per_model)
    imp = improvement(C)
    radar(per_model)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        commit = None
    man_p = os.path.join(PT, "MANIFEST.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {}
    man["figures"] = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "git_head_at_generation": commit, "script": os.path.relpath(__file__, REPO),
                      "config_sha256": hashlib.sha256(open(os.path.join(PT, "config.yaml"), "rb").read()).hexdigest(),
                      "episodes_read": len(C) * 5, "models": models, "band_models": BAND, "excluded_partial": EXCLUDED, "improvement": imp, "checks": CHECKS}
    json.dump(man, open(man_p, "w"), indent=1)
    print(f"figures ok: {len(CHECKS)} checks passed")
    for k, v in imp.items():
        print(k, f"{v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] {v['wins']}/{v['losses']}")


if __name__ == "__main__":
    main()
