"""EXP5 Task 0 / T0.12-C -- aggregation of the offline diagnosis (seen2scene env, CPU; descriptive).
Inputs MC/diag_c/{oracle,mag,v2_region,v2_level}/<group>/<tag>.json + the stored grid episodes / planner logs.
Scorers at a state: V2 (frozen V2K4 DistP1; stored scores on G-V2K4 / G-V2K4-NV, replay elsewhere), V2-level (probe:
same V2, region = whole level), MAG (MAGICIAN Greedy layer-0 gain; stored on G-MAG-*, replay elsewhere; visited = invalid),
oracle (true one-step C_obs gain, evaluation side).
  horizon  per group (10 episodes): share of the remaining work inside V2's region; share of the oracle-best candidate's
           gain that lies outside it; regret of the executed choice (oracle best - oracle of the choice)
  cross    focus episodes: per trajectory and scorer, hit rate (argmax = oracle best), mean regret, mean Spearman
  follow   MAG-B states just before it left the explored part: rank of MAG-B's pick under V2 / V2-level / oracle
  station  contrast on the 5 first starts: V2K4 on the station graph (T0.6, 100 moves) vs V2K4 on the grid -- share of the
           remaining work inside the active region (moves 50-100) and camera displacement between refreshes
Output MC/analysis/t12c.json, figures/grid/diag_c_horizon.png, figures/grid/diag_c_TbHJ.png
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_common as M  # noqa: E402

D = os.path.join(M.MC, "diag_c")
GROUPS = ("G-V2K4", "G-V2K4-NV", "G-MAG-G", "G-MAG-B")
FOCUS = ("TbHJrupSAjP_L1_S112", "TbHJrupSAjP_L1_S114", "oLBMNvg9in8_L1_S78")
FOLLOW = {"TbHJrupSAjP_L1_S112": range(50, 63), "TbHJrupSAjP_L1_S114": range(4, 15)}


def load(kind, g, tag):
    p = os.path.join(D, kind, g, f"{tag}.json")
    return json.load(open(p))["rows"] if os.path.exists(p) else None


def ep_dir(g, tag):
    return os.path.join(M.MC, "grid_protocol", "closed_loop", g, tag)


def stored_scores(g, tag):
    return [x["scores"] for x in map(json.loads, open(os.path.join(ep_dir(g, tag), "planner.jsonl")))]


def scorer_rows(kind, g, tag):
    """Per-state score lists of a scorer along trajectory (g, tag); None entries = invalid candidate."""
    if kind == "V2":
        return stored_scores(g, tag) if g in ("G-V2K4", "G-V2K4-NV") else [r["scores"] for r in load("v2_region", g, tag) or []] or None
    if kind == "V2-level":
        r = load("v2_level", g, tag)
        return [x["scores"] for x in r] if r else None
    if kind == "MAG":
        return stored_scores(g, tag) if g.startswith("G-MAG") else ([x["scores"] for x in load("mag", g, tag)] if load("mag", g, tag) else None)


def argmax_valid(s):
    a = np.asarray([-np.inf if v is None else v for v in s], float)
    return int(np.flatnonzero(a == a.max())[0]) if np.isfinite(a.max()) else None


def per_state(scores, orc):
    """hit, regret, spearman of one scorer against the oracle rows."""
    hit, reg, rho = [], [], []
    for s, o in zip(scores, orc):
        g = np.asarray(o["gain"])
        j = argmax_valid(s)
        if j is None or g.max() <= 0:
            continue
        hit.append(float(g[j] == g.max()))
        reg.append(float(g.max() - g[j]))
        ok = [k for k, v in enumerate(s) if v is not None]
        if len(ok) >= 3 and np.ptp(g[ok]) > 0 and np.ptp(np.asarray([s[k] for k in ok], float)) > 0:
            rho.append(float(spearmanr([s[k] for k in ok], g[ok])[0]))
    return {"n": len(hit), "hit": float(np.mean(hit)) if hit else None, "regret_x1e3": 1e3 * float(np.mean(reg)) if reg else None,
            "spearman": float(np.mean(rho)) if rho else None}


def main():
    out = {"doc": __doc__, "horizon": {}, "horizon_per_episode": {}, "cross": {}, "follow": {}, "reproduction": {}}
    tags = [t for t, *_ in M.grid_episodes()]
    for g in GROUPS:
        acc = {"rem_in_region_50_150": [], "best_gain_outside_share": [], "own_regret_x1e3": [], "own_hit": []}
        for tag in tags:
            orc = load("oracle", g, tag)
            if orc is None:
                continue
            ep = json.load(open(os.path.join(ep_dir(g, tag), "episode.json")))
            picks = [s["pick"] for s in ep["steps"]]
            ri = [r["remaining_in_region_share"] for r in orc[50:]]
            outs, regs, hits = [], [], []
            for r, p in zip(orc, picks):
                gg, gi = np.asarray(r["gain"]), np.asarray(r["gain_in_region"])
                if gg.max() <= 0:
                    continue
                b = int(np.argmax(gg))
                outs.append(float((gg[b] - gi[b]) / gg[b]))
                regs.append(float(gg.max() - gg[p]))
                hits.append(float(gg[p] == gg.max()))
            e = {"rem_in_region_50_150": float(np.mean(ri)), "best_gain_outside_share": float(np.mean(outs)),
                 "own_regret_x1e3": 1e3 * float(np.mean(regs)), "own_hit": float(np.mean(hits))}
            out["horizon_per_episode"].setdefault(g, {})[tag] = e
            for k in acc:
                acc[k].append(e[k])
        out["horizon"][g] = {k: float(np.mean(v)) for k, v in acc.items() if v}
        out["horizon"][g]["n_episodes"] = len(acc["own_hit"])
    for tag in FOCUS:
        for g in ("G-V2K4", "G-V2K4-NV", "G-MAG-B"):
            orc = load("oracle", g, tag)
            if orc is None:
                continue
            for kind in ("V2", "V2-level", "MAG"):
                sc = scorer_rows(kind, g, tag)
                if sc:
                    out["cross"].setdefault(tag, {}).setdefault(g, {})[kind] = per_state(sc[:len(orc)], orc)
    for tag, rng in FOLLOW.items():
        orc = load("oracle", "G-MAG-B", tag)
        ep = json.load(open(os.path.join(ep_dir("G-MAG-B", tag), "episode.json")))
        sc = {k: scorer_rows(k, "G-MAG-B", tag) for k in ("V2", "V2-level", "MAG")}
        rows = []
        for t in rng:
            p = ep["steps"][t]["pick"]
            r = {"t": t, "pick_pose": ep["executed"][t + 1], "n_legal": len(orc[t]["gain"])}
            for k, s in list(sc.items()) + [("oracle", [o["gain"] for o in orc] if orc else None)]:
                if s:
                    a = np.asarray([-np.inf if v is None else v for v in s[t]], float)
                    r[f"rank_{k}"] = int((a > a[p]).sum()) + 1
            rows.append(r)
        out["follow"][tag] = rows
    for kind, g, tag in (("v2_region", "G-V2K4", "TbHJrupSAjP_L1_S112"), ("mag", "G-MAG-B", "TbHJrupSAjP_L1_S112")):
        r = load(kind, g, tag)
        if r:
            out["reproduction"][f"{kind}:{g}:{tag}"] = max(x.get("max_abs_diff_vs_stored", 0.0) for x in r)
    out["station"] = station_contrast()
    json.dump(out, open(os.path.join(M.MC, "analysis", "t12c.json"), "w"), indent=1)
    print("reproduction", out["reproduction"])
    for g, h in out["horizon"].items():
        print("horizon", g, {k: round(v, 3) for k, v in h.items()})
    for tag, d in out["cross"].items():
        for g, s in d.items():
            print("cross", tag, g, {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in s.items()})
    for tag, rows in out["follow"].items():
        print("follow", tag, [(r["t"], r.get("rank_V2"), r.get("rank_V2-level"), r.get("rank_MAG"), r.get("rank_oracle"), r["n_legal"]) for r in rows])
    print("station", json.dumps(out["station"]["mean"]))
    figures(out)


def station_contrast():
    """Remaining work inside V2's active region (moves 50-100) and refresh-to-refresh camera displacement (m)."""
    S = M.S
    res = {}
    for tag in ("Z6MFQCViBuw_L0_S6", "oLBMNvg9in8_L1_S36", "QUCTc6BB5sX_L1_S34", "2azQ1b91cZZ_L0_S60", "TbHJrupSAjP_L1_S112"):
        scan, lv = tag.rsplit("_S", 1)[0].rsplit("_L", 1)
        crd = M.card(scan, int(lv))
        cp = np.load(os.path.join(S.W, "preprocessing", "coverage", f"{crd['key']}.npz"))
        P = S.rt_to_zup_m(cp["pts"])[:, :2]
        for proto, g, root, rep in (("station", "V2K4", os.path.join(S.W, "closed_loop"), os.path.join(M.MC, "viz", "covrep")),
                                    ("grid", "G-V2K4", os.path.join(M.MC, "grid_protocol", "closed_loop"), os.path.join(M.MC, "viz", "covrep_grid"))):
            z = np.load(os.path.join(rep, g, f"{tag}.npz"))
            first, vis = z["first"], z["visible"].astype(bool)
            logs = [json.loads(x) for x in open(os.path.join(root, g, tag, "planner.jsonl"))]
            share = []
            for t in range(50, 100):
                b = np.asarray(logs[10 * (t // 10)]["refresh"]["box_m"])
                inside = np.all((P >= b[0, :2]) & (P <= b[1, :2]), axis=1)
                rem = vis & ((first < 0) | (first > t))
                share.append(float((rem & inside).sum() / max(1, rem.sum())))
            cams = [S.rt_to_zup_m(np.load(M.svc_path(g, tag, t))["cam_center"])[:2] for t in range(0, 101, 10)]
            res.setdefault(tag, {})[proto] = {"rem_in_region_50_100": float(np.mean(share)),
                                              "refresh_displacement_m": float(np.mean([np.linalg.norm(a - b) for a, b in zip(cams[:-1], cams[1:])]))}
    mean = {pr: {k: float(np.mean([res[t][pr][k] for t in res])) for k in ("rem_in_region_50_100", "refresh_displacement_m")}
            for pr in ("station", "grid")}
    return {"per_episode": res, "mean": mean}


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import t11_viz as V
    import t12_viz as V12
    fig, axs = plt.subplots(1, 3, figsize=(V.W, 3.8))
    gs = [g for g in GROUPS if g in out["horizon"]]
    for ax, (k, name) in zip(axs, [("rem_in_region_50_150", "remaining work inside V2's region\n(share, moves 50-150)"),
                                   ("best_gain_outside_share", "oracle-best candidate: share of its gain\noutside V2's region"),
                                   ("own_regret_x1e3", "one-step regret of the executed move\n(oracle best - chosen, x1e-3 C_obs)")]):
        for i, g in enumerate(gs):
            vals = list(out["horizon_per_episode"][g].values())
            ax.bar(i, out["horizon"][g][k], color=V12.COL[g], width=0.66)
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.2, 0.2, len(vals)), [v[k] for v in vals], s=9, color=V.INK2, zorder=3)
        ax.set_xticks(range(len(gs)))
        ax.set_xticklabels([V12.BASE[g] for g in gs], fontsize=9)
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(axis="y", color=V.GRID)
    fig.suptitle("T0.12-C horizon audit, 10 grid episodes (bars = mean, dots = episodes; evaluation side)", x=0.01, y=1.1, ha="left",
                 fontsize=10.5, color=V.INK2)
    V12.save(fig, "diag_c_horizon.png")
    tag = "TbHJrupSAjP_L1_S112"
    fig, axs = plt.subplots(1, 2, figsize=(V.W, 3.8))
    for g in ("G-V2K4", "G-V2K4-NV", "G-MAG-B"):
        orc = load("oracle", g, tag)
        if orc is None:
            continue
        axs[0].plot([r["remaining_in_region_share"] for r in orc], color=V12.COL[g], lw=1.8, label=V12.LAB[g])
        sc = scorer_rows("V2", g, tag)
        scl = scorer_rows("V2-level", g, tag)
        if sc:
            axs[1].plot([max(s) for s in sc], color=V12.COL[g], lw=1.6, label=f"{V12.BASE[g]}: V2 (region)")
        if scl:
            axs[1].plot([max(s) for s in scl], color=V12.COL[g], lw=1.2, ls="--", label=f"{V12.BASE[g]}: V2 (whole level, probe)")
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel("move")
    axs[0].set_title("share of the remaining work inside V2's region", loc="left", fontsize=10)
    axs[1].set_yscale("log")
    axs[1].set_xlabel("move")
    axs[1].set_title("V2's best candidate score (DistP1) at the same states", loc="left", fontsize=10)
    for ax in axs:
        ax.grid(color=V.GRID)
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"{tag}: V2 sees only its 7.8 m region; the work left is mostly outside it", x=0.01, y=1.04, ha="left", fontsize=10.5, color=V.INK2)
    V12.save(fig, "diag_c_TbHJ.png")


if __name__ == "__main__":
    main()
