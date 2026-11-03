"""EXP Density A/B -- analysis of the 12-config fixed-state results (seen2scene env, CPU; FROZEN_PROTOCOL.md §3.1-§3.5).

Per state and config: ranking under the historical Task 2 choice rule (ab_load.pick_of; same as e2_analyze), oracle gains
(oracle.npz gain, pp; identical for all configs), score scale; per state and (checkpoint, inference density): G1 rates on
the ray sets of §3.1; per config: scoring-mask rates. Aggregation state -> episode -> scene (ab_load.hier_mean); scene-level
bootstrap (primary) and episode-level bootstrap (secondary) of the means and of the pre-registered paired differences
K1-K5. Indoor (VALSEL) and outdoor (Dev-3) separately.
Outputs results/fixed_summary.json, results/fixed_states.csv
Usage: python da_analyze_fixed.py
"""
import csv
import glob
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts"))
import e2_analyze as EA  # noqa: E402

EP, P, A, AL = EA.EP, EA.P, EA.A, EA.AL
ORIG_W = EA.ORIG_W
FIX = os.path.join(REPO, "work_dirs", "exp_density_ab", "fixed")
CK = ("B2", "E2S", "E2D")
DN = ("S10", "D100")
CFGS = [f"{m}|{i}|{s}" for m in CK for i in DN for s in DN]
TAU = EA.TAU
SETS = ("unk_common", "unk_only10", "unk_only100", "unk_own", "unk_union", "all", "new_common")


def pairs_k():
    """Pre-registered paired comparisons (hi, lo, family)."""
    out = []
    for m in CK:
        for s in DN:
            out.append((f"{m}|D100|{s}", f"{m}|S10|{s}", f"K1/K4 inference D100-S10 ({m}, score {s})"))
        for i in DN:
            out.append((f"{m}|{i}|D100", f"{m}|{i}|S10", f"K2/K4 scoring D100-S10 ({m}, inference {i})"))
    for i, s in itertools.product(DN, DN):
        out.append((f"E2D|{i}|{s}", f"E2S|{i}|{s}", f"K3 E2D-E2S (inf {i}, score {s})"))
        out.append((f"E2S|{i}|{s}", f"B2|{i}|{s}", f"K5 E2S-B2 (inf {i}, score {s})"))
        out.append((f"E2D|{i}|{s}", f"B2|{i}|{s}", f"E2D-B2 (inf {i}, score {s})"))
    out.append(("B2|D100|S10", "B2|S10|S10", "historical B2 - B0"))
    out.append(("E2D|D100|S10", "E2S|S10|S10", "historical E2D - E2S (matched train/test)"))
    out.append(("E2D|D100|S10", "B2|D100|S10", "historical E2D - B2"))
    return out


def ratio(a, b):
    return float(a) / float(b) if b else None


def load_rows():
    out = []
    for sset, dname in (("valsel", "indoor"), ("dev3", "outdoor")):
        rows = []
        for f in sorted(glob.glob(os.path.join(FIX, sset, "*", "*.jsonl"))):
            rows += [json.loads(x) for x in open(f)]
        if sset == "valsel":
            man = {s["sid"]: s for s in json.load(open(os.path.join(EP.W, "valsel", "manifests", "states_valsel.json")))["states"]}
        else:
            m0 = {s["sid"]: s for s in json.load(open(os.path.join(ORIG_W, "manifests", "states_outdoor.json")))["states"]}
            m2 = {s["sid"]: s for s in json.load(open(os.path.join(P.ROOT_B2, "manifests", "states_outdoor.json")))["states"]}
            man = {r["sid"]: (m0 if r["stratum"] == "B0" else m2)[r["sid"]] for r in rows}
        for r in rows:
            st = man[r["sid"]]
            C = np.asarray(st["cand"], int)
            gain = np.asarray(r["oracle_gain_pp"], float)
            g = gain[C]
            src = st["sources"][0] if sset == "valsel" else next(e for e in st["sources"] if e["source"] == "V2M")
            rec = {k: np.asarray(x) for k, x in np.load(os.path.join(EP.REPO, src["svc"])).items() if k in ("legal_idx", "executed_pose_idx_history")}
            loc = {int(c): i for i, c in enumerate(C)}
            x = {"set": sset, "dataset": dname, "sid": r["sid"], "scene": r["scene"], "start": r["start"], "t": r["t"], "phase": r["phase"],
                 "src": r["stratum"], "tag": r["tag"], "oracle_max": float(g.max()), "n_cand": int(len(C)), "n_legal": len(r["scores"]["B2|S10|S10"]),
                 "repro_all_ok": bool(all(r["repro"].values())), "n_repro": len(r["repro"]), "band_dil_eq": bool(r["band_dil_recompute_equal"])}
            picks = {}
            for c in CFGS:
                s = np.asarray([float(np.mean(k)) for k in r["scores"][c]])
                pk = AL.pick_of(dname, st, s, rec, "V2M")[0]
                picks[c] = int(pk)
                gp = g[loc[pk]]
                x.update({f"{c}#top1": float(gp >= g.max() - TAU), f"{c}#sel_gain": float(gp), f"{c}#regret": float(g.max() - gp),
                          f"{c}#rho": EA.corr(EA.spearmanr, s[C], g), f"{c}#tau": EA.corr(EA.kendalltau, s[C], g),
                          f"{c}#score_mean": float(s.mean()), f"{c}#score_zero_frac": float((s <= 0).mean()),
                          f"{c}#pick": int(pk)})
                mk = r["masks"][c]
                x[f"{c}#reward_frac"] = ratio(mk["reward"], mk["hit"])
                x[f"{c}#reward_truth_new_frac"] = ratio(mk["reward_truth_new"], mk["reward"])
                x[f"{c}#reward_truth_old_frac"] = ratio(mk["reward_truth_old"], mk["reward"])
                x[f"{c}#reward_no_gt_frac"] = ratio(mk["reward_no_gt"], mk["reward"])
                x[f"{c}#reward_accurate_frac"] = ratio(mk["reward_accurate"], mk["reward"])
                x[f"{c}#excl_truth_new_per_hit"] = ratio(mk["excl_truth_new"], mk["hit"])
                x[f"{c}#reward_per_cand"] = mk["reward"] / max(x["n_cand"], 1) / 4.0
            for a, b in itertools.combinations(CFGS, 2):
                x[f"chg#{a}#{b}"] = float(picks[a] != picks[b])
            for xs in ("10", "100"):
                gm = r["masks"][f"GT|{xs}"]
                x[f"GT|{xs}#excl_truth_new_frac"] = ratio(gm["excl_truth_new"], gm["truth_new_hit"])
                x[f"GT|{xs}#reward_truth_old_frac"] = ratio(gm["reward_truth_old"], gm["hit"])
                x[f"GT|{xs}#reward_frac"] = ratio(gm["hit"] - gm["excl"], gm["hit"])
            rc = r["ray_counts"]
            for k in ("unk10", "unk100", "unk_common", "unk_only10", "unk_only100", "truth_new", "unk10_truth_new", "unk100_truth_new"):
                x[f"rays#{k}_frac"] = rc[k] / rc["rays"]
            x["rays#only10_of_unk10"] = ratio(rc["unk_only10"], rc["unk10"])
            for m in CK:
                for xi, nm in (("10", "S10"), ("100", "D100")):
                    for sname in SETS:
                        q = r["g1"][f"{m}|{xi}"][sname]
                        pre = f"{m}|{nm}#{sname}"
                        x[f"{pre}#mae"] = ratio(q["sae"], q["tp"])
                        x[f"{pre}#hitacc"] = ratio(q["tp"] + q["tn"], q["n"])
                        x[f"{pre}#fp_rate"] = ratio(q["fp"], q["fp"] + q["tn"])
                        x[f"{pre}#fn_rate"] = ratio(q["fn"], q["gt"])
                        x[f"{pre}#hit_rate"] = ratio(q["ph"], q["n"])
                        x[f"{pre}#acc_first"] = ratio(q["acc"], q["gt"])
                        x[f"{pre}#n"] = q["n"] / 4.0
                    pe = r["pred_e2"][f"{m}|{xi}"]
                    x[f"{m}|{nm}#spread_rel_std"] = ratio(pe.get("rel_std_sum", 0.0), pe["n_allhit"])
                    x[f"{m}|{nm}#spread_err_rho"] = pe.get("rho_mean")
                    x[f"{m}|{nm}#hit_dis_unk"] = ratio(pe["hit_dis_unk"], pe["n_unk_rays"])
            out.append(x)
    return out


def boot_ep(vals, n=10000, seed=0):
    g = np.asarray(list(vals.values()), float)
    if len(g) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    bs = g[rng.integers(0, len(g), (n, len(g)))].mean(1)
    return {"mean": float(g.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "n": int(len(g))}


def ep_mean(rr, key):
    ep = {}
    for r in rr:
        v = r.get(key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        ep.setdefault(AL.episode_of(r), []).append(float(v))
    return {k: float(np.mean(v)) for k, v in ep.items()}


def summarize(recs):
    keys = sorted({k for r in recs for k in r if "#" in k and not k.endswith("#pick")})
    S = {"n_states": len(recs), "n_episodes": len({AL.episode_of(r) for r in recs}), "n_scenes": len({r["scene"] for r in recs}),
         "repro_all_ok": int(sum(r["repro_all_ok"] for r in recs)), "n_repro_checks": int(sum(r["n_repro"] for r in recs)),
         "band_dil_eq": int(sum(r["band_dil_eq"] for r in recs)), "oracle_max": AL.boot(AL.hier_mean(recs, "oracle_max")),
         "means": {}, "paired": {}}
    for k in keys:
        S["means"][k] = AL.boot(AL.hier_mean(recs, k))
    metrics_cfg = sorted({k.split("#", 1)[1] for k in keys if k.split("#")[0] in CFGS})
    for hi, lo, fam in pairs_k():
        for met in metrics_cfg:
            ha, hb = AL.hier_mean(recs, f"{hi}#{met}"), AL.hier_mean(recs, f"{lo}#{met}")
            dd = {s: ha[s] - hb[s] for s in ha if s in hb}
            ea, eb = ep_mean(recs, f"{hi}#{met}"), ep_mean(recs, f"{lo}#{met}")
            de = {s: ea[s] - eb[s] for s in ea if s in eb}
            bo = AL.boot(dd)
            bo.update({"wins": int(sum(v > 1e-12 for v in dd.values())), "losses": int(sum(v < -1e-12 for v in dd.values())), "per_scene": dd,
                       "episode": boot_ep(de), "ep_wins": int(sum(v > 1e-12 for v in de.values())), "ep_losses": int(sum(v < -1e-12 for v in de.values())),
                       "family": fam})
            S["paired"][f"{hi} - {lo}|{met}"] = bo
    # G1 paired: same checkpoint, inference D100 - S10 on the same ray set; E2D - E2S at the same inference density
    g1pairs = [(f"{m}|D100", f"{m}|S10") for m in CK] + [(f"E2D|{i}", f"E2S|{i}") for i in DN] + [(f"E2S|{i}", f"B2|{i}") for i in DN] \
        + [("E2D|D100", "B2|D100"), ("E2D|D100", "E2S|S10")]
    g1mets = sorted({k.split("#", 1)[1] for k in keys if k.split("#")[0] in [f"{m}|{n}" for m in CK for n in DN]})
    for hi, lo in g1pairs:
        for met in g1mets:
            ha, hb = AL.hier_mean(recs, f"{hi}#{met}"), AL.hier_mean(recs, f"{lo}#{met}")
            dd = {s: ha[s] - hb[s] for s in ha if s in hb}
            ea, eb = ep_mean(recs, f"{hi}#{met}"), ep_mean(recs, f"{lo}#{met}")
            de = {s: ea[s] - eb[s] for s in ea if s in eb}
            bo = AL.boot(dd)
            bo.update({"wins": int(sum(v > 1e-12 for v in dd.values())), "losses": int(sum(v < -1e-12 for v in dd.values())),
                       "episode": boot_ep(de), "ep_wins": int(sum(v > 1e-12 for v in de.values())), "ep_losses": int(sum(v < -1e-12 for v in de.values()))})
            S["paired"][f"G1:{hi} - {lo}|{met}"] = bo
    return S


def main():
    recs = load_rows()
    out = {}
    for d in ("indoor", "outdoor"):
        rr = [r for r in recs if r["dataset"] == d]
        out[d] = {"all": summarize(rr)}
        for ph in ("early", "middle", "late"):
            sub = [r for r in rr if r["phase"] == ph]
            if sub:
                out[d][ph] = {"n_states": len(sub), "means": {k: AL.boot(AL.hier_mean(sub, k)) for k in
                                                              [f"{c}#{m}" for c in CFGS for m in ("top1", "sel_gain", "rho")]}}
    os.makedirs(os.path.join(EXP, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(EXP, "results", "fixed_summary.json"), "w"), indent=1, default=float)
    cols = ["set", "dataset", "sid", "scene", "start", "t", "phase", "src", "tag", "oracle_max", "n_cand", "n_legal", "repro_all_ok"]
    cols += [f"{c}#{m}" for c in CFGS for m in ("pick", "top1", "sel_gain", "regret", "rho", "score_mean", "score_zero_frac", "reward_frac")]
    cols += [f"{m}|{n}#{s}#{q}" for m in CK for n in DN for s in ("unk_common", "unk_only10", "unk_own") for q in ("mae", "hitacc", "fn_rate", "acc_first")]
    with open(os.path.join(EXP, "results", "fixed_states.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            w.writerow([r.get(c) for c in cols])
    for d in ("indoor", "outdoor"):
        S = out[d]["all"]
        print(d, "states", S["n_states"], "episodes", S["n_episodes"], "scenes", S["n_scenes"], "repro ok", S["repro_all_ok"], "/", S["n_states"],
              "checks", S["n_repro_checks"], "band_dil_eq", S["band_dil_eq"])
        for c in CFGS:
            M = S["means"]
            print(f"  {c:16s} top1 {M[c + '#top1']['mean']:.3f} sel_gain {M[c + '#sel_gain']['mean']:.3f} rho {M[c + '#rho']['mean']:.3f}")


if __name__ == "__main__":
    main()
