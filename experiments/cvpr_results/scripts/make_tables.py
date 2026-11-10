"""CVPR overnight -- Table 1 (All-14 benchmark), Table 2 (splits), Table 3 (K x planning depth). CPU only. python make_tables.py
Reads only manifest/episodes.csv (COMPLETE rows = canonical records that passed cvpr_lib.check_record in build_manifest.py; run
build_manifest.py first); writes experiments/cvpr_results/fig/table{1,2,3}/.
A method enters an aggregate only for scenes with all 5 starts; a group value is printed only when every scene of the group is
complete for that method, otherwise the paper table shows a dash and the CSV carries the partial value with its scene count."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402
import render as RD  # noqa: E402

PAPER = ["AUC@20", "AUC@50", "AUC@100", "Cov@20", "Final"]
ALLM = sorted({f"{k}@{T}" for T in L.T_FULL for k in ("AUC", "Cov")} | {"Final"}, key=lambda n: (n != "Final", n.split("@")[0], int(n.split("@")[1]) if "@" in n else 0))


def valid(method):
    """COMPLETE records of a method, read from manifest/episodes.csv (the immutable intermediate written by build_manifest.py after
    the per-record validation); a second light check is applied here, anything that fails it is dropped and counted."""
    E, bad = {}, 0
    for k, r in L.load_manifest(method).items():
        ok, _, _ = L.check_record(r, deep=False)
        if ok and k[0] in L.SPLIT and 0 <= k[1] < L.N_START:
            E[k] = r
        else:
            bad += 1
    return E, bad


def status_of(m, E):
    if L.METHODS[m].get("blocked"):
        return "BLOCKED"
    return "COMPLETE" if len(E) == 70 else ("PARTIAL" if E else "NOT_RUN")


def table1():
    out = os.path.join(L.FIG, "table1")
    data = {m: valid(m)[0] for m in L.TABLE1}
    tabs = {m: L.scene_table(data[m], ALLM) for m in L.TABLE1}
    rows_full, rows_paper, bold = [], [], set()
    for m in L.TABLE1:
        d, E, tab = L.METHODS[m], data[m], tabs[m]
        st = status_of(m, E)
        vals = [L.group_mean(tab, L.SCENE_ORDER, n)[0] for n in ALLM]
        rows_full.append([m, d["display"], d["source"], st, len(E), 70, len(tab), 14] + ["" if v is None else repr(v) for v in vals])
        disp = d["display"] + ("†" if "adapted" in d["source"] else "")
        if st == "COMPLETE":
            rows_paper.append([disp] + [L.fmt(L.group_mean(tab, L.SCENE_ORDER, n)[0]) for n in PAPER])
        else:
            rows_paper.append([disp + (f" ({len(E)}/70)" if st == "PARTIAL" else "")] + ["—"] * len(PAPER))
    comp = [i for i, m in enumerate(L.TABLE1) if status_of(m, data[m]) == "COMPLETE"]
    for j, n in enumerate(PAPER):
        if comp:
            best = max(comp, key=lambda i: L.group_mean(tabs[L.TABLE1[i]], L.SCENE_ORDER, n)[0])
            bold.add((best, j + 1))
    L.write_csv(os.path.join(out, "table1_all14_full.csv"), ["method_id", "method", "source", "status", "n_completed", "n_expected", "n_scenes_complete", "n_scenes_expected"] + ALLM, rows_full)
    L.write_csv(os.path.join(out, "table1_paper.csv"), ["Method"] + PAPER, rows_paper)
    # every step T = 1..100 (long + wide)
    long_rows, wide_a, wide_c = [], [], []
    for m in L.TABLE1:
        mc, per = L.mean_curve(data[m], L.SCENE_ORDER)
        if mc is None or len(per) < 14:
            continue
        a = [float(np.mean([L.auc(np.mean([data[m][(s, k)]["coverage_official"] for k in range(5)], 0), T) for s in L.SCENE_ORDER])) for T in range(1, 101)]
        # AUC is linear in the curve, so the scene-equal mean of the per-start AUC equals the AUC of the scene-mean curve
        for T in range(1, 101):
            long_rows.append([m, T, repr(a[T - 1]), repr(float(mc[T]))])
        wide_a.append([m] + [repr(x) for x in a])
        wide_c.append([m, repr(float(mc[0]))] + [repr(float(mc[T])) for T in range(1, 101)])
    L.write_csv(os.path.join(out, "table1_auc_cov_long.csv"), ["method_id", "T", "AUC@T", "Cov@T"], long_rows)
    L.write_csv(os.path.join(out, "table1_auc_wide.csv"), ["method_id"] + [f"AUC@{T}" for T in range(1, 101)], wide_a)
    L.write_csv(os.path.join(out, "table1_cov_wide.csv"), ["method_id"] + [f"Cov@{T}" for T in range(0, 101)], wide_c)
    # paired scene-level differences of Ours vs every complete baseline
    pr = []
    for b in L.TABLE1:
        if b == "V3-T0-D1" or status_of(b, data[b]) != "COMPLETE":
            continue
        for n in PAPER:
            m_, lo, hi, ns, w, l = L.paired(tabs["V3-T0-D1"], tabs[b], L.SCENE_ORDER, n)
            pr.append(["V3-T0-D1", b, "All-14", n, repr(m_), repr(lo), repr(hi), ns, w, l])
    L.write_csv(os.path.join(out, "table1_paired_ours_minus_baseline.csv"), ["method_a", "method_b", "group", "metric", "mean_diff", "ci95_lo", "ci95_hi", "n_scenes", "scene_wins", "scene_losses"], pr)
    # provenance
    pa = L._SR["meta"]["paper_average_14_without_sofia"]
    pkey = {"RANDOM": "random_walk", "SCONE": "scone", "MACARONS": "macarons", "FISHERRF": "fisherrf", "MAG-B": "magician"}
    impl = {"RANDOM": "experiments/exp3b_macaronspp_transfer/m_t7_sim.py --method random (m_t6_common.uniform_choice, seed 0)", "MACARONS": "experiments/exp3b_macaronspp_transfer/m_t7_macarons.py (adapter repairs R1-R8, seed 0)",
            "MAG-B": "experiments/exp5_future_view/task2/scripts/t24_mag_out_planner.py (official MAGICIAN core e697009, beam 10x10)",
            "V3-T0-D1": "experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/sim_out_d.py --policy d1 (V3-T0 EMA@40k)"}
    notes = {"RANDOM": "our frozen definition (EXP3b): uniform choice over the simulator legal list; the MAGICIAN paper's random-walk code is not released, so this is not the paper's implementation",
             "MACARONS": "authors' weights trained_macarons.pth (trained on 8 of the 14 evaluation scenes); native planner randomness (single seed 0); EXP3b adapter, not the authors' test script",
             "MAG-B": "no weights of its own: uses trained_macarons.pth; reproduced with the official code under the frozen protocol; paper-reported numbers differ slightly (protocol_audit)",
             "V3-T0-D1": "T0 weights trained on Train-6 (minus S-c) + MP3D; Dev-5 / Test-3 never used for training or checkpoint selection"}
    pv = []
    for m in L.TABLE1:
        d, st = L.METHODS[m], status_of(m, data[m])
        same = "yes" if not d.get("blocked") else "n/a"
        pv.append([m, d["display"], st, f"{len(data[m])}/70", d["source"], impl.get(m, ""), d["ckpt"], same, same, same, same, same,
                   "deterministic (seeded)" if m in ("RANDOM", "V3-T0-D1") else ("native randomness, single run" if m in ("MACARONS", "MAG-B") else ""),
                   pa["auc"].get(pkey.get(m, ""), ""), pa["final"].get(pkey.get(m, ""), ""), d.get("blocked") or notes.get(m, "")])
    L.write_csv(os.path.join(out, "provenance.csv"), ["method_id", "method", "status", "episodes", "source", "implementation", "weights_sha256", "same_simulator", "same_5_official_starts", "same_legal_action_list",
                                                     "same_official_coverage_replay", "same_100_step_budget", "randomness", "paper_reported_AUC@100_14scenes_NOT_same_protocol",
                                                     "paper_reported_Final_14scenes_NOT_same_protocol", "notes"], pv)
    n_note = ["All-14: 14 Macarons++ scenes x 5 official starts x 100 steps, official MAGICIAN coverage, scene-equal mean. Bold = best among the methods re-evaluated under this protocol.",
              "† adapted implementation under the frozen protocol (see provenance.csv).  — = no runnable implementation under this protocol (BLOCKED) or not complete; paper-reported numbers are never mixed in."]
    RD.table(os.path.join(out, "table1"), ["Method"] + PAPER, rows_paper, bold=bold, notes=n_note, col_w=[2.6] + [0.85] * len(PAPER))
    RD.tex(os.path.join(out, "table1.tex"), ["Method"] + PAPER, rows_paper, bold=bold, caption="Main benchmark on Macarons++ (All-14: 14 scenes x 5 starts, 100 steps, official coverage; scene-equal means). "
           "† adapted implementation under the same frozen protocol; -- not available under this protocol.", label="tab:main", notes=n_note)
    return {m: (status_of(m, data[m]), len(data[m])) for m in L.TABLE1}, rows_paper, pr


def table2():
    out = os.path.join(L.FIG, "table2")
    mets = ["AUC@20", "AUC@50", "AUC@100", "Cov@20", "Final"]
    show = ["AUC@20", "AUC@50", "AUC@100", "Final"]
    meth = [m for m in ("RANDOM", "MACARONS", "MAG-B", "V3-T0-D1") if len(valid(m)[0]) == 70]
    tabs = {m: L.scene_table(valid(m)[0], mets) for m in meth}
    rows, paper, bold, rules = [], [], set(), []
    for g in ("Train-6", "Dev-5", "Test-3", "Dev+Test-8", "All-14"):
        ss = L.GROUPS[g]
        for m in meth:
            rows.append([g, len(ss), m, L.METHODS[m]["display"], "value"] + [repr(L.group_mean(tabs[m], ss, n)[0]) for n in mets] + [""] * 3)
        for b in [m for m in meth if m != "V3-T0-D1"]:
            for n in mets:
                m_, lo, hi, ns, w, l = L.paired(tabs["V3-T0-D1"], tabs[b], ss, n)
                rows.append([g, len(ss), f"V3-T0-D1 minus {b}", "paired scene difference", n, repr(m_), repr(lo), repr(hi), "", "", ns, w, l])
        if g != "All-14":
            i0 = len(paper)
            v = {m: [L.group_mean(tabs[m], ss, n)[0] for n in show] for m in ("MAG-B", "V3-T0-D1")}
            paper.append([g, "MAGICIAN"] + [L.fmt(x) for x in v["MAG-B"]])
            paper.append(["", "Ours (V3-T0 + D1)"] + [L.fmt(x) for x in v["V3-T0-D1"]])
            dd = [L.paired(tabs["V3-T0-D1"], tabs["MAG-B"], ss, n) for n in show]
            paper.append(["", "Δ (Ours − MAGICIAN) [95% CI]"] + [f"{L.fmt(d[0], sign=True)} [{L.fmt(d[1], sign=True)}, {L.fmt(d[2], sign=True)}]" for d in dd])
            for j in range(len(show)):
                bold.add((i0 + (1 if v["V3-T0-D1"][j] > v["MAG-B"][j] else 0), j + 2))
            rules.append(len(paper) - 1)
    L.write_csv(os.path.join(out, "table2_splits.csv"), ["split", "n_scenes", "method_id", "method", "row_kind_or_metric"] + mets + ["n_scenes_paired", "scene_wins", "scene_losses"],
                [r if r[4] == "value" else r[:5] + r[5:8] + ["", ""] + r[10:] for r in rows])
    # split manifest (provenance of every scene)
    sm = []
    for s in L.SCENE_ORDER:
        sc = L.SCENES[s]
        st = sc.get("official_starts_in_training_trajectories") or {}
        sm.append([s, sc["official_name"], L.SPLIT[s], sc["used_to_train"]["V3_T0_T1"], "S-c held-out trajectories of this scene" if L.SPLIT[s] == "train" else "no",
                   json.dumps(st.get("official_starts_with_trajectories", [])) if st else "none", "frozen a priori (H-v1 / D-v1); D1 picked among D1-D3 on Dev-5" + (" (this scene)" if L.SPLIT[s] == "dev" else ""),
                   "yes (All-14 rows of H / D / T1 / T0 planners were inspected before this table)", sc["macarons_weights_role"]["role"]])
    L.write_csv(os.path.join(out, "split_manifest.csv"), ["scene_id", "official_name", "split", "T0_trained_on_scene", "T0_checkpoint_selection_data", "official_starts_that_start_a_T0_training_trajectory",
                                                         "planner_selection", "results_on_scene_seen_before_this_table", "MACARONS_MAGICIAN_weights_role_of_scene"], sm)
    n_note = ["Same episodes as Table 1. Splits are those of our model (T0 was trained on Train-6 only; Dev-5 / Test-3 were never used for training or checkpoint selection).",
              "They are 'T0-held-out' scenes, not scenes unseen by every method: MAGICIAN's (MACARONS) weights were trained on 8 of the 14 scenes (split_manifest.csv), and",
              "Test-3 results of earlier planner variants had been inspected, so Test-3 is not a blind test. Δ: scene-paired difference, scene bootstrap 95% CI (few scenes: wide)."]
    RD.table(os.path.join(out, "table2"), ["Split", "Method"] + show, paper, bold=bold, notes=n_note, col_w=[1.0, 2.4] + [1.75] * len(show), rules_after=rules[:-1], align=["left", "left"] + ["center"] * len(show))
    RD.tex(os.path.join(out, "table2.tex"), ["Split", "Method"] + show, paper, bold=bold, caption="Performance on T0-held-out scenes (same episodes as Table 1; scene-equal means; Δ = scene-paired difference with scene-bootstrap 95% CI).",
           label="tab:splits", notes=n_note, rules_after=rules[:-1], colspec="ll" + "c" * len(show))
    return paper


def table3():
    out = os.path.join(L.FIG, "table3")
    cells = [("V3-T0-K1D1", 1, 1), ("V3-T0-K4D1", 4, 1), ("V3-T0-K1D2", 1, 2), ("V3-T0-D1", 4, 2)]
    mets = PAPER
    data = {m: valid(m)[0] for m, _, _ in cells}
    tabs = {m: L.scene_table(data[m], mets) for m, _, _ in cells}
    common = [s for s in L.SCENE_ORDER if all(s in tabs[m] for m, _, _ in cells)]
    full = len(common) == 14
    label = "All-14" if full else f"preliminary subset ({len(common)} of 14 scenes complete in all four cells)"
    rows, paper, man = [], [], []
    for m, K, d in cells:
        E = data[m]
        vals = [L.group_mean(tabs[m], common, n)[0] if common else None for n in mets]
        ds = []
        for r in E.values():
            try:
                ep = json.load(open(os.path.join(L.REPO, r["episode_file"])))
                ds += [x["planner"]["decision_s"] for x in ep["steps"] if x.get("planner") and "decision_s" in x["planner"]]
            except Exception:
                pass
        dec = float(np.mean(ds)) if ds else None
        rows.append([m, K, d, len(E), 70, len(tabs[m]), label] + ["" if v is None else repr(v) for v in vals] + ["" if dec is None else repr(dec)])
        paper.append([f"K = {K}", f"{d}" + (" (Ours)" if m == "V3-T0-D1" else ""), f"{len(E)}/70"] + [L.fmt(v) for v in vals])
        hs = {}
        for r in list(E.values())[:70]:
            try:
                pl = json.load(open(os.path.join(L.REPO, r["episode_file"])))["planner_hello"][0]["planner"]
                hs.setdefault("d_cfg", set()).add(pl["d_policy"]["config_sha256"]); hs.setdefault("d_pol", set()).add(pl["d_policy"]["d_policy_file_sha256"])
                hs.setdefault("h_pol", set()).add(pl["d_policy"]["h_policy_file_sha256"]); hs.setdefault("kd", set()).add((pl.get("kd") or {}).get("kd_policy_file_sha256", "frozen sim_out_d.py path (no kd wrapper)"))
                hs.setdefault("K", set()).add(pl["K"]); hs.setdefault("depth", set()).add(pl["d_policy"]["policy_cfg"]["depth"]); hs.setdefault("lam", set()).add(pl["d_policy"]["policy_cfg"]["lambda_risk"])
            except Exception:
                pass
        man.append([m, K, d, 0, len(E), 70, len(tabs[m]), L.T0_SHA] + [";".join(sorted(str(x) for x in hs.get(k, []))) for k in ("K", "depth", "lam", "d_cfg", "d_pol", "h_pol", "kd")])
    # effects on the common scenes (paired, scene bootstrap)
    eff = []
    if common:
        T = tabs
        for n in mets:
            a = {m: np.array([T[m][s][n] for s in common]) for m, _, _ in cells}
            defs = [("K: 1 -> 4 at depth 1", a["V3-T0-K4D1"] - a["V3-T0-K1D1"]), ("K: 1 -> 4 at depth 2", a["V3-T0-D1"] - a["V3-T0-K1D2"]),
                    ("depth: 1 -> 2 at K = 1", a["V3-T0-K1D2"] - a["V3-T0-K1D1"]), ("depth: 1 -> 2 at K = 4", a["V3-T0-D1"] - a["V3-T0-K4D1"]),
                    ("main effect K (mean over depths)", (a["V3-T0-K4D1"] - a["V3-T0-K1D1"] + a["V3-T0-D1"] - a["V3-T0-K1D2"]) / 2),
                    ("main effect depth (mean over K)", (a["V3-T0-K1D2"] - a["V3-T0-K1D1"] + a["V3-T0-D1"] - a["V3-T0-K4D1"]) / 2),
                    ("interaction (K effect at depth 2 minus at depth 1)", (a["V3-T0-D1"] - a["V3-T0-K1D2"]) - (a["V3-T0-K4D1"] - a["V3-T0-K1D1"]))]
            for name, d in defs:
                m_, lo, hi = L.boot_ci(d)
                eff.append([label, name, n, repr(m_), repr(lo), repr(hi), len(common), int((d > 0).sum()), int((d < 0).sum())])
    L.write_csv(os.path.join(out, "table3_effects.csv"), ["scenes", "contrast", "metric", "mean_diff", "ci95_lo", "ci95_hi", "n_scenes", "scene_wins", "scene_losses"], eff)
    # K = 1 noise-set replicates (noise maps 0..3 of the K = 4 set), complete ones only
    noise = []
    for d in (1, 2):
        ids = [f"V3-T0-K1D{d}"] + [f"V3-T0-K1D{d}-N{j}" for j in (1, 2, 3)]
        tn = {}
        for j, m in enumerate(ids):
            E = valid(m)[0]
            man.append([m, 1, d, j, len(E), 70, len(L.scene_table(E, mets)), L.T0_SHA] + [""] * 7) if j else None
            if len(E) == 70:
                tn[j] = L.scene_table(E, mets)
        k4 = tabs["V3-T0-K4D1" if d == 1 else "V3-T0-D1"]
        for n in mets:
            v = {j: L.group_mean(t, L.SCENE_ORDER, n)[0] for j, t in tn.items()}
            if len(v) >= 2 and len(k4) == 14:
                per_scene_mean = np.mean([[tn[j][s][n] for s in L.SCENE_ORDER] for j in tn], 0)
                dm, lo, hi = L.boot_ci(np.array([k4[s][n] for s in L.SCENE_ORDER]) - per_scene_mean)
                noise.append([d, n, len(v), ";".join(str(j) for j in sorted(v)), ";".join(repr(v[j]) for j in sorted(v)), repr(float(np.mean(list(v.values())))), repr(float(np.std(list(v.values()), ddof=1))),
                              repr(float(min(v.values()))), repr(float(max(v.values()))), repr(L.group_mean(k4, L.SCENE_ORDER, n)[0]), repr(dm), repr(lo), repr(hi)])
    L.write_csv(os.path.join(out, "table3_k1_noise_replicates.csv"), ["depth", "metric", "n_noise_sets_complete", "noise_maps", "all14_value_per_noise_map", "mean_over_noise_maps", "sd_over_noise_maps(ddof1)",
                                                                     "min", "max", "K4_value", "K4_minus_mean_K1(scene-paired)", "ci95_lo", "ci95_hi"], noise)
    L.write_csv(os.path.join(out, "table3_ablation.csv"), ["method_id", "K", "planning_depth", "n_completed", "n_expected", "n_scenes_complete", "scenes_used"] + mets + ["planner_decision_s_per_step_shared_gpu"], rows)
    L.write_csv(os.path.join(out, "ablation_manifest.csv"), ["method_id", "K", "planning_depth", "noise_map_offset", "n_completed", "n_expected", "n_scenes_complete", "checkpoint_sha256", "K_in_planner_handshake",
                                                            "depth_in_planner_handshake", "lambda_risk", "d_config_sha256", "d_policy_sha256", "h_policy_sha256", "kd_policy_sha256"], man)
    bold = set()
    if common:
        for j, n in enumerate(mets):
            vals = [L.group_mean(tabs[m], common, n)[0] for m, _, _ in cells]
            bold.add((int(np.argmax(vals)), j + 3))
    n_note = [f"Scenes used: {label}. Same frozen V3-T0 checkpoint, simulator, candidates, noise keys, value_cov (eps 1 m, 1 m cells), lambda 0 in every row; only K and the planning depth change.",
              "Depth 1 = one-step value_cov averaged over the K samples (not the DistP1 score). K = 1 uses noise map 0 of the K = 4 set. Effects with scene-bootstrap CIs: table3_effects.csv."]
    RD.table(os.path.join(out, "table3"), ["Samples", "Plan depth", "Episodes"] + mets, paper, bold=bold, notes=n_note, col_w=[0.9, 1.25, 0.95] + [0.85] * len(mets), rules_after=[1],
             align=["left", "left", "center"] + ["center"] * len(mets))
    RD.tex(os.path.join(out, "table3.tex"), ["Samples", "Plan depth", "Episodes"] + mets, paper, bold=bold, caption=f"K x planning-depth ablation on the frozen V3-T0 ({label}; scene-equal means).", label="tab:kdepth",
           notes=n_note, rules_after=[1], colspec="llc" + "c" * len(mets))
    return label, paper, eff, noise


if __name__ == "__main__":
    st, p1, pr = table1()
    print("TABLE 1", st)
    for r in p1:
        print("  ", " | ".join(r))
    for r in pr:
        if r[3] in ("AUC@20", "AUC@100", "Final"):
            print(f"   paired {r[0]} - {r[1]} {r[3]}: {float(r[4]):+.3f} [{float(r[5]):+.3f}, {float(r[6]):+.3f}] wins {r[8]}/{r[7]}")
    print("TABLE 2")
    for r in table2():
        print("  ", " | ".join(r))
    lab, p3, eff, noise = table3()
    print("TABLE 3 --", lab)
    for r in p3:
        print("  ", " | ".join(r))
    for r in eff:
        if r[2] in ("AUC@20", "AUC@100", "Final"):
            print(f"   {r[1]:52s} {r[2]:8s} {float(r[3]):+.4f} [{float(r[4]):+.4f}, {float(r[5]):+.4f}] wins {r[7]}/{r[6]}")
    for r in noise:
        print("   noise", r[:4], "mean", r[5][:6], "sd", r[6][:7], "K4", r[9][:6], "K4-K1", r[10][:7], "[", r[11][:7], r[12][:7], "]")
