"""Geometry / ROI audit F (CPU, read-only): aggregate results/rays/*.jsonl -> results/f_ray_summary.json, and select the colosseum
rays for the report -> results/top_rays_colosseum.{json,csv}.
Shares are of the D1-novel predicted value (w(z)-weighted, mean over K) that D-v2 removes, on first-layer candidate views of the
Table 1 D1 states (every 5 steps). 'occluder' = first observed cell on the centre ray (the bundle test decides, the centre cell
locates). Top rays: the 10 (state, candidate) pairs of colosseum with the largest first-layer novelty reduction (at most 3 per
episode), each with one removed ray drawn with probability proportional to its weight (value-representative; seeded in c_ray_audit); the final two-step scores of the same candidate (D1 / D-v2) are joined from the
E3 fixed-state records (same states, same replay)."""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
E3 = os.path.join(REPO, "experiments", "exp6_v3", "ablation_study", "experiment_E_stability_plateau", "E3_dv2_closed_loop")
EP = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "episodes")


def onset(c):
    return next((t for t in range(101) if c[min(t + 20, 100)] - c[t] < 0.02), None)


def main():
    a2 = json.load(open(os.path.join(AUD, "results", "a2_p1_roi.json")))
    eps = {}
    for f in sorted(glob.glob(os.path.join(AUD, "results", "rays", "*.jsonl"))):
        base = os.path.basename(f)[:-6]
        eps[base] = [json.loads(l) for l in open(f)]
    out = {"episodes": {}, "groups": {}}
    keys = None

    def agg(rows):
        nonlocal keys
        keys = keys or [k for k in rows[0]["acc"]]
        A = {k: sum(r["acc"][k] for r in rows) for k in keys}
        V = {k: sum(r["viol"][k] for r in rows) for k in rows[0]["viol"]}
        cut = max(A["cut"], 1e-9)
        inb = max(A["cut_occ_c_in"], 1e-9)
        nv = sum(r["n_legal"] for r in rows)
        d = {"states": len(rows), "views": nv, "cams_in_box": sum(r["cams"]["in_box"] for r in rows) / max(1, nv), "cams_in_p1_grid": sum(r["cams"]["in_grid"] for r in rows) / max(1, nv),
             "novel_value": A["novel"], "cut_share_of_novel": A["cut"] / max(A["novel"], 1e-9),
             "cut_cam_outside_box": A["cut_cam_out_box"] / cut, "cut_occluder_outside_box": A["cut_occ_c_out"] / cut, "cut_block_position_outside_box": A["cut_occ_b_out"] / cut,
             "cut_camout_occout": A["cut_camout_occout"] / cut, "cut_camout_occin": A["cut_camout_occin"] / cut, "cut_camin_occin": A["cut_camin_occin"] / cut,
             "cut_camin_occout": A["cut_camin_occout"] / cut,
             "gt_verified": A["cut_gt_ok"] / cut, "gt_verified_surface_outside_box": A["cut_gt_ok_surface_out"] / cut, "gt_verified_surface_inside_box": A["cut_gt_ok_surface_in"] / cut,
             "gt_not_verified_surface_behind": A["cut_gt_surface_behind"] / cut, "gt_not_verified_no_surface": A["cut_no_gt_surface"] / cut,
             "gt_verified_given_occluder_outside": (A["gt_ok_camout_occout"] + A["gt_ok_camin_occout"]) / max(A["cut_occ_c_out"], 1e-9),
             "gt_verified_given_occluder_inside": (A["gt_ok_camout_occin"] + A["gt_ok_camin_occin"]) / inb,
             "strict_observed_point": A["strict_ok"] / max(A["strict_n"], 1), "strict_observed_point_value_weighted": A["strict_w_ok"] / max(A["strict_w"], 1e-9), "strict_n": A["strict_n"],
             "inside_occluder_why": {"no_band_on_ray": A["inbox_no_band"] / inb, "band_positive_only": A["inbox_band_positive_only"] / inb,
                                     "negative_without_front_crossing": A["inbox_negative_no_crossing"] / inb, "front_crossing_present": A["inbox_crossing"] / inb},
             "true_surface_inside_roi_share_of_cut": A.get("ingt", 0.0) / cut,
             "true_surface_inside_roi_p1_state": {"no_band_at_true_surface": A.get("ingt_no_band", 0.0) / max(A.get("ingt", 0.0), 1e-9),
                                                  "band_positive_only": A.get("ingt_band_positive_only", 0.0) / max(A.get("ingt", 0.0), 1e-9),
                                                  "negative_without_front_crossing": A.get("ingt_negative_no_crossing", 0.0) / max(A.get("ingt", 0.0), 1e-9),
                                                  "front_crossing_present": A.get("ingt_crossing", 0.0) / max(A.get("ingt", 0.0), 1e-9)},
             "violations": V, "p1_unchanged_by_scoring": all(r["p1_unchanged_by_scoring"] for r in rows), "band_empty_monotone": all(r["band_empty_monotone_so_far"] for r in rows),
             "in_domain_points_are_band_min": min(r["frame"]["in_domain_points_are_band"] for r in rows if r["frame"]["in_domain_points_are_band"] is not None),
             "frame_points_outside_box_mean": float(np.mean([r["frame"]["outside_roi_box"] for r in rows])),
             "replay_value_equals_log": None}
        return d

    for base, rows in eps.items():
        out["episodes"][base] = agg(rows)
    col = {b: r for b, r in eps.items() if b.startswith("colosseum")}
    if col:
        allr = [r for rows in col.values() for r in rows]
        out["groups"]["colosseum all states"] = agg(allr)
        out["groups"]["colosseum t<20"] = agg([r for r in allr if r["t"] < 20])
        out["groups"]["colosseum t>=20"] = agg([r for r in allr if r["t"] >= 20])
        plat = []
        for b, rows in col.items():
            k = int(b.rsplit("_s", 1)[1])
            cov = json.load(open(os.path.join(EP, "V3-D1-H3-NORISK", f"colosseum_s{k}.json")))["coverage_official"]
            on = onset(cov)
            if cov[100] < 0.70 and on is not None:
                plat += [r for r in rows if r["t"] >= on]
        if plat:
            out["groups"]["colosseum plateau states"] = agg(plat)
    out["groups"]["all audited episodes"] = agg([r for rows in eps.values() for r in rows])
    json.dump(out, open(os.path.join(AUD, "results", "f_ray_summary.json"), "w"), indent=1)
    hdr = ("states", "cams_in_box", "cut_share_of_novel", "cut_cam_outside_box", "cut_occluder_outside_box", "cut_camin_occin", "cut_camout_occin", "cut_camin_occout",
           "gt_verified", "gt_verified_given_occluder_outside", "gt_verified_given_occluder_inside", "strict_observed_point")
    print("| set | " + " | ".join(hdr) + " |")
    for name, d in list(out["episodes"].items()) + list(out["groups"].items()):
        print(f"| {name} | " + " | ".join(f"{d[h]:.3f}" if isinstance(d[h], float) else str(d[h]) for h in hdr) + " |")
    g = out["groups"]["all audited episodes"]
    print("violations (all):", g["violations"], "| P1 unchanged:", g["p1_unchanged_by_scoring"], "| monotone:", g["band_empty_monotone"], "| in-domain pts are BAND (min):", g["in_domain_points_are_band_min"])
    for name, d in list(out["episodes"].items()) + list(out["groups"].items()):
        print(f"   {name}: true surface inside ROI = {d['true_surface_inside_roi_share_of_cut']:.3f} of removed value; P1 at that surface:", {k: round(v, 3) for k, v in d["true_surface_inside_roi_p1_state"].items()})
    for name in ("colosseum all states", "all audited episodes"):
        if name in out["groups"]:
            print(name, "inside-box occluders, why P1 did not stop the ray:", {k: round(v, 3) for k, v in out["groups"][name]["inside_occluder_why"].items()})

    # ---------------- top rays (colosseum)
    cand = []
    for b, rows in col.items():
        k = int(b.rsplit("_s", 1)[1])
        fx = os.path.join(E3, "results", "fixed", f"dv2_fixed_colosseum_s{k}.jsonl")
        fxr = {json.loads(l)["t"]: json.loads(l) for l in open(fx)} if os.path.exists(fx) else {}
        for r in rows:
            for v in r["views"]:
                if not v["rays"] or v["rays"][0].get("observed_point") is None:
                    continue
                f_ = fxr.get(r["t"])
                cand.append({"episode": b, "start": k, "t": r["t"], "candidate": v["j"], "novelty_before": v["value_off"], "novelty_after": v["value_on"],
                             "reduction": v["value_off"] - v["value_on"], "cam": v["cam"], "cam_in_box": v["cam_in_box"], "cam_in_p1_grid": v["cam_in_p1_grid"],
                             "view_removed_value_occluder_outside_box": v.get("removed_value_occluder_outside_box"), "view_removed_value_gt_verified": v.get("removed_value_gt_verified"),
                             "final_score_d1": f_["off"]["scores"][v["j"]] if f_ and len(f_["off"]["scores"]) > v["j"] else None,
                             "final_score_dv2": f_["dv2"]["scores"][v["j"]] if f_ and len(f_["dv2"]["scores"]) > v["j"] else None,
                             "one_step_d1_log": f_["off"]["h_mean"][v["j"]] if f_ and f_["off"].get("h_mean") else None,
                             "one_step_dv2_planner": f_["dv2"]["h_mean"][v["j"]] if f_ and f_["dv2"].get("h_mean") else None,
                             "ray": v["rays"][0]})
    cand.sort(key=lambda c: -c["reduction"])
    sel, per = [], {}
    for c in cand:
        if per.get(c["episode"], 0) >= 3 or any(c["episode"] == s["episode"] and c["t"] == s["t"] for s in sel):
            continue
        sel.append(c)
        per[c["episode"]] = per.get(c["episode"], 0) + 1
        if len(sel) == 10:
            break
    o = a2["colosseum"]
    top = {"scene": "colosseum", "roi_box": o["roi_box"], "p1_grid_rt": o["p1_grid_rt"], "p1_domain_rt": o["p1_domain_rt"], "voxel_rt": o["voxel_rt"],
           "dv2_rule": "removed iff t_pred > t_block_bundle + 1.0 (runtime units); t_block_bundle = max over the 5 bundle rays of the first observed 0.5 cell", "rays": sel}
    for c in sel:
        r = c["ray"]
        c["rule_reproduced"] = bool(r["t_pred"] > r["t_block_bundle"] + 1.0)
        c["replay_matches_planner_log"] = (abs(c["novelty_before"] - c["one_step_d1_log"]) <= 1e-6 * max(1.0, abs(c["one_step_d1_log"]))) if c["one_step_d1_log"] is not None else None
        c["after_matches_planner"] = (abs(c["novelty_after"] - c["one_step_dv2_planner"]) <= 1e-6 * max(1.0, abs(c["one_step_dv2_planner"]))) if c["one_step_dv2_planner"] is not None else None
    json.dump(top, open(os.path.join(AUD, "results", "top_rays_colosseum.json"), "w"), indent=1)
    with open(os.path.join(AUD, "results", "top_rays_colosseum.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "step", "candidate", "cam_x", "cam_y", "cam_z", "cam_in_box", "dir_x", "dir_y", "dir_z", "pred_x", "pred_y", "pred_z", "pred_in_domain",
                    "obs_x", "obs_y", "obs_z", "obs_in_box", "obs_in_domain", "t_first_observed_centre", "t_block_bundle", "t_observed_point", "t_pred", "t_true_gt",
                    "novelty_before", "novelty_after", "final_score_d1", "final_score_dv2", "rule_reproduced"])
        for c in sel:
            r = c["ray"]
            w.writerow([c["start"], c["t"], c["candidate"], *[round(x, 2) for x in c["cam"]], c["cam_in_box"], *[round(x, 4) for x in r["dir"]], *[round(x, 2) for x in r["pred_hit"]],
                        r["pred_hit_in_domain"], *[round(x, 2) for x in r["observed_point"]], r["observed_point_in_box"], r["observed_point_in_domain"],
                        round(r["t_first_observed_centre"], 2), round(r["t_block_bundle"], 2), round(r["observed_point_depth"], 2), round(r["t_pred"], 2),
                        None if r["t_true_gt"] is None else round(r["t_true_gt"], 2), round(c["novelty_before"], 1), round(c["novelty_after"], 1),
                        None if c["final_score_d1"] is None else round(c["final_score_d1"], 1), None if c["final_score_dv2"] is None else round(c["final_score_dv2"], 1), c["rule_reproduced"]])
    print("top rays:")
    for c in sel:
        r = c["ray"]
        print(f"  s{c['start']} t={c['t']} cand {c['candidate']} cam {np.round(c['cam'], 1)} in_box {c['cam_in_box']} | pred {np.round(r['pred_hit'], 1)} in_dom {r['pred_hit_in_domain']} | obs {np.round(r['observed_point'], 1)} "
              f"in_box {r['observed_point_in_box']} | depth obs {r['observed_point_depth']:.1f} block {r['t_block_bundle']:.1f} pred {r['t_pred']:.1f} gt {r['t_true_gt']} | novelty {c['novelty_before']:.0f}->{c['novelty_after']:.0f} "
              f"final {c['final_score_d1']:.0f}->{c['final_score_dv2']:.0f} | rule {c['rule_reproduced']} log {c['replay_matches_planner_log']} after {c['after_matches_planner']} "
              f"| first cell in box {r['first_observed_in_box']} t_c {r['t_first_observed_centre']:.1f} | view occluder-outside share {c['view_removed_value_occluder_outside_box']}")


if __name__ == "__main__":
    main()
