"""E2 -- aggregate geom_audit outputs -> results/audit_summary.json (+ printed table). Shares are of the value that D1's
value_cov counts as novel (weights w(z), sample-averaged), summed over states; groups: colosseum low-plateau episode in its
plateau (t >= onset), colosseum other states, other scenes t < 20 / t >= 20; separately for all candidates and for the
executed candidate only. Recurrence = share of the executed view's B / C novel cells already present in an earlier executed
view of the same episode."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(E2)))))
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
CL = ("A", "A_out_obs", "A_out_unobs", "A_in_obs", "A_in_unobs", "B", "C", "T", "F", "N", "AC", "BC")


def onset(c):
    return next((t for t in range(101) if c[min(t + 20, 100)] - c[t] < 0.02), None)


def main():
    acc = {}
    rec_eps = {}
    for f in sorted(glob.glob(os.path.join(E2, "results", "audit_*_s*.jsonl"))):
        base = os.path.basename(f)[6:-6]
        scene, k = base.rsplit("_s", 1)
        k = int(k)
        cov = json.load(open(os.path.join(EP, "V3-D1-H3-NORISK", f"{scene}_s{k}.json")))["coverage_official"]
        on, low = onset(cov), cov[100] < 0.70
        rec_eps[base] = {"final": cov[100], "onset": on, "low": low}
        for line in open(f):
            r = json.loads(line)
            t = r["t"]
            if scene == "colosseum":
                g = "colosseum low-plateau (t >= onset)" if (low and on is not None and t >= on) else "colosseum other states"
            else:
                g = f"{scene} t<20" if t < 20 else f"{scene} t>=20"
            for gg in (g, "ALL"):
                for scope in ("all candidates", "executed"):
                    a = acc.setdefault(f"{gg} | {scope}", {"states": 0, "views": 0, "w_novel": 0.0, **{f"w_{c}": 0.0 for c in CL},
                                                            "known_agree": [], "leak": {"B_rays": 0, "band_on_ray": 0, "neg_on_ray": 0, "first_nonpos": 0},
                                                            "replay_max_rel": 0.0, "dcov_exec": []})
                    a["states"] += 1
                    for c in r["cands"]:
                        if scope == "executed" and c["j"] != r["executed"]:
                            continue
                        a["views"] += 1
                        a["w_novel"] += c["agg"]["w_novel"]
                        for cl in CL:
                            a[f"w_{cl}"] += c["agg"][f"w_{cl}"]
                        if c["known_surface_agrees_with_gt"] == c["known_surface_agrees_with_gt"]:
                            a["known_agree"].append(c["known_surface_agrees_with_gt"])
                        for kk in a["leak"]:
                            a["leak"][kk] += c["leak"][kk]
                        if c.get("replay_rel_err") is not None:
                            a["replay_max_rel"] = max(a["replay_max_rel"], c["replay_rel_err"])
                        if scope == "executed" and c.get("true_dcov_pp") is not None:
                            a["dcov_exec"].append(c["true_dcov_pp"])
    out = {"episodes": rec_eps, "groups": {}}
    for key, a in acc.items():
        W = max(a["w_novel"], 1e-9)
        out["groups"][key] = {"states": a["states"], "views": a["views"], "w_novel": a["w_novel"], **{f"share_{c}": a[f"w_{c}"] / W for c in CL},
                              "known_surface_agrees_with_gt_mean": float(np.mean(a["known_agree"])) if a["known_agree"] else None,
                              "leak": a["leak"], "replay_max_rel_err": a["replay_max_rel"],
                              "executed_true_dcov_mean_pp": float(np.mean(a["dcov_exec"])) if a["dcov_exec"] else None}
    rec = {}
    for f in sorted(glob.glob(os.path.join(E2, "results", "audit_*_cells.npz"))):
        z = np.load(f)
        base = os.path.basename(f)[6:-10]
        for cl in ("B", "C"):
            seen, tot, rep = set(), 0, 0
            for i in range(len([x for x in z.files if x.startswith(cl + "_")])):
                cells = set(z[f"{cl}_{i}"].tolist())
                tot += len(cells)
                rep += len(cells & seen)
                seen |= cells
            rec.setdefault(cl, []).append(rep / tot if tot else np.nan)
    out["recurrence_executed"] = {cl: float(np.nanmean(v)) for cl, v in rec.items()}
    json.dump(out, open(os.path.join(E2, "results", "audit_summary.json"), "w"), indent=1, default=float)
    print("| group | scope | states | views | A | A_out_obs | A_out_unobs | A_in_obs | A_in_unobs | B | C | T | F | N | known~GT | B leak: band / neg / first<=0 | exec dcov pp |")
    for key, g in out["groups"].items():
        gg, sc = key.split(" | ")
        L = g["leak"]
        lk = f"{L['band_on_ray'] / max(1, L['B_rays']):.2f} / {L['neg_on_ray'] / max(1, L['B_rays']):.2f} / {L['first_nonpos'] / max(1, L['B_rays']):.2f}"
        print(f"| {gg} | {sc} | {g['states']} | {g['views']} | " + " | ".join(f"{g['share_' + c]:.2f}" for c in ("A", "A_out_obs", "A_out_unobs", "A_in_obs", "A_in_unobs", "B", "C", "T", "F", "N"))
              + f" | {g['known_surface_agrees_with_gt_mean'] or 0:.2f} | {lk} | {g['executed_true_dcov_mean_pp'] if g['executed_true_dcov_mean_pp'] is None else round(g['executed_true_dcov_mean_pp'], 2)} |")
    print("recurrence", out["recurrence_executed"], "replay max rel", max(g["replay_max_rel_err"] for g in out["groups"].values()))


if __name__ == "__main__":
    main()
