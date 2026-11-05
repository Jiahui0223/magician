"""E3 Task 1 -- aggregate results/fixed/dv2_fixed_*.jsonl -> results/fixed_summary.json and the fixed-state gates F1-F5 of
configs/dv2_v2.yaml. Gate episodes: colosseum s0-s4 + pisa s1; alhambra s0 (Test-3) is reported separately and never enters a gate."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.dirname(HERE)
EE = os.path.dirname(E3)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(EE))))
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
MODES = ("off", "ksv1", "dv2", "dv2_first")


def onset(c):
    return next((t for t in range(101) if c[min(t + 20, 100)] - c[t] < 0.02), None)


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(E3, "results", "fixed", "dv2_fixed_*_s*.jsonl"))):
        scene, k = os.path.basename(f)[10:-6].rsplit("_s", 1)
        cov = json.load(open(os.path.join(EP, "V3-D1-H3-NORISK", f"{scene}_s{k}.json")))["coverage_official"]
        on, low = onset(cov), cov[100] < 0.70
        e2 = os.path.join(EE, "E2_observed_geometry", "results", f"ksv_fixed_{scene}_s{k}.jsonl")
        e2r = {json.loads(l)["t"]: json.loads(l) for l in open(e2)} if os.path.exists(e2) else {}
        for line in open(f):
            r = json.loads(line)
            r.update({"scene": scene, "start": int(k), "plateau": bool(scene == "colosseum" and low and on is not None and r["t"] >= on),
                      "ksv1_equals_e2": (r["ksv1"]["scores"] == e2r[r["t"]]["on"]["scores"]) if r["t"] in e2r else None})
            rows.append(r)
    gate_rows = [r for r in rows if r["scene"] != "alhambra"]
    groups = {"gate set: all": lambda r: r["scene"] != "alhambra", "gate set: t<20": lambda r: r["scene"] != "alhambra" and r["t"] < 20,
              "gate set: t>=20": lambda r: r["scene"] != "alhambra" and r["t"] >= 20, "colosseum plateau": lambda r: r["plateau"],
              "colosseum": lambda r: r["scene"] == "colosseum", "pisa s1": lambda r: r["scene"] == "pisa", "alhambra s0 (report only)": lambda r: r["scene"] == "alhambra"}
    m = lambda xs: float(np.mean(xs)) if len(xs) else None  # noqa: E731
    out = {"n_states": len(rows), "groups": {}}
    for g, f in groups.items():
        sel = [r for r in rows if f(r)]
        ev = [r for r in sel if r["off"].get("sel_true_gain_pp") is not None]
        G = {"states": len(sel), "states_with_oracle": len(ev), "best_gain_pp": m([r["best_true_gain_pp"] for r in ev])}
        for md in MODES:
            d = np.asarray([r[md]["sel_true_gain_pp"] - r["off"]["sel_true_gain_pp"] for r in ev]) if ev else np.zeros(0)
            x = {"top1": m([r[md]["top1"] for r in ev]), "sel_gain_pp": m([r[md]["sel_true_gain_pp"] for r in ev]),
                 "rho": m([r[md]["rho"] for r in ev if r[md]["rho"] is not None]), "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum()),
                 "pick_changed": m([r[md]["pick"] != r["off"]["pick"] for r in sel]), "seconds": m([r[md]["seconds"] for r in sel]),
                 "cut_share_L1": float(sum(r[md]["cut"][0] for r in sel) / max(1, sum(r[md]["rays"][0] for r in sel))) if md != "off" else 0.0,
                 "cut_share_L2": float(sum(r[md]["cut"][1] for r in sel) / max(1, sum(r[md]["rays"][1] for r in sel))) if md not in ("off", "dv2_first") else 0.0,
                 "cut_L2": int(sum(r[md]["cut"][1] for r in sel))}
            if md in ("ksv1", "dv2"):
                q = [r[md]["cut_quality"] for r in sel]
                S = lambda key: sum(z[key] for z in q)  # noqa: E731
                x.update({"wrong_cut_share": S("wrong_w") / max(1e-9, S("cut_w")), "cut_w_per_state": S("cut_w") / max(1, len(q)),
                          "beyond_true_off": S("beyond_off") / max(1e-9, S("nov_off")), "beyond_true_on": S("beyond_on") / max(1e-9, S("nov_on")),
                          "removed_share_of_novel": S("cut_w") / max(1e-9, S("nov_off"))})
            G[md] = x
        out["groups"][g] = G
    A, Ee = out["groups"]["gate set: all"], out["groups"]["gate set: t<20"]
    gates = {"F1_off_equals_log": all(r["regression_off_equals_log"] for r in gate_rows),
             "F2_ksv1_equals_E2": all(r["ksv1_equals_e2"] for r in gate_rows if r["ksv1_equals_e2"] is not None) and any(r["ksv1_equals_e2"] for r in gate_rows),
             "F3_gain": bool(A["dv2"]["sel_gain_pp"] > A["off"]["sel_gain_pp"] and Ee["dv2"]["sel_gain_pp"] >= Ee["off"]["sel_gain_pp"]),
             "F4_wrong_cut_early": bool(Ee["dv2"]["wrong_cut_share"] < Ee["ksv1"]["wrong_cut_share"]),
             "F5_layer2_cuts": bool(A["dv2"]["cut_L2"] > 0)}
    out["gates"] = gates
    out["gate_pass"] = all(gates.values())
    out["regression_counts"] = {"off_equals_log": sum(r["regression_off_equals_log"] for r in rows), "ksv1_equals_e2": sum(bool(r["ksv1_equals_e2"]) for r in rows), "states": len(rows)}
    json.dump(out, open(os.path.join(E3, "results", "fixed_summary.json"), "w"), indent=1, default=float)
    print(json.dumps({"gates": gates, "gate_pass": out["gate_pass"], **out["regression_counts"]}))
    f2 = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    print("| group | states | mode | Top-1 | sel gain pp (best) | better/worse | rho | pick changed | cut L1 / L2 | wrong-cut | beyond true off→on | s/state |")
    for g, G in out["groups"].items():
        for md in MODES:
            x = G[md]
            print(f"| {g} | {G['states']} | {md} | {f2(x['top1'])} | {f2(x['sel_gain_pp'])} ({f2(G['best_gain_pp'])}) | {x['better']}/{x['worse']} | {f2(x['rho'])} | {f2(x['pick_changed'])} | "
                  f"{x['cut_share_L1']:.2f} / {x['cut_share_L2']:.2f} | {f2(x.get('wrong_cut_share'))} | {f2(x.get('beyond_true_off'))}→{f2(x.get('beyond_true_on'))} | {f2(x['seconds'])} |")


if __name__ == "__main__":
    main()
