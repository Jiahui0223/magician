"""E2 Task 2 -- aggregate results/ksv_fixed_*.jsonl -> results/ksv_fixed_summary.json and the gate decision of configs/ksv_v1.yaml
(ON must raise the mean selected true one-step gain over all fixed states and must not lower it on states with t < 20)."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(E2)))))
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")


def onset(c):
    return next((t for t in range(101) if c[min(t + 20, 100)] - c[t] < 0.02), None)


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(E2, "results", "ksv_fixed_*_s*.jsonl"))):
        scene, k = os.path.basename(f)[10:-6].rsplit("_s", 1)
        cov = json.load(open(os.path.join(EP, "V3-D1-H3-NORISK", f"{scene}_s{k}.json")))["coverage_official"]
        on, low = onset(cov), cov[100] < 0.70
        for line in open(f):
            r = json.loads(line)
            r.update({"scene": scene, "start": int(k), "plateau": bool(scene == "colosseum" and low and on is not None and r["t"] >= on)})
            rows.append(r)
    groups = {"all": lambda r: True, "t<20": lambda r: r["t"] < 20, "t>=20": lambda r: r["t"] >= 20, "colosseum plateau": lambda r: r["plateau"],
              "colosseum": lambda r: r["scene"] == "colosseum", "pisa": lambda r: r["scene"] == "pisa", "alhambra": lambda r: r["scene"] == "alhambra"}
    out = {"n_states": len(rows), "regression_off_equals_log": sum(r["regression_off_equals_log"] for r in rows), "groups": {}}
    for g, f in groups.items():
        sel = [r for r in rows if f(r)]
        ev = [r for r in sel if r["off"].get("sel_true_gain_pp") is not None]
        cq = [r["on"]["cut_quality"] for r in sel]
        m = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
        d = np.asarray([r["on"]["sel_true_gain_pp"] - r["off"]["sel_true_gain_pp"] for r in ev]) if ev else np.zeros(0)
        out["groups"][g] = {
            "states": len(sel), "states_with_oracle": len(ev),
            "top1_off": m([r["off"]["top1"] for r in ev]), "top1_on": m([r["on"]["top1"] for r in ev]),
            "sel_gain_off_pp": m([r["off"]["sel_true_gain_pp"] for r in ev]), "sel_gain_on_pp": m([r["on"]["sel_true_gain_pp"] for r in ev]),
            "best_gain_pp": m([r["best_true_gain_pp"] for r in ev]),
            "gain_diff_pp": float(d.mean()) if len(d) else None, "states_better": int((d > 1e-9).sum()), "states_worse": int((d < -1e-9).sum()),
            "rho_off": m([r["off"]["rho"] for r in ev if r["off"]["rho"] is not None]), "rho_on": m([r["on"]["rho"] for r in ev if r["on"]["rho"] is not None]),
            "pick_changed": m([r["off"]["pick"] != r["on"]["pick"] for r in sel]),
            "novel_beyond_true_off": m([q["beyond_share_off"] for q in cq]), "novel_beyond_true_on": m([q["beyond_share_on"] for q in cq]),
            "wrong_cut_share_of_cut_value": float(sum(q["wrong_cut_w"] for q in cq) / max(1e-9, sum(q["cut_w"] for q in cq))) if cq else None,
            "seconds_off": m([r["off"]["seconds"] for r in sel]), "seconds_on": m([r["on"]["seconds"] for r in sel])}
    a, e = out["groups"]["all"], out["groups"]["t<20"]
    out["gate_pass"] = bool(a["gain_diff_pp"] is not None and a["gain_diff_pp"] > 0 and e["gain_diff_pp"] is not None and e["gain_diff_pp"] >= 0)
    json.dump(out, open(os.path.join(E2, "results", "ksv_fixed_summary.json"), "w"), indent=1)
    print("states", out["n_states"], "regression OFF==log", out["regression_off_equals_log"], "gate_pass", out["gate_pass"])
    print("| group | states (oracle) | Top-1 OFF→ON | sel. true gain pp OFF→ON (best) | diff pp | better/worse | ρ OFF→ON | pick changed | novel beyond true OFF→ON | wrong-cut share | s/state OFF→ON |")
    for g, x in out["groups"].items():
        f = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"| {g} | {x['states']} ({x['states_with_oracle']}) | {f(x['top1_off'])}→{f(x['top1_on'])} | {f(x['sel_gain_off_pp'])}→{f(x['sel_gain_on_pp'])} ({f(x['best_gain_pp'])}) | "
              f"{f(x['gain_diff_pp'])} | {x['states_better']}/{x['states_worse']} | {f(x['rho_off'])}→{f(x['rho_on'])} | {f(x['pick_changed'])} | "
              f"{f(x['novel_beyond_true_off'])}→{f(x['novel_beyond_true_on'])} | {f(x['wrong_cut_share_of_cut_value'])} | {f(x['seconds_off'])}→{f(x['seconds_on'])} |")


if __name__ == "__main__":
    main()
