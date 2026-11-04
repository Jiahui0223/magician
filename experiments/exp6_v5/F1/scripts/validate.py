"""EXP6 V5 / F1 -- completeness / validity of the G3 records (CPU). Expected: A2k / A5k / B2k / B5k x Dev-3 x starts 0..4 and the
B0 parent records of CVPR Table 1. Checks: 101 coverage values in [0, 1], official == metrics replay (checked at import),
monotone curve non-decreasing, 101 executed keys, checkpoint sha equal to the file. Missing episodes are listed, never
filled. Output results/validation.json + RUN_STATUS appendix lines on stdout. Usage: python validate.py"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(F1)))
W = os.path.join(R, "work_dirs", "exp6_v5", "F1")
T1E = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
SCENES = ("barts", "pantheon", "colosseum")
CK = {"A2k": "A/ema_002000.pt", "A5k": "A/ema_005000.pt", "B2k": "B/ema_002000.pt", "B5k": "B/ema_005000.pt"}


def main():
    sha = {}
    for m, p in CK.items():
        fp = os.path.join(W, "runs", p)
        sha[m] = hashlib.sha256(open(fp, "rb").read()).hexdigest() if os.path.exists(fp) else None
    out = {"valid": {}, "missing": {}, "bad": []}
    for m in ("B0",) + tuple(CK):
        for s in SCENES:
            for k in range(5):
                p = os.path.join(T1E, "B0", f"{s}_s{k}.json") if m == "B0" else os.path.join(W, "g3", "episodes", m, f"{s}_s{k}.json")
                if not os.path.exists(p):
                    out["missing"].setdefault(m, []).append(f"{s}_s{k}")
                    continue
                r = json.load(open(p))
                c, cm = r["coverage_official"], r["coverage_mono"]
                ok = len(c) == 101 and len(cm) == 101 and all(0 <= x <= 1 for x in c + cm) and all(b >= a - 1e-12 for a, b in zip(cm, cm[1:])) \
                    and len(r["executed_keys"]) == 101 and (m == "B0" or r["ckpt_sha256"] == sha[m])
                if ok:
                    out["valid"][m] = out["valid"].get(m, 0) + 1
                else:
                    out["bad"].append(p)
    out["checkpoint_sha256"] = sha
    json.dump(out, open(os.path.join(F1, "results", "validation.json"), "w"), indent=1)
    print("valid", out["valid"], "missing", {k: len(v) for k, v in out["missing"].items()}, "bad", len(out["bad"]))


if __name__ == "__main__":
    main()
