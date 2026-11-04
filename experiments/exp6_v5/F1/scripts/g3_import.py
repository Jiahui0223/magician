"""EXP6 V5 / F1 -- canonical record of one finished G3 episode (seen2scene env, CPU) -> work_dirs/exp6_v5/F1/g3/episodes/
<MODEL>/<scene>_s<k>.json (+ done marker). Checks: 101 executed keys / 100 moves / 101 coverage values; metrics
cov_official == episode coverage (bitwise); planner checkpoint sha == sha256 of the registered checkpoint file. Never
overwrites a record with a different curve (STOP). Usage: python g3_import.py MODEL SCENE START"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
W = os.path.join(REPO, "work_dirs", "exp6_v5", "F1")
G3 = os.path.join(W, "g3")
CKPT = {"A2k": ("A", 2000), "A5k": ("A", 5000), "B2k": ("B", 2000), "B5k": ("B", 5000), "A1k": ("A", 1000), "B1k": ("B", 1000)}


def ckpt_of(m):
    arm, step = CKPT[m]
    return os.path.join(W, "runs", arm, f"ema_{step:06d}.pt")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    m, s, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tag = f"V2M_{s}_s{k}_r0"
    ep_p = os.path.join(G3, "closed_loop", "outdoor", m, "V2M", "episodes", f"ep_{tag}.json")
    mt_p = os.path.join(G3, "metrics", "outdoor", m, "V2M", f"{tag}.json")
    ep, mt = json.load(open(ep_p)), json.load(open(mt_p))
    assert len(ep["executed_keys"]) == 101 and len(ep["coverage"]) == 101 and ep["moves"] == 100, ep_p
    assert mt["cov_official"] == ep["coverage"], "official coverage replay differs"
    assert len(mt["cov_mono"]) == 101
    want = sha(ckpt_of(m))
    got = (ep["planner_hello"][0] if ep.get("planner_hello") else {}).get("ckpt_sha256")
    assert got == want, ("planner checkpoint sha", got, want)
    st = [x for x in ep["steps"] if x.get("planner")]
    dec = [x["planner"]["decision_s"] for x in st if x["planner"].get("decision_s") is not None]
    rec = {"model": m, "scene": s, "start": k, "split": "dev", "coverage_official": ep["coverage"], "coverage_mono": mt["cov_mono"],
           "executed_keys": ep["executed_keys"], "ckpt": os.path.relpath(ckpt_of(m), REPO), "ckpt_sha256": want, "n_hover": ep.get("n_hover", 0),
           "decision_s_median": sorted(dec)[len(dec) // 2] if dec else None, "decision_s_mean": sum(dec) / len(dec) if dec else None,
           "planner_peak_mem_gb": (ep.get("planner_bye") or {}).get("peak_mem_gb"), "seconds_this_process": ep.get("seconds_this_process"),
           "host": ep.get("host"), "episode_file": os.path.relpath(ep_p, REPO), "metrics_file": os.path.relpath(mt_p, REPO),
           "eval_protocol": "M-protocol frozen simulator via exp6_v5 sim_out_v5.py (V4_P1_MODE=sparse) + t25_out_cov official/mono",
           "imported": time.strftime("%F %T")}
    p = os.path.join(G3, "episodes", m, f"{s}_s{k}.json")
    if os.path.exists(p):
        if json.load(open(p))["coverage_official"] != rec["coverage_official"]:
            raise SystemExit(f"STOP: existing record differs {p}")
        print("exists", m, s, k)
        return
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rec, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    print("written", m, s, k, "final", round(rec["coverage_official"][-1], 4))


if __name__ == "__main__":
    main()
