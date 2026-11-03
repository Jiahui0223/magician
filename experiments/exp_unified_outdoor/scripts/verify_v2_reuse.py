"""CPU-only: evidence for reusing the historical V2 Original run (task1 training/main) as the unified V2-S10 arm.
  1. sha256 of the EMA checkpoints that matter (10k / 20k / 40k / 45k / 50k / 60k) and of ckpt_last.pt (60k raw state)
  2. sha256 of the 180 S10 view shards now on disk vs the data_files_sha256 recorded by the run (run_none.json)
  3. sha256 of the training code / configs now on disk vs the run's code fingerprint (code_sha256)
Writes work_dirs/exp_unified_outdoor/verify/v2_reuse.json. Never writes into the historical run directory.
"""
import hashlib
import json
import os
import sys
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
RUN = os.path.join(R, "work_dirs", "exp5_future_view", "task1", "training", "main")
S10 = os.path.join(R, "work_dirs", "exp5_future_view", "task1", "data", "views")
OUT = os.path.join(R, "work_dirs", "exp_unified_outdoor", "verify", "v2_reuse.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    run = json.load(open(os.path.join(RUN, "run_none.json")))
    out = {"run_dir": RUN, "ckpt": {}, "shards": {}, "code": {}}
    for f in ["ema_010000.pt", "ema_020000.pt", "ema_040000.pt", "ema_045000.pt", "ema_050000.pt", "ema_060000.pt", "ckpt_last.pt"]:
        p = os.path.join(RUN, f)
        out["ckpt"][f] = {"exists": os.path.exists(p), "bytes": os.path.getsize(p) if os.path.exists(p) else None,
                          "sha256": sha(p) if os.path.exists(p) else None}
    out["ckpt"]["ema_048000.pt"] = {"exists": os.path.exists(os.path.join(RUN, "ema_048000.pt"))}
    out["ckpt_final_matches_run_record"] = out["ckpt"]["ema_060000.pt"]["sha256"] == run["final_checkpoint_sha256"]
    bad, n = [], 0
    for k, v in sorted(run["data_files_sha256"].items()):
        s, tj = k.split("/")
        p = os.path.join(S10, s, f"{tj}.npz")
        n += 1
        cur = sha(p) if os.path.exists(p) else None
        if cur != v:
            bad.append({"file": k, "recorded": v, "now": cur})
    out["shards"] = {"n_recorded": len(run["data_files_sha256"]), "n_checked": n, "n_mismatch": len(bad), "mismatch": bad[:20]}
    for k, v in sorted(run["code_sha256"].items()):
        p = os.path.join(R, k)
        cur = sha(p) if os.path.exists(p) else None
        out["code"][k] = {"run": v, "now": cur, "equal": cur == v}
    out["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({"ckpt": {k: (v.get("sha256") or v) for k, v in out["ckpt"].items()}, "final_matches": out["ckpt_final_matches_run_record"],
                      "shards": {k: v for k, v in out["shards"].items() if k != "mismatch"},
                      "code_changed": [k for k, v in out["code"].items() if not v["equal"]], "seconds": round(out["seconds"], 1)}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
