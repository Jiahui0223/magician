"""EXP6 V2 (B2 / E2) -- minimal inference smoke and before / after equality check of the file organisation
(seen2scene env, 1 GPU; read-only: nothing under the historical work roots is written).

For one fixed-state episode (all its states, all legal candidates) the E2-pilot fixed-state replay (e2_pilot/scripts/
e2_fixed.py run_episode: same maps, R10 refresh, K = 4 seeded flow noise, DistP1 with the S10 band_dil) is run twice:
  OLD = planner cores loaded from the ORIGINAL checkpoint paths,  NEW = cores loaded through the new registry paths
        (work_dirs/checkpoints/exp6_v2/..., via the arm config experiments/exp6_v2/<arm>/configs/<arm>.json)
and compared:
  (1) future depth: sha256 of every predicted first-hit tensor (K samples x rays, per candidate)  OLD == NEW (bitwise)
  (2) DistP1 per candidate and K sample                                                            OLD == NEW (bitwise)
  (3) NEW vs the HISTORICAL rows written by the E2 pilot (work_dirs/.../e2/fixed/<set>_s5000/<stratum>/<tag>.jsonl):
      DistP1 scores (exact), camera ranking (mean over K, descending), prediction-error statistics vs t* (exact)
  (4) lineage at tensor level: E2 step-0 weights == V2-Mix EMA@17.5k; V2-Mix step-0 weights == V2 Original EMA@60k.
Models: arm B2 -> B2 (V2-Mix on D100) and B0 (same weights on D10, needed for the S10 views); arm E2 -> D100 (E2-D100)
and S10 (E2-S10 control). Output JSON (--out). Usage: python smoke_v2.py --arm B2|E2 --set dev3|valsel [--tag TAG] --out J"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_arm as VA  # noqa: E402

R = VA.R
E2P = os.path.join(R, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts")
sys.path.insert(0, E2P)
import e2_fixed as F  # noqa: E402

X = F.X
ARM_MODELS = {"B2": {"B0": ("V2M", 10), "B2": ("V2M", 100)}, "E2": {"S10": ("E2S", 10), "D100": ("E2D", 100)}}
REC = []


def cores_for(arm, dev, which):
    cfg = VA.load(arm)

    def mk(entry):
        p = (entry.get("path") or entry["checkpoint"]) if which == "new" else entry["original_path"]
        return X.PC.Core(VA.abspath(p), dev, expect_sha=entry["sha256"]), VA.abspath(p)
    if arm == "B2":
        c, p = mk(cfg["checkpoint"])
        return {"V2M": c}, {"V2M": p}
    c1, p1 = mk(cfg["checkpoint"])
    c2, p2 = mk(cfg["control_arm_E2S10"])
    return {"E2D": c1, "E2S": c2}, {"E2D": p1, "E2S": p2}


def install_recorder():
    orig = X.PC.score_state

    def rec(core, *a, **k):
        out, tm, views = orig(core, *a, **k)
        h = hashlib.sha256()
        for x in out["hits"]:
            h.update(x["t"].detach().float().cpu().numpy().tobytes())
        REC.append({"ckpt_sha16": core.ckpt_sha[:16], "future_depth_sha256": h.hexdigest(), "distp1_k": out["distp1_k"]})
        return out, tm, views
    X.PC.score_state = rec


def lineage():
    import torch
    L = lambda p: torch.load(os.path.join(R, p), map_location="cpu", weights_only=False)["ema"]  # noqa: E731
    C = "work_dirs/checkpoints/exp6_v2"

    def same(a, b):
        return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    return {"E2D_step0_weights_eq_V2Mix_ema17500": same(L(f"{C}/E2/intermediate/e2d100_ema000000.pt"), L(f"{C}/B2/v2mix_ema17500.pt")),
            "V2Mix_step0_weights_eq_V2Original_ema60000": same(L(f"{C}/B2/intermediate/v2mix_ema000000.pt"), L(f"{C}/B2/lineage/v2orig_ema60000.pt")),
            "E2D_ema5000_differs_from_init": not same(L(f"{C}/E2/e2d100_ema5000.pt"), L(f"{C}/B2/v2mix_ema17500.pt"))}


def rank(s):
    import numpy as np
    m = np.asarray([float(np.mean(x)) for x in s])
    return [int(i) for i in np.argsort(-m, kind="stable")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["B2", "E2"], required=True)
    ap.add_argument("--set", choices=["dev3", "valsel"], required=True)
    ap.add_argument("--stratum", default="B0")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task2", "scripts"))
    torch = X.torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    F.MODELS = ARM_MODELS[a.arm]
    install_recorder()
    eps = dict(F.states(a.set, a.stratum))
    tag = a.tag or sorted(eps)[0]
    sts = eps[tag]
    res = {"arm": a.arm, "set": a.set, "stratum": a.stratum, "tag": tag, "states": len(sts), "models": F.MODELS, "gpu": torch.cuda.get_device_name(0), "time": time.strftime("%F %T")}
    runs = {}
    for which in ("old", "new"):
        REC.clear()
        cores, paths = cores_for(a.arm, dev, which)
        t0 = time.time()
        with torch.no_grad():
            rows = F.run_episode(a.set, a.stratum, tag, sts, cores, dev, None)
        runs[which] = {"rows": json.loads(json.dumps(rows, default=float)), "rec": list(REC), "paths": {k: os.path.relpath(v, R) for k, v in paths.items()}, "seconds": time.time() - t0}
        del cores
        torch.cuda.empty_cache()
    o, n = runs["old"], runs["new"]
    res["paths"] = {"old": o["paths"], "new": n["paths"]}
    res["score_state_calls"] = len(n["rec"])
    res["future_depth_old_eq_new"] = [x["future_depth_sha256"] for x in o["rec"]] == [x["future_depth_sha256"] for x in n["rec"]]
    res["distp1_old_eq_new"] = [x["distp1_k"] for x in o["rec"]] == [x["distp1_k"] for x in n["rec"]]
    res["future_depth_sha256_first_call"] = n["rec"][0]["future_depth_sha256"] if n["rec"] else None
    hist_p = os.path.join(R, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2", "fixed", f"{a.set}_s5000", a.stratum, f"{tag}.jsonl")
    hist = {r["sid"]: r for r in (json.loads(x) for x in open(hist_p))}
    per = []
    for r in n["rows"]:
        h = hist[r["sid"]]
        d = {"sid": r["sid"], "t": r["t"], "n_cand": len(r["scores"][list(F.MODELS)[0]])}
        for m in F.MODELS:
            d[f"{m}_scores_eq_hist"] = r["scores"][m] == h["scores"][m]
            d[f"{m}_rank_eq_hist"] = rank(r["scores"][m]) == rank(h["scores"][m])
            d[f"{m}_pred_eq_hist"] = r["pred"][m] == h["pred"][m]
            d[f"{m}_top1"] = rank(r["scores"][m])[0]
        per.append(d)
    res["historical_rows"] = os.path.relpath(hist_p, R)
    res["per_state"] = per
    res["all_equal_to_history"] = all(v for d in per for k, v in d.items() if k.endswith("_eq_hist"))
    res["lineage"] = lineage()
    res["seconds"] = {"old": o["seconds"], "new": n["seconds"]}
    res["ok"] = res["future_depth_old_eq_new"] and res["distp1_old_eq_new"] and res["all_equal_to_history"] and all(res["lineage"].values())
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("arm", "set", "tag", "states", "score_state_calls", "future_depth_old_eq_new", "distp1_old_eq_new", "all_equal_to_history", "lineage", "ok")}, indent=1))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
