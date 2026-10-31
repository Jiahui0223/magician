"""EXP6 V3 Task 2 -- planner-side patch (seen2scene env, planner children only): replace the DistP1 candidate score of the
frozen V2 / V3 planners by a Task 2 scorer variant and / or add the frontier-progress term, everything else unchanged.

install() patches, in this process only:
  m_common.load_service_record -> also records the 10 % sample points of every loaded record (the historical observed
                                  surface, deduplicated by record step) into HIST (runtime y-up; global, region-free)
  v2_plan_core.score_state     -> calls the frozen function with keep_hits=True, then for every candidate and every
                                  sample re-scores the decoded first-hit distances with t2_scorers.score_view on the
                                  planner's own geometry (B2 mode: dense t_known for S3, sparse band_dil for S0) and
                                  the historical sample points; out["distp1"] <- K-mean of the variant
                                  (out["distp1_orig"] keeps the frozen value; variant "S0" reproduces it bitwise)
  frontier (MP3D, env T2_FRONTIER = bonus | trigger): out["distp1"] <- Gain + lambda FP (bonus) or FP-first when
                                  max Gain < theta (trigger); FP from t2_frontier.FrontierMap (records only, no GT)
Env: T2_SCORER (variant name, default S0), T2_DATASET (indoor | outdoor), T2_FRONTIER (none | bonus | trigger),
     T2_LAMBDA, T2_THETA, T2_LOG (optional jsonl path for per-step diagnostics).
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_scorers as TS  # noqa: E402

HIST = TS.Hist()
HIST_T = set()
CFG = {"variant": os.environ.get("T2_SCORER", "S0"), "dataset": os.environ.get("T2_DATASET", "indoor"),
       "frontier": os.environ.get("T2_FRONTIER", "none"), "lam": float(os.environ.get("T2_LAMBDA", "0")), "theta": float(os.environ.get("T2_THETA", "0")),
       "log": os.environ.get("T2_LOG", "")}
EPS_MODEL_INDOOR = 0.08
EPS_RT_OUTDOOR = 1.0
STATE = {"fmap": None, "last_rec": None, "step": 0}


def _record_hook(orig):
    def load_service_record(path):
        rec = orig(path)
        t = int(rec["t"])
        if t not in HIST_T:
            HIST_T.add(t)
            HIST.add(rec["sample_points"])
            if STATE["fmap"] is not None:
                STATE["fmap"].add_record(rec)
        STATE["last_rec"] = rec
        return rec
    return load_service_record


def _score_hook(orig):
    @torch.no_grad()
    def score_state(core, geo, h, legal_R, legal_T, eps, z0, deltas=(), want_flow=False, want_feat=False, keep_hits=False, jitter=(0.0, 0.0)):
        out, tm, views = orig(core, geo, h, legal_R, legal_T, eps, z0, deltas=deltas, want_flow=want_flow, want_feat=want_feat, keep_hits=True, jitter=jitter)
        t0 = time.time()
        var = CFG["variant"]
        k_M = float(geo.fr.k_M)
        eps_rt = EPS_MODEL_INDOOR * k_M if CFG["dataset"] == "indoor" else EPS_RT_OUTDOOR
        HIST.build()
        new_scores, orig_scores, per_k = [], list(out["distp1"]), []
        for j, v in enumerate(views):
            v["struct"] = geo.struct
            hh = out["hits"][j]
            vals = []
            for k in range(hh["t"].shape[0]):
                sc, _ = TS.score_view(hh["t"][k], v, v, geo, HIST, eps_rt, z0, geo.fr)
                vals.append(sc[var])
            per_k.append(vals)
            new_scores.append(float(np.mean(vals)))
        out["distp1_orig"] = orig_scores
        out["distp1_k_orig"] = out["distp1_k"]
        out["distp1"] = new_scores
        out["distp1_k"] = per_k
        out["t2_variant"] = var
        tm["t2_score_s"] = time.time() - t0
        tm["t2_hist_points"] = HIST.n
        if var == "S0":
            d = max(abs(a - b) for a, b in zip(new_scores, orig_scores)) if new_scores else 0.0
            assert d <= 1e-9 * max(1.0, max(abs(x) for x in orig_scores) if orig_scores else 1.0), ("S0 patch does not reproduce DistP1", d)
        if CFG["frontier"] != "none" and STATE["fmap"] is not None and STATE["last_rec"] is not None:
            t1 = time.time()
            fp, finfo = STATE["fmap"].progress(STATE["last_rec"], legal_R, legal_T)
            gain = np.asarray(out["distp1"], float)
            if CFG["frontier"] == "bonus":
                sc = gain + CFG["lam"] * np.asarray(fp, float)
            else:
                trig = bool(len(gain) and gain.max() < CFG["theta"] and finfo.get("n_reachable_frontier", 0) > 0)
                sc = (np.asarray(fp, float) * 1e6 + gain) if trig else gain
                finfo["triggered"] = trig
            out["distp1_gain"] = [float(x) for x in gain]
            out["frontier_progress"] = [float(x) for x in fp]
            out["frontier_info"] = finfo
            out["distp1"] = [float(x) for x in sc]
            tm["t2_frontier_s"] = time.time() - t1
        if not keep_hits:
            del out["hits"]
        if CFG["log"]:
            with open(CFG["log"], "a") as f:
                f.write(json.dumps({"step": STATE["step"], "variant": var, "scores": out["distp1"], "orig": orig_scores, "t2_score_s": tm["t2_score_s"],
                                    "frontier": out.get("frontier_info"), "fp": out.get("frontier_progress")}, default=float) + "\n")
        STATE["step"] += 1
        return out, tm, views
    return score_state


def install():
    import m_common as MC
    import v2_plan_core as PC
    MC.load_service_record = _record_hook(MC.load_service_record)
    PC.score_state = _score_hook(PC.score_state)
    if CFG["frontier"] != "none":
        import t2_frontier as TF
        STATE["fmap"] = TF.FrontierMap.from_env()
    sys.stderr.write(f"T2 patch: {CFG}\n")
    return CFG
