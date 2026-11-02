"""EXP6 V5.2-M (Scratch70k) -- planner-side integration (copy of exp6_v5_2/scripts/v52_core.py with the V5.2-M model / scorer) (seen2scene env, inside the frozen V2 planner children).

install(lam, score_cfg) patches, in this process only:
  * v2_plan_core.Core.__init__: a checkpoint whose model_cfg arch is "v52" builds seen2scene.futureview.v52_model.V52Model
    (strict EMA load, sha checked by the caller's expect_sha); the predictor is only used for encode() (D100 coarse grid)
  * v2_plan_core.score_state: for a V5.2 core the per-candidate score is the V5.2 camera utility
    S_vis_hat + lam (eta P_surf_hat + (1 - eta) P_space_hat) (seen2scene/futureview/v52_score.py), returned under the
    key "distp1" so the planner's own choice rule (m_score.choose / prefer-unvisited) is unchanged; all raw terms,
    normalised terms, both lambda scores and diagnostics are returned too and appended to $V52_STEPLOG
  * guards: flow sampling (model.sample), the predictor's cond / sample, the depth decoder (geometry.decode) and the
    DistP1 functions (scoring.new_set_from_depth / new_mask_from_depth) raise if called while a V5.2 core is active;
    every call is counted first (GUARD_COUNTS, logged with every step: all zero in a valid run)
Candidate generation, legality, n_win, refresh (R10), the real P1 fusion (D100 + S10, p1_full b2) and the simulator
are untouched."""
import json
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.futureview import predictor as PR  # noqa: E402
from seen2scene.futureview import scoring as SCO  # noqa: E402
from seen2scene.futureview import v52m_model as V52M  # noqa: E402
from seen2scene.futureview import v52m_score as SC52  # noqa: E402

GUARD_COUNTS = {"flow_sample": 0, "pred_cond": 0, "pred_sample": 0, "decode_depth": 0, "distp1_new_set": 0, "distp1_new_mask": 0}
STATE = {"active": False, "lam": None, "cfg": None}


def _guard(name, fn):
    def g(*a, **k):
        if STATE["active"]:
            GUARD_COUNTS[name] += 1
            raise RuntimeError(f"V5.2 guard: {name} must not be called (no flow / depth image / DistP1 in V5.2)")
        return fn(*a, **k)
    return g


def install(lam, score_cfg):
    import v2_plan_core as PC
    cfg = dict(SC52.DEFAULT, **score_cfg)
    STATE.update({"lam": float(lam), "cfg": cfg})
    M.sample = _guard("flow_sample", M.sample)
    G.decode = _guard("decode_depth", G.decode)
    SCO.new_set_from_depth = _guard("distp1_new_set", SCO.new_set_from_depth)
    SCO.new_mask_from_depth = _guard("distp1_new_mask", SCO.new_mask_from_depth)
    PR.V2Predictor.cond = _guard("pred_cond", PR.V2Predictor.cond)
    PR.V2Predictor.sample = _guard("pred_sample", PR.V2Predictor.sample)
    orig_init, orig_score = PC.Core.__init__, PC.score_state

    def __init__(self, ckpt_path, device, expect_sha=None, steps=10):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["model_cfg"].get("arch") != "v52m":
            return orig_init(self, ckpt_path, device, expect_sha, steps)
        sha = PC.VF.sha256_file(ckpt_path)
        if expect_sha is not None and sha != expect_sha:
            raise PC.VF.InvalidRun(f"INVALID: checkpoint sha {sha} != frozen {expect_sha}")
        self.model = V52M.V52MModel(ck["model_cfg"]).to(device)
        self.model.load_state_dict(ck["ema"], strict=True)
        self.model.eval()
        self.pred = PR.V2Predictor(self.model, device, steps)
        self.dev, self.ckpt_sha, self.ckpt = device, sha, ckpt_path
        self.is_v52 = True
        STATE["active"] = True
        sys.stderr.write(f"V5.2-M Core: {ckpt_path} sha {sha[:12]} lambda {STATE['lam']} eta {cfg['eta']} floors {cfg['floors']}; flow / DistP1 guards on\n")

    @torch.no_grad()
    def score_state(core, geo, h, legal_R, legal_T, eps, z0, *a, **k):
        if not getattr(core, "is_v52", False):
            return orig_score(core, geo, h, legal_R, legal_T, eps, z0, *a, **k)
        torch.cuda.synchronize(); t0 = time.time()
        views = [geo.view(R, T) for R, T in zip(legal_R, legal_T)]
        torch.cuda.synchronize(); t1 = time.time()
        raw = SC52.score_state(core.model, h, geo, views, float(geo.fr.k_M), z0, cfg)
        torch.cuda.synchronize(); t2 = time.time()
        out = {"v52_s_vis": raw["s_vis"], "v52_p_surf": raw["p_surf"], "v52_p_space": raw["p_space"], "v52_state": raw["state"],
               "v52_diag": raw["diag"], "guard_counts": dict(GUARD_COUNTS)}
        for lm in sorted({0.0, 0.1, STATE["lam"]}):
            sc, parts = SC52.combine(raw, lm, cfg["eta"], cfg["floors"])
            out[f"v52_score_l{lm:g}"] = sc
            if lm == STATE["lam"]:
                out["distp1"] = sc                                              # the planner's choice key (unchanged rule)
                out["v52_parts"] = parts
        out["distp1_k"] = [[s] for s in out["distp1"]]
        tm = {"projection_s": t1 - t0, "v52_score_s": t2 - t1}
        p = os.environ.get("V52_STEPLOG")
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "a") as f:
                f.write(json.dumps({"lam": STATE["lam"], "n": len(views), **{kk: v for kk, v in out.items() if kk != "distp1_k"}, "timing": tm}, default=float) + "\n")
        return out, tm, views

    PC.Core.__init__ = __init__
    PC.score_state = score_state
    return PC
