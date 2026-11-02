"""EXP6 V5 -- planner-side model loading: v2_plan_core.Core builds M.V2Model(ck["model_cfg"]); install() makes it build
seen2scene.futureview.v5_model.V5Model when the checkpoint's model_cfg has arch == "v5" (strict state-dict load),
everything else (Core fields, predictor, scoring, candidate generator, S10 DistP1, K, n_win) unchanged. A guard makes
any torch grid_sample call inside a V5 predictor forward raise (the old 32-point ray sampling must not run)."""
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from seen2scene.futureview import v5_model as V5  # noqa: E402


class _NoGridSample:
    def __enter__(self):
        self.orig = F.grid_sample

        def guard(*a, **k):
            raise RuntimeError("V5 inference called torch grid_sample (old 32-point ray sampling)")
        F.grid_sample = guard

    def __exit__(self, *exc):
        F.grid_sample = self.orig
        return False


def install():
    import v2_plan_core as PC
    import seen2scene.futureview.predictor as PR
    orig = PC.Core.__init__

    def __init__(self, ckpt_path, device, expect_sha=None, steps=10):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["model_cfg"].get("arch") != "v5":
            return orig(self, ckpt_path, device, expect_sha, steps)
        sha = PC.VF.sha256_file(ckpt_path)
        if expect_sha is not None and sha != expect_sha:
            raise PC.VF.InvalidRun(f"INVALID: checkpoint sha {sha} != frozen {expect_sha}")
        self.model = V5.V5Model(ck["model_cfg"]).to(device)
        self.model.load_state_dict(ck["ema"], strict=True)
        self.model.eval()
        self.pred = PR.V2Predictor(self.model, device, steps)
        self.dev, self.ckpt_sha, self.ckpt = device, sha, ckpt_path
        for nm in ("encode", "cond", "sample"):
            fn = getattr(self.pred, nm, None)
            if fn is not None:
                def wrapped(*a, _fn=fn, **k):
                    with _NoGridSample():
                        return _fn(*a, **k)
                setattr(self.pred, nm, wrapped)
        sys.stderr.write(f"V5 Core: {ckpt_path} sha {sha[:12]} arch v5, grid_sample guard on\n")
    PC.Core.__init__ = __init__
    return PC
