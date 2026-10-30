"""EXP6 V3 -- planner-side patch: run the frozen Task 2 / Task 3 closed-loop planners with a V3 model.

The V2 planners (task1 v2_planner.py, task0 t06_planner.py) build `geo = PC.Geometry.from_fusion(...)` and refresh the
encoder with `core.pred.encode(geo.coarse())`. With install():
  PC.Core                 -> V3Core (V3 EMA checkpoint, sha256 checked; .pred = V3Predictor)
  Geometry.from_fusion    -> also keeps the fusion object on the geometry (composes with p1_full's B2 patch, which
                             swaps band_dil to the sparse twin: D100 input, sparse DistP1 scoring, as in training)
  Geometry.coarse         -> partial_tsdf.pack_grid(aggregate_fusion(fusion)): the 4x grid as one uint8 tensor (the
                             planner hashes it into its log); V3Predictor.encode unpacks it
Everything else (candidates, noise seeds, DistP1, choice rule, records, coverage) is the frozen V2 path.
Env V3_CKPT: the checkpoint replaces the group's --ckpt / --sha256 of the planner command line (ckpt_override()).
"""
import hashlib
import os
import sys

import torch

from seen2scene.v3 import partial_tsdf as PT
from seen2scene.v3 import predictor as V3P


def ckpt_override():
    ck = os.environ.get("V3_CKPT")
    if not ck:
        raise RuntimeError("V3_CKPT not set")
    sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
    for flag, val in (("--ckpt", ck), ("--sha256", sha)):
        i = sys.argv.index(flag)
        sys.argv[i + 1] = val
    return ck, sha


class V3Core:
    def __init__(self, ckpt_path, device, expect_sha=None, steps=10):
        self.model, self.pred, self.ckpt_sha = V3P.load_core(ckpt_path, device, steps, expect_sha)
        self.dev, self.ckpt = device, ckpt_path


def install():
    import v2_plan_core as PC
    PC.Core = V3Core
    orig = PC.Geometry.from_fusion.__func__

    def from_fusion(cls, fr, dims, dom, fu, device):
        geo = orig(cls, fr, dims, dom, fu, device)
        geo.fusion = fu
        return geo

    def coarse(self):
        with torch.no_grad():
            return PT.pack_grid(PT.aggregate_fusion(self.fusion))
    PC.Geometry.from_fusion = classmethod(from_fusion)
    PC.Geometry.coarse = coarse
    return PC
