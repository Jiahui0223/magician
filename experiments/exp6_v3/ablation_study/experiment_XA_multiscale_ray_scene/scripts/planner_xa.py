"""Experiment XA -- planner-side core: exp6_v3/shared/planner_v3.install() (V3 planner patch: D100 fusion kept on the geometry, 4x grid as the
encoder input) with the model loader replaced by seen2scene.v3.xa.xa_model.load_core, which builds an XA model for an XA checkpoint and the V3
model for a V3 checkpoint (so V3-T1 runs through this chain unchanged: regression). Nothing else of the planner changes."""
import planner_v3 as PV3

from seen2scene.v3.xa import xa_model as XM


class XACore:
    def __init__(self, ckpt_path, device, expect_sha=None, steps=10):
        self.model, self.pred, self.ckpt_sha = XM.load_core(ckpt_path, device, steps, expect_sha)
        self.dev, self.ckpt = device, ckpt_path


def install():
    PC = PV3.install()
    PC.Core = XACore
    return PC
