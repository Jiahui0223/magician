"""EXP Density A/B -- P1 modes of the closed-loop planner child (FROZEN_PROTOCOL.md §4). Mode from env DA_P1_MODE:
  s10s10   = p1_full.patch("sparse")  (historical B0 / E2-S10: S10 inference, S10 scoring)
  d100s10  = p1_full.patch("b2")      (historical B2 / E2-D100: D100 inference, S10 DistP1 novelty)
  d100d100 = p1_full.patch("full")    (historical B1: D100 for everything)
  s10d100  = NEW mirror of b2: model inputs (coarse grid, projection channels, decoding) from the frozen S10 fusion; a D100
             twin map (P1FusionFull of the same records) provides ONLY caster.band_dil, computed exactly as
             m_score.SceneCaster / p1_full._patch_geometry_b2 do (|v| < 0.999 trunc within the domain, 3x3x3 max pool).
The historical p1_full.py is imported, never modified."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "scripts"))
import p1_full as PF  # noqa: E402

MC, m_p1 = PF.MC, PF.m_p1
MODES = {"s10s10": "sparse", "d100s10": "b2", "d100d100": "full", "s10d100": None}


class P1FusionS10D100(PF._P1):
    """S10 model-input map (the frozen class itself) + D100 scoring map of the same records."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_map = PF.P1FusionFull(*args, **kwargs)

    def integrate_record(self, rec):
        super().integrate_record(rec)
        self.score_map.integrate_record(rec)


def _patch_geometry_score_map():
    import torch
    import torch.nn.functional as F
    import v2_plan_core as PC
    orig = PC.Geometry.from_fusion.__func__

    def from_fusion(cls, fr, dims, dom, fu, device):
        geo = orig(cls, fr, dims, dom, fu, device)
        sm = getattr(fu, "score_map", None)
        if sm is not None:
            c = geo.caster
            v = sm.input_volume().to(c.dev).to(c.dt)
            ob = (v.abs() < MC.TRUNC * 0.999) & c.domain
            c.band_dil = F.max_pool3d(ob[None, None].half(), 3, 1, 1)[0, 0] > 0
            del v, ob
            torch.cuda.empty_cache()
        return geo
    PC.Geometry.from_fusion = classmethod(from_fusion)


def patch(mode=None):
    mode = mode or os.environ["DA_P1_MODE"]
    if mode not in MODES:
        raise ValueError(f"DA_P1_MODE must be one of {sorted(MODES)}, got {mode}")
    if MODES[mode] is not None:
        PF.patch(MODES[mode])
    else:
        m_p1.P1Fusion = P1FusionS10D100
        _patch_geometry_score_map()
    return mode
