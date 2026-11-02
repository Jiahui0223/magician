"""Shared fixtures of the EXP6 V5.2 G0 tests (seen2scene env, CPU): an analytic belief model with walls along the rays
(known per-wall opacity and surface belief) and a fake planner geometry (fine VS grid, domain, S10 band_dil, D100
structure, coarse grid), so the scoring semantics can be checked against exact expectations."""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
import torch  # noqa: E402

from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import v52_model as V52  # noqa: E402
from seen2scene.futureview import v52_score as SC  # noqa: E402

VS = V52.VS


class Synth:
    """Walls at distances `walls` (model units) on every ray; wall k stops a ray that reaches it with probability opac[k]
    (hazard a = -log(1 - opac) inside the interval holding the wall); surface belief b = bel[k] within Delta / 2 of a wall,
    b_out elsewhere."""

    def __init__(self, walls, opac, bel, n_bins=256, c_vox=16, b_out=0.01, delta_vox=8):
        self.walls, self.opac, self.bel = walls, opac, bel
        self.n_bins, self.c_bins, self.b_out, self.delta = n_bins, c_vox * VS, b_out, delta_vox * VS

    def memory(self, h):
        return None

    def belief(self, mem, o, d, tcap, ch, edges=None):
        edges = V52.intervals(tcap.float(), self.n_bins, self.c_bins) if edges is None else edges
        t_mid = 0.5 * (edges[:, 1:] + edges[:, :-1])
        dt = edges[:, 1:] - edges[:, :-1]
        a = torch.zeros_like(t_mid)
        lb = torch.full_like(t_mid, math.log(self.b_out / (1 - self.b_out)))
        for tw, op, bb in zip(self.walls, self.opac, self.bel):
            inside = (edges[:, :-1] <= tw) & (edges[:, 1:] > tw)
            a = a + inside.float() * (-math.log(max(1.0 - op, 1e-12)))
            near = (t_mid - tw).abs() <= self.delta / 2
            lb = torch.where(near, torch.full_like(lb, math.log(bb / (1 - bb))), lb)
        out = V52.firsthit(a)
        out.update({"edges": edges, "t_mid": t_mid, "dt": dt.clamp(min=1e-6), "logit_b": lb, "z": None})
        return out


class Caster:
    pass


class FakeGeo:
    def __init__(self, dims=(1000, 64, 64)):
        self.dims = list(dims)
        self.domain = torch.ones(dims, dtype=torch.bool)
        self.struct = torch.full(dims, G.UNKNOWN, dtype=torch.uint8)
        self.caster = Caster()
        self.caster.band_dil = torch.zeros(dims, dtype=torch.bool)

    def coarse(self):
        return G.coarse_grid(self.struct, self.domain)

    def mark(self, attr, t0, t1, o, value=True):
        """Mark the fine voxels of the x-slab [o_x + t0, o_x + t1) (rays along +x) in band_dil ('band') or struct ('free')."""
        i0, i1 = int((o[0] + t0) / VS), int((o[0] + t1) / VS) + 1
        if attr == "band":
            self.caster.band_dil[i0:i1] = value
        elif attr == "free":
            self.struct[i0:i1] = G.EMPTY
        elif attr == "known_band":
            self.struct[i0:i1] = G.BAND


def views_x(n_rays=64, o=(0.05, 0.35, 0.35), spread=0.02, tcap=9.4, identical=False, seed=0):
    """One view of n rays around +x from o (model units); dz = 1 (runtime z-depth = ray length)."""
    g = torch.Generator().manual_seed(seed)
    d = torch.zeros(n_rays, 3)
    d[:, 0] = 1.0
    if not identical:
        d[:, 1:] = (torch.rand(n_rays, 2, generator=g) - 0.5) * spread
    d = torch.nn.functional.normalize(d, dim=-1)
    return {"o": torch.tensor(o), "d": d, "tcap": torch.full((n_rays,), tcap), "dz": torch.ones(n_rays), "ch": torch.zeros(n_rays, 11)}


def score(model, geo, view, k_M=1.0, z0=1.0, **cfg):
    c = dict(SC.DEFAULT, **cfg)
    res = SC.score_state(model, None, geo, [view], k_M, z0, c)
    return {k: (v[0] if isinstance(v, list) else v) for k, v in res.items()}


def run(tests):
    bad = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:                     # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print("ALL PASS" if bad == 0 else f"{bad} FAILED")
    return bad
