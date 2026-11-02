"""V5.2-M camera utility: the V5.2 scorer (seen2scene/futureview/v52_score.py: S_visible / P_surf / P_space with the
same world-voxel de-duplication, DistP1 distance weight, S10 band_dil novelty, D100 FREE q_valid, normalisation and
combination) applied to the V5.2-M first-hit pmf. V5.2-M needs the whole 64 x 114 ray grid of a view at once (2D
spatial context), so every view is first decoded completely (model.view_belief) and the V5.2 aggregation then consumes
the stored per-ray outputs chunk by chunk through a replay adapter (exactly the tensors of view_belief, in ray order).
No flow, no depth image, no DistP1 call."""
import torch

from seen2scene.futureview import v52_score as SC52

DEFAULT = SC52.DEFAULT
combine = SC52.combine
space_weight = SC52.space_weight


class _Replay:
    """Model stand-in for v52_score.score_view: returns the next len(d) rays of a precomputed view_belief output."""

    KEYS = ("p_first", "log_surv", "a", "edges", "t_mid", "dt", "logit_b", "p_nohit")

    def __init__(self, out):
        self.out, self.pos = out, 0

    def memory(self, h):
        return None

    def belief(self, mem, o, d, tcap, ch):
        n = d.shape[0]
        sl = slice(self.pos, self.pos + n)
        self.pos += n
        return {k: self.out[k][sl] for k in self.KEYS}


@torch.no_grad()
def score_state(model, h, geo, views, k_M, z0, cfg):
    cfg = dict(DEFAULT, **(cfg or {}))
    if not views:
        return {"s_vis": [], "p_surf": [], "p_space": [], "diag": [], "state": {}}
    dev = views[0]["d"].device
    mem = model.memory(h)
    coarse = geo.coarse().to(dev)
    phi, sinfo = space_weight(coarse, cfg)
    ctx = {"dims": torch.tensor(geo.dims, device=dev), "dims_c": torch.tensor(phi.shape, device=dev), "phi": phi.reshape(-1),
           "domain": geo.domain.reshape(-1), "band": geo.caster.band_dil.reshape(-1), "struct": geo.struct.reshape(-1)}
    res = {"s_vis": [], "p_surf": [], "p_space": [], "diag": [], "state": sinfo, "free_frac_mean": []}
    for v in views:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.view_belief(h, v, mem)
        rp = _Replay(out)
        r = SC52.score_view(rp, None, dict(v, ch=torch.zeros(v["d"].shape[0], 11, device=dev)), ctx, k_M, z0, cfg)
        assert rp.pos == v["d"].shape[0], ("replay adapter consumed", rp.pos, v["d"].shape[0])
        for k in ("s_vis", "p_surf", "p_space", "diag"):
            res[k].append(r[k])
        res["free_frac_mean"].append(float(out["free_frac"].mean()))
    return res
