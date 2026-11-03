"""Camera Pose Phase 1 -- FrontierPrior of the legal candidates (protocol.md §2, C1). No GT, no network, no new model:
it reads only the planner's own per-view ray statistics, computed by v2_plan_core.Geometry.view from the D100 P1 map
(B2 mode: the D100 struct / FREE; the S10 map only enters DistP1 novelty and is not used here).

For candidate j (its 64 x 114 planner rays): rays with cls == UNKNOWN and a finite first UNKNOWN sample d_lo < tcap are
rays that cross known-FREE space and then enter in-domain UNKNOWN space within the sensor range (rays that hit a known
TSDF crossing first are cls KNOWN and are not counted -- occlusion by observed surfaces is respected; nothing beyond the
FREE -> UNKNOWN boundary is assumed visible). Their entry points p = o + d * d_lo are quantised to coarse cells (8 fine
voxels per axis) and F_j = number of distinct cells (de-duplicated inside the view).
Diagnostics (logged, not used in the score): shares of UNKNOWN / KNOWN / EMPTY rays, mean entry distance."""
import torch

from seen2scene.futureview import geometry as G

COARSE = 8


@torch.no_grad()
def features(views, geo):
    ext = torch.as_tensor(geo.ext, dtype=torch.float32).reshape(-1).cpu()
    dims = torch.as_tensor([float(x) for x in geo.dims], dtype=torch.float32)
    cell = (ext / dims * COARSE)
    out = []
    for v in views:
        pr = v["pr"]
        cls, dlo, tcap = pr["cls"], pr["d_lo"].float(), v["tcap"].float()
        m = (cls == G.CLS_UNKNOWN) & torch.isfinite(dlo) & (dlo < tcap)
        n = int(cls.numel())
        f = {"unknown_ray_share": float(m.float().mean()) if n else 0.0,
             "known_ray_share": float((cls == G.CLS_KNOWN).float().mean()) if n else 0.0,
             "empty_ray_share": float((cls == G.CLS_EMPTY).float().mean()) if n else 0.0}
        if bool(m.any()):
            o = v["o"].float().reshape(1, 3)
            p = (o + v["d"][m].float() * dlo[m, None]).cpu()
            c = torch.floor(p / cell).long()
            f["F"] = int(torch.unique(c, dim=0).shape[0])
            f["entry_dist_mean"] = float(dlo[m].mean())
        else:
            f["F"], f["entry_dist_mean"] = 0, None
        out.append(f)
    return out


def combine(D, F, lam):
    """Score_C1 = D / max D + lam * F / max F (maxima over all legal candidates; a zero maximum -> that term is 0).
    lam == 0 returns D unchanged (identical to C0)."""
    if lam == 0:
        return list(D)
    md, mf = max(D) if D else 0.0, max(F) if F else 0
    return [(d / md if md > 0 else 0.0) + lam * (f / mf if mf > 0 else 0.0) for d, f in zip(D, F)]
