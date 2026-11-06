"""Queries of Experiment XA on the 16 x 32 anchor grid of the padded 64 x 128 ray map (one cell = 4 x 4 scoring rays; cells
whose rays are all padding are masked). Every builder uses only what the planner has at decision time: the candidate's
rays, the projection channels of the current P1 (class, t_known, first / last UNKNOWN sample, UNKNOWN length, segments) and
the encoder's input grid. No label, GT depth or gain enters.

Query tensor Q [V, 512, 2 + F] float32: [anchor cell index, mask, features].
  2d        features = Fourier(mean ray direction of the cell), Fourier(camera position)           (the T1 query encoding)
  ray       one representative ray per cell: the ray with the largest UNKNOWN length (ties: the position nearest to the cell
            centre, fixed order); features = its ray-encoder context, its 11 projection channels, its sub-cell offset, an
            UNKNOWN flag, Fourier(direction), Fourier(camera position), Fourier(midpoint of its UNKNOWN envelope, or of its
            known hit / cap when it has no UNKNOWN sample)
  interval  the UNKNOWN connected intervals of the same representative rays before t_known, read from the encoder input
            grid (a 4x cell is UNKNOWN when 1 - band - free >= 0.5 inside the domain; samples every 2 fine voxels, clipped
            to the ray's current UNKNOWN envelope); at most 4 per ray; a ray whose envelope shows no run on the grid gets its
            envelope as one interval. Selection of <= 512: all first intervals in raster order, then the second ones, ...
            (every ray with an interval keeps its first one). features = the ray features with the interval midpoint +
            start, end, length (relative to the cap), share of the ray's UNKNOWN length, order one-hot, intervals of the ray,
            envelope-fallback flag. Several intervals of one ray are averaged into the ray's cell after the attention.
Local samples (multi-scale memory): 16 points along the query's ray, uniformly over
  2d: [0, cap] of the cell's mean ray;  ray: its UNKNOWN envelope +- one 4x cell (no UNKNOWN: [0, known hit / cap]);
  interval: the interval +- one 4x cell.  Each sample = [F4 (trilinear), mu (trilinear), band, free, valid of the 4x grid,
  position along the range, t / cap, Fourier(position)], masked outside the grid.
"""
import math

import torch
import torch.nn.functional as F

from seen2scene.futureview import geometry as G
from seen2scene.v3.cross_attention import fourier

GH, GW, GW_PAD = 64, 114, 128
CELL = 4
QH, QW = GH // CELL, GW_PAD // CELL
NQ = QH * QW
VS = 0.011
MARGIN = CELL * VS                     # one 4x cell (model units)
_order = sorted(range(CELL * CELL), key=lambda i: ((i // CELL - 1.5) ** 2 + (i % CELL - 1.5) ** 2, i))
RANK = torch.zeros(CELL * CELL, dtype=torch.long)
for _r, _i in enumerate(_order):
    RANK[_i] = _r


def _cells(t, value):
    """[V, 64, 114] -> [V, 512, 16] (cell raster, local row-major), padding rays = value."""
    V = t.shape[0]
    t = F.pad(t, (0, GW_PAD - GW), value=value)
    return t.reshape(V, QH, CELL, QW, CELL).permute(0, 1, 3, 2, 4).reshape(V, NQ, CELL * CELL)


def rep_rays(ch, tcap):
    """ch [V, 11, R], tcap [V, R] -> ray index [V, 512] (row-major 64 x 114), valid [V, 512], local position [V, 512]."""
    V = ch.shape[0]
    lu = _cells((ch[:, 6].float() * tcap.float()).reshape(V, GH, GW), -1.0)
    m = lu.max(-1, keepdim=True).values
    loc = (RANK.to(lu.device)[None, None] + (lu < m).long() * 100).argmin(-1)
    cell = torch.arange(NQ, device=lu.device)[None]
    row = (cell // QW) * CELL + loc // CELL
    col = ((cell % QW) * CELL + loc % CELL).clamp(max=GW - 1)
    return row * GW + col, m[..., 0] >= 0, loc


def _pack(cell, mask, feat):
    return torch.cat([cell.float()[..., None], mask.float()[..., None], feat.float()], -1)


def spatial_queries(o, d, tcap, ext, n_freq):
    V = d.shape[0]
    dev = d.device
    dd = F.pad(d.float().reshape(V, GH, GW, 3).permute(0, 3, 1, 2), (0, GW_PAD - GW))
    cnt = F.avg_pool2d(F.pad(torch.ones(V, 1, GH, GW, device=dev), (0, GW_PAD - GW)), CELL).reshape(V, NQ) * CELL * CELL
    dm = F.normalize(F.avg_pool2d(dd, CELL).permute(0, 2, 3, 1).reshape(V, NQ, 3), dim=-1)
    tc = F.avg_pool2d(F.pad(tcap.float().reshape(V, 1, GH, GW), (0, GW_PAD - GW)), CELL).reshape(V, NQ) * CELL * CELL / cnt.clamp(min=1.0)
    e = ext.to(dev).float()
    pos = (2.0 * o.float() / e - 1.0)[:, None].expand(V, NQ, 3)
    mask = cnt > 0
    cell = torch.arange(NQ, device=dev)[None].expand(V, NQ)
    Q = _pack(cell, mask, torch.cat([fourier(dm, n_freq), fourier(pos, n_freq)], -1))
    aux = {"o": o.float()[:, None].expand(V, NQ, 3), "d": dm, "tlo": torch.zeros_like(tc), "thi": tc, "tcap": tc, "mask": mask}
    return Q, aux


def _ray_base(o, d, tcap, ch, rf, ext, n_freq):
    V = d.shape[0]
    ridx, valid, loc = rep_rays(ch, tcap)
    take = lambda t: t.gather(1, ridx[..., None].expand(-1, -1, t.shape[-1]))  # noqa: E731
    b = {"d": take(d.float()), "tcap": tcap.float().gather(1, ridx), "ch": take(ch.float().transpose(1, 2)), "rf": take(rf.float()), "valid": valid}
    b["off"] = torch.stack([loc // CELL, loc % CELL], -1).float() / (CELL - 1) * 2.0 - 1.0
    b["has"] = (b["ch"][..., 6] > 0) & valid
    b["ext"] = ext.to(d.device).float()
    b["o"] = o.float()[:, None].expand(V, NQ, 3)
    b["pos"] = 2.0 * b["o"] / b["ext"] - 1.0
    return b


def _ray_feat(b, n_freq, tmid, sel=None):
    """Ray features [V, N, 222] of the rays `sel` ([V, N] cell ids; None = all 512), midpoint ray length tmid [V, N]."""
    g = (lambda t: t) if sel is None else (lambda t: t.gather(1, sel[..., None].expand(-1, -1, t.shape[-1])))
    d, o = g(b["d"]), g(b["o"])
    has = b["has"] if sel is None else b["has"].gather(1, sel)
    pmid = (2.0 * (o + d * tmid[..., None]) / b["ext"] - 1.0).clamp(-1.0, 1.0)
    return torch.cat([g(b["rf"]), g(b["ch"]), g(b["off"]), has.float()[..., None], fourier(d, n_freq), fourier(g(b["pos"]), n_freq),
                      fourier(pmid, n_freq)], -1)


def ray_queries(o, d, tcap, ch, rf, ext, n_freq):
    V = d.shape[0]
    b = _ray_base(o, d, tcap, ch, rf, ext, n_freq)
    tk, dlo, dhi, tc, has = b["ch"][..., 3].clamp(max=1.0), b["ch"][..., 4], b["ch"][..., 5], b["tcap"], b["has"]
    tmid = torch.where(has, 0.5 * (dlo + dhi), tk) * tc
    cell = torch.arange(NQ, device=d.device)[None].expand(V, NQ)
    Q = _pack(cell, b["valid"], _ray_feat(b, n_freq, tmid))
    tlo = torch.where(has, dlo * tc - MARGIN, torch.zeros_like(tc)).clamp(min=0.0)
    thi = torch.minimum(torch.where(has, dhi * tc + MARGIN, tk * tc), tc)
    aux = {"o": b["o"], "d": b["d"], "tlo": tlo, "thi": thi, "tcap": tc, "mask": b["valid"]}
    return Q, aux


@torch.no_grad()
def unknown_intervals(x, valid4, b, step=2 * VS, max_int=4, m_cap=1024):
    """Runs of UNKNOWN 4x cells along the representative rays -> t0, t1 [V, 512, max_int], exist [V, 512, max_int] bool,
    from_stats [V, 512] bool (the ray's envelope used because the grid shows no run)."""
    V = b["d"].shape[0]
    dev = b["d"].device
    u4 = (((1.0 - x[:, 1:2].float() - x[:, 2:3].float()) >= 0.5) & (valid4 > 0)).float()
    T0 = torch.zeros(V, NQ, max_int, device=dev)
    T1 = torch.zeros(V, NQ, max_int, device=dev)
    EX = torch.zeros(V, NQ, max_int, dtype=torch.bool, device=dev)
    for v in range(V):
        d, tc, ch, has = b["d"][v], b["tcap"][v], b["ch"][v], b["has"][v]
        tmin, tmax = G.entry_exit(b["o"][v], d, b["ext"])
        tk, dlo, dhi = ch[:, 3].clamp(max=1.0), ch[:, 4], ch[:, 5]
        hi = torch.minimum(torch.minimum(tmax, tc), torch.minimum(tk, dhi) * tc + step)
        lo = torch.maximum(tmin, dlo * tc - step)
        ok = has & (hi > lo)
        if not ok.any():
            continue
        M = int(min(m_cap, max(1, math.ceil(float(torch.where(ok, hi - lo, torch.zeros_like(hi)).max()) / step))))
        ts = lo[:, None] + (torch.arange(M, device=dev) + 0.5) * step
        g = 2.0 * (b["o"][v][:, None] + d[:, None] * ts[..., None]) / b["ext"] - 1.0
        grid = torch.stack([g[..., 2], g[..., 1], g[..., 0]], -1)[None, :, :, None]
        u = F.grid_sample(u4, grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0, :, :, 0] > 0.5
        u &= (ts < hi[:, None]) & ok[:, None]
        start = u & ~F.pad(u[:, :-1], (1, 0), value=False)
        rid = torch.cumsum(start.int(), 1) * u.int()
        for k in range(max_int):
            mk = rid == k + 1
            EX[v, :, k] = mk.any(1)
            T0[v, :, k] = torch.where(mk, ts, torch.full_like(ts, float("inf"))).amin(1) - 0.5 * step
            T1[v, :, k] = torch.where(mk, ts, torch.full_like(ts, float("-inf"))).amax(1) + 0.5 * step
    from_stats = b["has"] & ~EX.any(-1)
    tc = b["tcap"]
    T0[..., 0] = torch.where(from_stats, b["ch"][..., 4] * tc, T0[..., 0])
    T1[..., 0] = torch.where(from_stats, b["ch"][..., 5] * tc, T1[..., 0])
    EX[..., 0] |= from_stats
    T0 = torch.where(EX, T0, torch.zeros_like(T0)).clamp(min=0.0)
    T1 = torch.minimum(torch.where(EX, T1, torch.zeros_like(T1)), tc[..., None])
    return T0, torch.maximum(T1, T0), EX, from_stats


def interval_queries(x, valid4, o, d, tcap, ch, rf, ext, n_freq, max_int=4):
    V = d.shape[0]
    b = _ray_base(o, d, tcap, ch, rf, ext, n_freq)
    T0, T1, EX, from_stats = unknown_intervals(x, valid4, b, max_int=max_int)
    ex = EX.permute(0, 2, 1).reshape(V, max_int * NQ)                       # order-major: first intervals, then second, ...
    order = torch.argsort((~ex).int(), dim=1, stable=True)[:, :NQ]
    mask = ex.gather(1, order)
    k, cell = order // NQ, order % NQ
    pick = lambda t: t.gather(1, cell[..., None].expand(-1, -1, max_int)).gather(2, k[..., None])[..., 0]  # noqa: E731
    t0, t1 = pick(T0), pick(T1)
    tc = b["tcap"].gather(1, cell)
    tot = ((T1 - T0) * EX).sum(-1).gather(1, cell).clamp(min=1e-6)
    nint = EX.sum(-1).gather(1, cell).float() / max_int
    feat = torch.cat([_ray_feat(b, n_freq, 0.5 * (t0 + t1), cell), (t0 / tc)[..., None], (t1 / tc)[..., None], ((t1 - t0) / tc)[..., None],
                      ((t1 - t0) / tot)[..., None], F.one_hot(k, max_int).float(), nint[..., None], from_stats.gather(1, cell).float()[..., None]], -1)
    Q = _pack(cell, mask, feat * mask[..., None])
    g3 = lambda t: t.gather(1, cell[..., None].expand(-1, -1, 3))  # noqa: E731
    aux = {"o": g3(b["o"]), "d": g3(b["d"]), "tlo": (t0 - MARGIN).clamp(min=0.0), "thi": torch.minimum(t1 + MARGIN, tc), "tcap": tc, "mask": mask,
           "stats": {"intervals": EX.sum((1, 2)), "selected": mask.sum(1), "rays_with_interval": EX.any(-1).sum(1),
                     "rays_covered": (F.one_hot(cell, NQ) * mask[..., None]).amax(1).sum(1), "from_stats": from_stats.sum(1), "rays_valid": b["valid"].sum(1)}}
    return Q, aux


def local_tokens(h, aux, n_local, n_freq_s=4):
    """-> L [V, Nq, n_local, 32 + 16 + 3 + 2 + 2 n_freq_s + 1] float32 (last entry = mask)."""
    V, Nq = aux["tlo"].shape
    dev = aux["tlo"].device
    ext = h["ext"].to(dev).float()
    s = (torch.arange(n_local, device=dev) + 0.5) / n_local
    t = aux["tlo"][..., None] + (aux["thi"] - aux["tlo"])[..., None] * s
    g = 2.0 * (aux["o"][:, :, None] + aux["d"][:, :, None] * t[..., None]) / ext - 1.0
    grid = torch.stack([g[..., 2], g[..., 1], g[..., 0]], -1).reshape(1, V * Nq, n_local, 1, 3)
    with torch.autocast("cuda", enabled=False):
        a = F.grid_sample(h["f4"].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        m = F.grid_sample(h["mu"].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        c = F.grid_sample(torch.cat([h["x"][:, 1:3].float(), h["valid4"].float()], 1), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    inside = ((g > -1) & (g < 1)).all(-1) & aux["mask"][..., None]
    feat = torch.cat([a, m, c], 1)[0, ..., 0].permute(1, 2, 0).reshape(V, Nq, n_local, -1) * (c[0, 2, :, :, 0].reshape(V, Nq, n_local, 1) > 0)
    geo = torch.cat([s.expand(V, Nq, n_local)[..., None], (t / aux["tcap"].clamp(min=1e-6)[..., None])[..., None],
                     fourier((2.0 * s - 1.0)[:, None], n_freq_s).expand(V, Nq, n_local, -1)], -1)
    return torch.cat([feat, geo, inside.float()[..., None]], -1)
