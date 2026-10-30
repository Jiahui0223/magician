"""Deterministic candidate-view projection on the observed partial TSDF, label encoding and decoding (EXP5 Task 1,
protocol §2.3-§2.4, architecture §2 / §5).

Sample positions are those of the frozen SceneCaster.cast (exp3b m_score): t_min = entry into the scene-grid box
(>= 0), t_max = min(exit, t_cap), ts_j = t_min + h j with h = VS / 2, same float32 expressions. A sample is UNKNOWN
when its voxel floor(p / VS) lies inside the grid and the representation domain, has P1 state UNKNOWN and ts_j <
t_known (first observed-surface crossing, computed by the caller with SceneCaster.cast on the observation-only cast
volume). Hits are decoded onto UNKNOWN samples only, so a predicted hit can never lie in observed EMPTY, behind the
known surface or outside the domain.
"""
import math

import torch

VS = 0.011
TRUNC = 3 * VS
H_STEP = 0.5 * VS
EMPTY, UNKNOWN, BAND = 0, 1, 2
PATCH_H = 256 * VS                         # model height of every scene grid (2.816)
NEAR_CLIP = 0.5                            # runtime z-depth
CLS_UNKNOWN, CLS_KNOWN, CLS_EMPTY = 0, 1, 2
# label categories (protocol §2.4); HIT / NOHIT / KNOWN are training targets on M_unknown rays
CAT = {"OUT": 0, "NOHIT": 1, "KNOWN": 2, "HIT": 3, "BLOCKED": 4, "NEAR": 5,
       "INF_EMPTY": 6, "INF_BAND": 7, "INF_OOD": 8, "INF_OTHER": 9}
N_CH = 11


def entry_exit(o, d, ext):
    """SceneCaster._entry_exit (o [3] or [R, 3], d [R, 3], ext [3] = dims * VS)."""
    o = o.expand_as(d) if o.dim() == 1 else o
    inv = 1.0 / torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    ta, tb = (0.0 - o) * inv, (ext - o) * inv
    return torch.minimum(ta, tb).amax(-1).clamp(min=0.0), torch.maximum(ta, tb).amin(-1)


def march_range(o, d, tcap, ext):
    """(t_min, t_max, valid, M) exactly as SceneCaster.cast."""
    tmin, tmax = entry_exit(o[None].expand_as(d), d, ext)
    tmax = torch.minimum(tmax, tcap)
    valid = tmax > tmin
    M = int(torch.ceil(((tmax - tmin).clamp(min=0).max()) / H_STEP).item()) + 2
    return tmin, tmax, valid, M


def sample_ts(tmin, M):
    steps = torch.arange(M, device=tmin.device)
    return tmin[:, None] + H_STEP * steps[None]


def unknown_samples(o, d, tmin, tmax, valid, M, t_known, struct, domain):
    """Boolean [R, M] of UNKNOWN samples, and ts [R, M]. struct int8 [X, Y, Z] (P1 structure), domain bool [X, Y, Z]."""
    dims = struct.shape
    ts = sample_ts(tmin, M)
    alive = (ts <= tmax[:, None]) & valid[:, None] & (ts < t_known[:, None])
    p = o[None, None] + d[:, None] * ts[..., None]
    ijk = torch.floor(p / VS).long()
    inside = alive & (ijk >= 0).all(-1) & (ijk[..., 0] < dims[0]) & (ijk[..., 1] < dims[1]) & (ijk[..., 2] < dims[2])
    unk = torch.zeros_like(inside)
    if inside.any():
        q = ijk[inside]
        flat = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
        unk[inside] = (struct.reshape(-1)[flat] == UNKNOWN) & domain.reshape(-1)[flat]
    return unk, ts


def ray_stats(unk, ts):
    """n_u, d_lo, d_hi (ts of the first / last UNKNOWN sample; inf if none), n_seg."""
    n_u = unk.sum(1)
    any_ = n_u > 0
    first = torch.argmax(unk.int(), dim=1)
    last = unk.shape[1] - 1 - torch.argmax(unk.flip(1).int(), dim=1)
    inf = torch.full_like(ts[:, 0], float("inf"))
    d_lo = torch.where(any_, ts.gather(1, first[:, None])[:, 0], inf)
    d_hi = torch.where(any_, ts.gather(1, last[:, None])[:, 0], inf)
    n_seg = (unk[:, 1:] & ~unk[:, :-1]).sum(1) + unk[:, 0].long()
    return n_u, d_lo, d_hi, n_seg


def ray_class(n_u, t_known):
    cls = torch.full_like(n_u, CLS_EMPTY)
    cls = torch.where(torch.isfinite(t_known), torch.full_like(cls, CLS_KNOWN), cls)
    return torch.where(n_u > 0, torch.full_like(cls, CLS_UNKNOWN), cls)


def project(o, d, tcap, ext, t_known, struct, domain, chunk=2048):
    """All projection quantities of one view. o [3], d [R, 3], tcap [R] (caster dtype, model units). Returns a dict of
    per-ray tensors + the march range; unknown samples are recomputed per chunk (not kept)."""
    tmin, tmax, valid, M = march_range(o, d, tcap, ext)
    R = d.shape[0]
    n_u = torch.zeros(R, dtype=torch.long, device=d.device)
    d_lo = torch.full((R,), float("inf"), dtype=d.dtype, device=d.device)
    d_hi, n_seg = d_lo.clone(), torch.zeros_like(n_u)
    for s in range(0, R, chunk):
        sl = slice(s, min(R, s + chunk))
        unk, ts = unknown_samples(o, d[sl], tmin[sl], tmax[sl], valid[sl], M, t_known[sl], struct, domain)
        n_u[sl], d_lo[sl], d_hi[sl], n_seg[sl] = ray_stats(unk, ts)
    return {"tmin": tmin, "tmax": tmax, "valid": valid, "M": M, "t_known": t_known, "n_u": n_u, "d_lo": d_lo,
            "d_hi": d_hi, "n_seg": n_seg, "cls": ray_class(n_u, t_known)}


def encode_labels(o, d, pr, tstar, dz_rt, k_M, struct, domain, chunk=2048):
    """Label category, b in {-1, 0, +1} (0 = no target) and y of every ray (protocol §2.4).
    tstar: first GT intersection along the ray in model units (inf = none); dz_rt: z-depth per runtime ray length."""
    tmin, tmax, valid, M = pr["tmin"], pr["tmax"], pr["valid"], pr["M"]
    R, dims = d.shape[0], struct.shape
    cat = torch.full((R,), CAT["NOHIT"], dtype=torch.long, device=d.device)
    s_val = torch.ones(R, dtype=d.dtype, device=d.device)
    fin = torch.isfinite(tstar)
    blocked = fin & (tstar < tmin - VS)
    near = fin & ~blocked & (tstar * k_M * dz_rt.to(tstar.dtype) < NEAR_CLIP)
    nohit = ~fin | (tstar > tmax)
    known = ~blocked & ~near & ~nohit & (tstar >= pr["t_known"] - VS)
    rest = ~blocked & ~near & ~nohit & ~known
    hit = torch.zeros_like(rest)
    for s in range(0, R, chunk):
        sl = slice(s, min(R, s + chunk))
        r_idx = torch.nonzero(rest[sl])[:, 0]
        if len(r_idx) == 0:
            continue
        unk, ts = unknown_samples(o, d[sl], tmin[sl], tmax[sl], valid[sl], M, pr["t_known"][sl], struct, domain)
        unk, ts, tq = unk[r_idx], ts[r_idx], tstar[sl][r_idx]
        j0 = torch.round((tq - ts[:, 0]) / H_STEP).long()
        best_j = torch.full_like(j0, -1)
        best_e = torch.full_like(tq, float("inf"))
        for off in (0, -1, 1, -2, 2, -3, 3):          # nearest UNKNOWN sample within one voxel (ties -> earlier)
            j = j0 + off
            ok = (j >= 0) & (j < M)
            jc = j.clamp(0, M - 1)
            e = (ts.gather(1, jc[:, None])[:, 0] - tq).abs()
            good = ok & unk.gather(1, jc[:, None])[:, 0] & (e <= VS)
            better = good & ((e < best_e) | ((e == best_e) & (j < best_j)))
            best_e = torch.where(better, e, best_e)
            best_j = torch.where(better, j, best_j)
        found = best_j >= 0
        cs = torch.cumsum(unk.long(), dim=1)
        rank = cs.gather(1, best_j.clamp(min=0)[:, None])[:, 0] - 1
        n_u = unk.sum(1)
        g = r_idx + s
        hit[g[found]] = True
        s_val[g[found]] = ((rank[found].to(d.dtype) + 0.5) / n_u[found].to(d.dtype))
    infeas = rest & ~hit
    if infeas.any():
        q = torch.nonzero(infeas)[:, 0]
        p = o[None] + d[q] * tstar[q][:, None]
        ijk = torch.floor(p / VS).long()
        inside = (ijk >= 0).all(-1) & (ijk[:, 0] < dims[0]) & (ijk[:, 1] < dims[1]) & (ijk[:, 2] < dims[2])
        sub = torch.full_like(q, CAT["INF_OOD"])
        if inside.any():
            qi = ijk[inside]
            flat = (qi[:, 0] * dims[1] + qi[:, 1]) * dims[2] + qi[:, 2]
            st = struct.reshape(-1)[flat].long()
            dm = domain.reshape(-1)[flat]
            c = torch.where(st == EMPTY, CAT["INF_EMPTY"], torch.where(st == BAND, CAT["INF_BAND"], CAT["INF_OTHER"]))
            sub[inside] = torch.where(dm, c, torch.full_like(c, CAT["INF_OOD"]))
        cat[q] = sub
    cat[known] = CAT["KNOWN"]
    cat[hit] = CAT["HIT"]
    cat[near] = CAT["NEAR"]
    cat[blocked] = CAT["BLOCKED"]
    cat[~valid] = CAT["OUT"]
    m_unk = pr["n_u"] > 0
    b = torch.zeros(R, dtype=torch.int8, device=d.device)
    b[m_unk & (cat == CAT["HIT"])] = 1
    b[m_unk & ((cat == CAT["NOHIT"]) | (cat == CAT["KNOWN"]))] = -1
    y = torch.where(cat == CAT["HIT"], 2.0 * s_val - 1.0, torch.ones_like(s_val))
    return {"cat": cat, "b": b, "y": y}


def stats_fp16(pr, tcap):
    """Stored projection statistics (protocol §2.3): t_known, d_lo, d_hi, L_u relative to t_cap, as float16 (the exact
    values the training data holds; the planner rounds the same way, so train and deploy inputs are identical)."""
    one = torch.ones_like(tcap)

    def rel(x):
        return torch.where(torch.isfinite(x), (x / tcap).clamp(0.0, 1.0), one)
    lu = (pr["n_u"].to(tcap.dtype) * H_STEP / tcap).clamp(0.0, 1.0)
    return torch.stack([rel(pr["t_known"]), rel(pr["d_lo"]), rel(pr["d_hi"]), lu]).half()


def channels_from_stats(cls, st16, nseg, d_z, cam_z, tcap):
    """Input channels [..., 11, R] (architecture §2) from class [..., R], float16 stats [..., 4, R], n_seg [..., R],
    ray direction z [..., R], camera model z [...] and t_cap [..., R]; float32."""
    oh = torch.nn.functional.one_hot(cls.long(), 3).movedim(-1, -2).float()
    rest = torch.stack([nseg.clamp(max=3).float() / 3.0, d_z.float(),
                        (cam_z.float() / PATCH_H)[..., None].expand_as(tcap), torch.log(tcap.float())], -2)
    return torch.cat([oh, st16.float(), rest], -2)


def channels(pr, tcap, d, o):
    """Input channels [11, R] of one view, through the same float16 statistics as the training data."""
    return channels_from_stats(pr["cls"], stats_fp16(pr, tcap), pr["n_seg"], d[:, 2], o[2], tcap)


def decode(b_hat, y_hat, o, d, pr, struct, domain, chunk=2048, eps=1e-6):
    """Hits of K samples (b_hat, y_hat [K, R]) -> ray distance t [K, R] (model units, inf = no hit), hit_in_unknown
    [K, R]. Non-M_unknown rays and samples without an UNKNOWN hit take the projection outcome (t_known or no hit)."""
    K, R = b_hat.shape
    tk = pr["t_known"]
    t_out = tk[None].expand(K, R).clone()
    hit_u = (b_hat > 0) & (y_hat < 1.0) & (pr["n_u"] > 0)[None]
    s = ((y_hat.to(tk.dtype) + 1.0) / 2.0).clamp(0.0, 1.0 - eps)
    for st in range(0, R, chunk):
        sl = slice(st, min(R, st + chunk))
        hs = hit_u[:, sl]
        if not hs.any():
            continue
        unk, ts = unknown_samples(o, d[sl], pr["tmin"][sl], pr["tmax"][sl], pr["valid"][sl], pr["M"], tk[sl],
                                  struct, domain)
        cs = torch.cumsum(unk.long(), dim=1)                                 # [r, M]
        n_u = pr["n_u"][sl]
        i = torch.floor(s[:, sl] * n_u[None].to(s.dtype)).long().clamp(min=0)
        i = torch.minimum(i, (n_u[None] - 1).clamp(min=0))                    # [K, r]
        j = torch.searchsorted(cs, (i + 1).T.contiguous())                   # [r, K]: first sample with cs >= i + 1
        j = j.clamp(max=pr["M"] - 1)
        t_hat = ts.gather(1, j).T                                            # [K, r]
        t_out[:, sl] = torch.where(hs, t_hat, t_out[:, sl])
    return t_out, hit_u


def coarse_grid(struct, domain, block=8, pad_multiple=4):
    """uint8 [3, X', Y', Z'] block fractions (EMPTY, BAND, in-domain) x 255 of the P1 structure (architecture §1)."""
    X, Y, Z = struct.shape
    dims_c = [int(math.ceil(n / block / pad_multiple) * pad_multiple) if a < 2 else n // block
              for a, n in enumerate((X, Y, Z))]
    pad = (0, 0, 0, dims_c[1] * block - Y, 0, dims_c[0] * block - X)
    out = []
    for m in ((struct == EMPTY) & domain, (struct == BAND) & domain, domain):
        v = torch.nn.functional.pad(m.to(torch.half if m.is_cuda else torch.float)[None, None], pad)
        out.append(torch.nn.functional.avg_pool3d(v, block)[0, 0].float())   # k / 512 is exact in half
    return torch.round(torch.stack(out) * 255.0).to(torch.uint8)
