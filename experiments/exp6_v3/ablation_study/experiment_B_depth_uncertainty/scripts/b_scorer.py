"""EXP6 V3 Ablation Experiment B -- planner scorers on the saved K-sample predictions of one candidate camera (numpy,
pure functions; protocol.md v1.0 §3). No GT, no model, no P1 change: only the score changes.

Arrays per candidate (R rays): t [K, R] decoded ray distance (model units, inf = no hit), hit_u [K, R] bool (sample
decoded a surface inside the UNKNOWN part of the ray), valid [R] bool, n_u [R] (number of UNKNOWN samples along the
ray; M_unknown = n_u > 0), d_lo / d_hi [R] (ray distance of the first / last UNKNOWN sample; span = d_hi - d_lo).
"""
import warnings

import numpy as np

ALPHA = 0.5          # frozen weight of U_hit vs U_depth
LAMBDA = 1.0         # frozen weight of the normalised DepthU in P2
LAMBDA_SENS = (0.5, 2.0)
RISK_LAMBDA = 0.5    # exploratory RiskAware = mean_k - 0.5 std_k


def binary_entropy(q):
    q = np.clip(np.asarray(q, float), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return np.where((q <= 0) | (q >= 1), 0.0, h)


def depth_u_rays(t, hit_u, valid, n_u, d_lo, d_hi, alpha=ALPHA):
    """Per-ray DepthU terms of one candidate. Returns dict with u_hit, u_depth, m_unknown, u (combined), valid."""
    t = np.asarray(t, float); hit = np.asarray(hit_u, bool)
    K = t.shape[0]
    valid = np.asarray(valid, bool)
    m_unknown = (np.asarray(n_u) > 0) & valid
    q = hit.sum(0) / K
    u_hit = binary_entropy(q)
    # depth disagreement among the HIT samples of the same ray, normalised by the ray's UNKNOWN span
    span = np.asarray(d_hi, float) - np.asarray(d_lo, float)
    n_hit = hit.sum(0)
    th = np.where(hit, t, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)       # all-NaN rays (no HIT sample) are masked below
        sd = np.nanstd(th, axis=0)                       # population std over the HIT samples (nan if < 1)
    ok = (n_hit >= 2) & np.isfinite(span) & (span > 0) & np.isfinite(sd)
    u_depth = np.zeros_like(u_hit)
    u_depth[ok] = np.clip(sd[ok] / span[ok], 0.0, 1.0)
    u = m_unknown * (alpha * u_hit + (1 - alpha) * u_depth)
    return {"u_hit": u_hit, "u_depth": u_depth, "m_unknown": m_unknown, "u": u, "valid": valid}


def s_depth_u(t, hit_u, valid, n_u, d_lo, d_hi, alpha=ALPHA):
    """S_depthU(a) = mean over valid rays of M_unknown * [alpha U_hit + (1 - alpha) U_depth]."""
    r = depth_u_rays(t, hit_u, valid, n_u, d_lo, d_hi, alpha)
    nv = int(r["valid"].sum())
    return float(r["u"][r["valid"]].sum() / nv) if nv else 0.0


def minmax(x):
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo <= 0 else (x - lo) / (hi - lo)


def scores_of_state(distp1_k, depth_u, lam=LAMBDA):
    """distp1_k [n_cand, K], depth_u [n_cand] -> dict scorer -> [n_cand]."""
    dk = np.asarray(distp1_k, float)
    p0 = dk.mean(1)
    out = {"P0": p0, "P1": np.asarray(depth_u, float)}
    out["P2"] = minmax(p0) + lam * minmax(out["P1"])
    for l2 in LAMBDA_SENS:
        out[f"P2_l{l2}"] = minmax(p0) + l2 * minmax(out["P1"])
    out["RiskAware"] = p0 - RISK_LAMBDA * dk.std(1)
    return out
