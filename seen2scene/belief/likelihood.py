"""Depth likelihood of world hypotheses (configs.belief.DepthLikelihood documents the model).

`DepthLikelihoodModel.score(worlds, frame, rays, prior_structure)` casts the new rays into every world
(belief.raycast.march_first_hit) and returns the tempered log-likelihood of every world with per-world consistency
diagnostics. A world with non-finite values gets -inf and is reported in `invalid`.
"""
import dataclasses
import math
from typing import Dict, Optional

import torch

from seen2scene.belief.observation import UNKNOWN, RayBundle
from seen2scene.belief.raycast import march_first_hit
from seen2scene.belief.state import GridFrame
from seen2scene.configs.belief import DepthLikelihood


@dataclasses.dataclass
class LikelihoodResult:
    loglik: torch.Tensor  # [K] float64 tempered log-likelihood; -inf = invalid world
    mean_ray_logprob: torch.Tensor  # [K] float64, mean over the scored rays
    inlier_rate: torch.Tensor  # [K] float64, scored rays the world explains
    free_violation_rate: torch.Tensor  # [K] float64, world surface in observed free space (pass hit / hit in front)
    missing_rate: torch.Tensor  # [K] float64, surface rays without a world hit at / in front of t_obs
    invalid: torch.Tensor  # [K] bool
    counts: Dict[str, int]

    def to_dict(self) -> Dict:
        return {"loglik": self.loglik.tolist(), "mean_ray_logprob": self.mean_ray_logprob.tolist(),
                "inlier_rate": self.inlier_rate.tolist(), "free_violation_rate": self.free_violation_rate.tolist(),
                "missing_rate": self.missing_rate.tolist(), "invalid": self.invalid.tolist(), "rays": self.counts}


@torch.no_grad()
def crosses_unknown(structure: torch.Tensor, frame: GridFrame, origins: torch.Tensor, dirs: torch.Tensor,
                    t0: torch.Tensor, t1: torch.Tensor, chunk: int = 1 << 24) -> torch.Tensor:
    """[R] bool: the segment t in [t0, t1] of a ray passes through an UNKNOWN voxel of `structure` (samples at voxel
    centres of the march, every voxel_size)."""
    dev = structure.device
    R = dirs.shape[0]
    out = torch.zeros(R, dtype=torch.bool, device=dev)
    length = (t1 - t0).clamp(min=0)
    ok = length > 0
    if not ok.any():
        return out
    vs = frame.voxel_size
    M = int(torch.ceil(length.max() / vs).item()) + 1
    lo = torch.as_tensor(frame.origin, dtype=dirs.dtype, device=dev)
    shape = torch.as_tensor(frame.shape, device=dev)
    per = max(1, chunk // M)
    unk = structure == UNKNOWN
    for s in range(0, R, per):
        sl = slice(s, min(R, s + per))
        ts = t0[sl, None] + vs * (torch.arange(M, device=dev, dtype=dirs.dtype)[None] + 0.5)
        inside = ts <= t1[sl, None]
        p = origins[sl, None] + dirs[sl, None] * ts[..., None] - lo
        ijk = torch.floor(p / vs).long()
        inb = ((ijk >= 0) & (ijk < shape)).all(-1) & inside
        ijk = torch.where(inb[..., None], ijk, torch.zeros_like(ijk))
        hit = unk[ijk[..., 0], ijk[..., 1], ijk[..., 2]] & inb
        out[sl] = hit.any(-1) & ok[sl]
    return out


class DepthLikelihoodModel:
    def __init__(self, cfg: Optional[DepthLikelihood] = None):
        self.cfg = cfg or DepthLikelihood()

    @torch.no_grad()
    def score(self, worlds: torch.Tensor, frame: GridFrame, rays: RayBundle,
              prior_structure: Optional[torch.Tensor] = None, domain: Optional[torch.Tensor] = None
              ) -> LikelihoodResult:
        """prior_structure: structure before the new rays (novel_only). domain: [D, H, W] bool, voxels the worlds
        represent; outside it a world counts as free (+truncation), as in the EXP3b scene caster."""
        c = self.cfg
        K, dev = worlds.shape[0], worlds.device
        rays = rays.to(dev)
        invalid = ~torch.isfinite(worlds.reshape(K, -1)).all(1)
        safe = torch.nan_to_num(worlds, nan=frame.truncation) if invalid.any() else worlds
        if domain is not None:
            safe = torch.where(domain.to(dev)[None], safe, torch.full_like(safe, frame.truncation))
        hits = march_first_hit(safe, frame, rays.origins, rays.dirs, step_vox=c.step_vox)
        t_hat, t_in, t_out, crosses = hits["t"].double(), hits["t_in"].double(), hits["t_out"].double(), hits["valid"]
        t_obs = rays.t_obs.double()
        vs = frame.voxel_size
        margin = c.margin_vox * vs
        blocked = crosses & (t_obs < t_in - margin)
        surface = crosses & ~blocked & (t_obs <= t_out + margin)
        passing = crosses & ~blocked & ~surface  # beyond the box or no return
        novel = torch.ones_like(crosses)
        if c.novel_only:
            if prior_structure is None:
                raise ValueError("novel_only needs the structure before the new observation")
            t_end = torch.where(surface, torch.minimum(t_obs + margin, t_out), t_out)
            novel = crosses_unknown(prior_structure.to(dev), frame, rays.origins.double(), rays.dirs.double(),
                                    t_in, t_end)
        scored = (surface | passing) & novel
        counts = {"rays": rays.n_rays, "frames": rays.n_frames, "surface": int(surface.sum()),
                  "pass": int(passing.sum()), "blocked": int(blocked.sum()), "outside": int((~crosses).sum()),
                  "scored": int(scored.sum()), "scored_surface": int((scored & surface).sum())}
        zeros = torch.zeros(K, dtype=torch.float64, device=dev)
        neg_inf = torch.full_like(zeros, -math.inf)
        if not scored.any():  # the rays say nothing new about the grid: weights stay
            return LikelihoodResult(torch.where(invalid, neg_inf, zeros), zeros.clone(), zeros + 1.0, zeros.clone(),
                                    zeros.clone(), invalid, counts)

        # surface rays: probability of the observed range bin under the world, plus a uniform outlier
        sigma = c.sigma0_vox * vs + c.sigma_rel * torch.where(surface, t_obs, torch.zeros_like(t_obs))
        half = 0.5 * c.bin_vox * vs
        seg = (t_out - t_in).clamp(min=vs) + 2 * margin
        t_ref = torch.where(surface, t_obs, torch.zeros_like(t_obs))[None]
        finite = torch.isfinite(t_hat)
        th = torch.where(finite, t_hat, torch.zeros_like(t_hat))
        p_bin = torch.special.ndtr((t_ref + half - th) / sigma) - torch.special.ndtr((t_ref - half - th) / sigma)
        p_bin = torch.where(finite, p_bin.clamp(min=0.0), torch.zeros_like(p_bin))
        log_in = math.log1p(-c.outlier_prob) + torch.log(p_bin)  # -inf where p_bin == 0
        log_out = (math.log(c.outlier_prob) + math.log(2 * half) - torch.log(seg))[None].expand_as(log_in)
        ll_surface = torch.logaddexp(log_in, log_out)
        ll_pass = torch.where(finite, torch.full_like(t_hat, math.log(c.free_violation_prob)),
                              torch.full_like(t_hat, math.log1p(-c.free_violation_prob)))
        ll_ray = torch.where(surface[None], ll_surface, ll_pass)
        ll_ray = torch.where(scored[None], ll_ray, torch.zeros_like(ll_ray))

        # tempering: every frame counts as effective_rays_per_frame observations (mean over its scored rays)
        fid = torch.unique(rays.frame_ids[scored], return_inverse=False)
        loglik = torch.zeros(K, dtype=torch.float64, device=dev)
        for f in fid:
            m = scored & (rays.frame_ids == f)
            loglik = loglik + c.effective_rays_per_frame * ll_ray[:, m].mean(1)
        loglik = torch.where(invalid, neg_inf, loglik)

        n = int(scored.sum())
        z = (th - t_ref) / sigma[None]
        inlier = (surface[None] & finite & (z.abs() <= c.inlier_sigmas)) | (passing[None] & ~finite)
        front = surface[None] & finite & (z < -c.inlier_sigmas)
        free_viol = (passing[None] & finite) | front
        missing = surface[None] & (~finite | (z > c.inlier_sigmas))
        sc_surf = max(int((scored & surface).sum()), 1)
        return LikelihoodResult(loglik, ll_ray[:, scored].mean(1), (inlier & scored[None]).sum(1).double() / n,
                                (free_viol & scored[None]).sum(1).double() / n,
                                (missing & (scored & surface)[None]).sum(1).double() / sc_surf, invalid, counts)
