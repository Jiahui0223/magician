"""Belief state shared by every belief updater (C0 particle update, C1 latent update, later variants).

A belief is a weighted set of K world hypotheses of one grid (a patch or a scene domain):
- worlds: dense TSDF [K, D, H, W] in the convention of decoded Seen2Scene worlds (band values, +truncation
  elsewhere; `TSDFPatch.to_dense()`), and/or
- latents: dense generator latents [K, C, X, Y, Z] on the latent grid (factor 8, the grid of
  `Net.randn_like`), the representation the frozen VAE decoder turns into worlds;
- log_weights: normalised log particle weights [K] (float64);
- particle_ids / ancestors / seed_keys: identity, lineage and generator seed key of every particle, so a refreshed
  or resampled particle can be traced and regenerated with the same seed (`tools.seeding.seed_world`).
`obs_version` counts the observations the belief has been conditioned on. Index (i, j, k) of a grid is world
(x, y, z) = origin + ((i, j, k) + 0.5) * voxel_size (the convention of active.raycast.PatchRaycaster).
"""
import dataclasses
import math
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import torch


@dataclasses.dataclass(frozen=True)
class GridFrame:
    """Geometry of the world grid. origin = world position of the lower corner of voxel (0, 0, 0), metres."""

    origin: Tuple[float, float, float]
    voxel_size: float
    shape: Tuple[int, int, int]
    truncation: float
    latent_factor: int = 8

    @classmethod
    def front3d_patch(cls, bbox_min: Sequence[float], size: int = 256) -> "GridFrame":
        """A 3D-FRONT target patch as used by active.raycast / active.episode (0.011 m voxels, 3-voxel band)."""
        return cls(tuple(float(x) for x in bbox_min[:3]), 0.011, (size,) * 3, 0.033)

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return tuple(math.ceil(s / self.latent_factor) for s in self.shape)

    @property
    def extent(self) -> Tuple[float, float, float]:
        return tuple(s * self.voxel_size for s in self.shape)


class UpdateDiagnostics(dict):
    """Diagnostics of one update; a plain dict of JSON-serialisable values (tensors are converted)."""

    def set(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = _jsonable(v)
        return self


def _jsonable(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        return v.item() if v.numel() == 1 and v.dim() == 0 else v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v


def normalize_log_weights(log_w: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Normalised float64 log weights; (uniform, True) when no weight is finite (all -inf / NaN)."""
    lw = log_w.detach().to(torch.float64)
    lw = torch.where(torch.isnan(lw), torch.full_like(lw, -math.inf), lw)
    if not torch.isfinite(lw).any():
        return torch.full_like(lw, -math.log(lw.numel())), True
    return lw - torch.logsumexp(lw, 0), False


def effective_sample_size(log_w: torch.Tensor) -> float:
    """ESS = 1 / sum w_k^2 of the normalised weights (K for uniform, 1 for a single particle)."""
    lw, _ = normalize_log_weights(log_w)
    return float(torch.exp(-torch.logsumexp(2 * lw, 0)))


@dataclasses.dataclass
class BeliefState:
    frame: GridFrame
    log_weights: torch.Tensor
    particle_ids: torch.Tensor
    seed_keys: torch.Tensor
    ancestors: torch.Tensor
    worlds: Optional[torch.Tensor] = None
    latents: Optional[torch.Tensor] = None
    obs_version: int = 0
    step: int = 0
    next_particle_id: int = 0
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_particles(cls, frame: GridFrame, worlds: Optional[torch.Tensor] = None,
                       latents: Optional[torch.Tensor] = None, log_weights: Optional[torch.Tensor] = None,
                       seed_keys: Optional[Sequence[int]] = None, obs_version: int = 0, step: int = 0,
                       meta: Optional[Dict[str, Any]] = None) -> "BeliefState":
        """K fresh particles (ids 0..K-1, uniform weights unless given). seed_keys default to the ids, i.e.
        world k of the generator (`WorldModel.generate(..., ks=(k,))`)."""
        ref = worlds if worlds is not None else latents
        if ref is None:
            raise ValueError("a belief needs worlds or latents")
        K = ref.shape[0]
        lw = torch.full((K,), -math.log(K), dtype=torch.float64) if log_weights is None else \
            normalize_log_weights(log_weights)[0]
        ids = torch.arange(K, dtype=torch.long)
        keys = ids.clone() if seed_keys is None else torch.as_tensor(list(seed_keys), dtype=torch.long)
        st = cls(frame=frame, log_weights=lw, particle_ids=ids, seed_keys=keys, ancestors=ids.clone(), worlds=worlds,
                 latents=latents, obs_version=obs_version, step=step, next_particle_id=K, meta=dict(meta or {}))
        st.validate()
        return st

    # ------------------------------------------------------------------ queries
    @property
    def K(self) -> int:
        return int(self.log_weights.numel())

    @property
    def device(self) -> torch.device:
        ref = self.worlds if self.worlds is not None else self.latents
        return ref.device if ref is not None else torch.device("cpu")

    def weights(self) -> torch.Tensor:
        return torch.exp(normalize_log_weights(self.log_weights)[0])

    def ess(self) -> float:
        return effective_sample_size(self.log_weights)

    # ------------------------------------------------------------------ transformations (new state, inputs untouched)
    def replace(self, **changes) -> "BeliefState":
        return dataclasses.replace(self, **changes)

    def select(self, index: Sequence[int], new_ids_for_copies: bool = True) -> "BeliefState":
        """Particles at `index` (resampling). Weights become uniform. The first occurrence of a particle keeps its
        id; further copies get new ids (lineage in `ancestors`, same seed key) when new_ids_for_copies."""
        idx = torch.as_tensor(list(index), dtype=torch.long)
        K = idx.numel()
        ids = self.particle_ids[idx].clone()
        nxt = self.next_particle_id
        if new_ids_for_copies:
            seen = set()
            for n, i in enumerate(idx.tolist()):
                if i in seen:
                    ids[n] = nxt
                    nxt += 1
                seen.add(i)
        dev_idx = idx.to(self.device)
        return self.replace(
            log_weights=torch.full((K,), -math.log(K), dtype=torch.float64), particle_ids=ids,
            seed_keys=self.seed_keys[idx].clone(), ancestors=self.particle_ids[idx].clone(),
            worlds=None if self.worlds is None else self.worlds[dev_idx].clone(),
            latents=None if self.latents is None else self.latents[dev_idx].clone(), next_particle_id=nxt)

    def to(self, device) -> "BeliefState":
        return self.replace(worlds=None if self.worlds is None else self.worlds.to(device),
                            latents=None if self.latents is None else self.latents.to(device))

    def clone(self) -> "BeliefState":
        return self.replace(log_weights=self.log_weights.clone(), particle_ids=self.particle_ids.clone(),
                            seed_keys=self.seed_keys.clone(), ancestors=self.ancestors.clone(),
                            worlds=None if self.worlds is None else self.worlds.clone(),
                            latents=None if self.latents is None else self.latents.clone(), meta=dict(self.meta))

    # ------------------------------------------------------------------ checks
    def validate(self, check_finite: bool = True) -> "BeliefState":
        """Raise ValueError on any inconsistency (shapes, weights, ids, non-finite values)."""
        K = self.K
        if K < 1:
            raise ValueError("empty belief")
        for name in ("particle_ids", "seed_keys", "ancestors"):
            v = getattr(self, name)
            if v.shape != (K,) or v.dtype != torch.long:
                raise ValueError(f"{name} must be int64 [{K}], got {tuple(v.shape)} {v.dtype}")
        if self.log_weights.dtype != torch.float64:
            raise ValueError("log_weights must be float64")
        if torch.isnan(self.log_weights).any() or not torch.isfinite(self.log_weights).any():
            raise ValueError("log_weights contain NaN or no finite weight")
        if abs(float(torch.logsumexp(self.log_weights, 0))) > 1e-6:
            raise ValueError("log_weights are not normalised")
        if len(set(self.particle_ids.tolist())) != K:
            raise ValueError("particle ids are not unique")
        if self.worlds is None and self.latents is None:
            raise ValueError("a belief needs worlds or latents")
        if self.worlds is not None and tuple(self.worlds.shape) != (K, *self.frame.shape):
            raise ValueError(f"worlds {tuple(self.worlds.shape)} != {(K, *self.frame.shape)}")
        if self.latents is not None:
            if self.latents.dim() != 5 or self.latents.shape[0] != K or \
                    tuple(self.latents.shape[2:]) != self.frame.latent_shape:
                raise ValueError(f"latents {tuple(self.latents.shape)} != [K, C, {self.frame.latent_shape}]")
        if check_finite:
            for name in ("worlds", "latents"):
                v = getattr(self, name)
                if v is not None and not torch.isfinite(v).all():
                    raise ValueError(f"{name} contain non-finite values")
        return self

    def summary(self) -> Dict[str, Any]:
        return _jsonable({"K": self.K, "ess": self.ess(), "weights": self.weights(), "particle_ids": self.particle_ids,
                          "ancestors": self.ancestors, "seed_keys": self.seed_keys, "obs_version": self.obs_version,
                          "step": self.step, "has_worlds": self.worlds is not None,
                          "has_latents": self.latents is not None})


class UpdateResult(NamedTuple):
    state: BeliefState
    diagnostics: UpdateDiagnostics

