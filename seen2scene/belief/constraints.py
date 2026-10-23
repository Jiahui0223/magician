"""Known-observation constraint of updated worlds (configs.belief.ObservationConstraint).

Violations of a world against the fused observation (structure EMPTY / UNKNOWN / BAND):
- free_violation: fraction of observed EMPTY voxels where the world is not free (value < 0.999 truncation);
- band_unexplained: fraction of observed BAND voxels with no world BAND voxel within `band_dilation` voxels.
Projection ("paste" / "carve") writes the observation into the world: observed EMPTY -> +truncation (free),
and for "paste" observed BAND -> the observed TSDF value. UNKNOWN voxels are never touched.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from seen2scene.belief.observation import BAND, BAND_RATIO, EMPTY, ObservationState
from seen2scene.configs.belief import ObservationConstraint


class ObservationConstraintModel:
    def __init__(self, cfg: Optional[ObservationConstraint] = None):
        self.cfg = cfg or ObservationConstraint()

    @torch.no_grad()
    def violations(self, worlds: torch.Tensor, obs: ObservationState) -> Dict[str, list]:
        tr = obs.frame.truncation * BAND_RATIO
        st = obs.structure.to(worlds.device)
        empty, band = st == EMPTY, st == BAND
        n_empty, n_band = int(empty.sum()), int(band.sum())
        free_v, band_u = [], []
        r = self.cfg.band_dilation
        for w in worlds:  # one world at a time (memory)
            free_v.append(float((empty & (w < tr)).sum()) / max(n_empty, 1))
            wb = (w.abs() < tr).float()[None, None]
            if r > 0:
                wb = F.max_pool3d(wb, 2 * r + 1, 1, r)
            band_u.append(float((band & (wb[0, 0] <= 0)).sum()) / max(n_band, 1))
        return {"free_violation": free_v, "band_unexplained": band_u, "n_empty": n_empty, "n_band": n_band}

    @torch.no_grad()
    def apply(self, worlds: torch.Tensor, obs: ObservationState) -> Tuple[torch.Tensor, Dict]:
        """(projected worlds (a new tensor unless mode is check / none), diagnostics before / after)."""
        mode = self.cfg.mode
        if mode == "none":
            return worlds, {"mode": mode}
        before = self.violations(worlds, obs)
        if mode == "check":
            return worlds, {"mode": mode, "before": before}
        st = obs.structure.to(worlds.device)
        out = worlds.clone()
        out[:, st == EMPTY] = obs.frame.truncation
        if mode == "paste":
            band = st == BAND
            out[:, band] = obs.tsdf.to(worlds.device, worlds.dtype)[band]
        return out, {"mode": mode, "before": before, "after": self.violations(out, obs)}
