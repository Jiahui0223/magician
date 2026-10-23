"""Observation condition of the C1 latent update on the generator's latent grid ([cond_channels, X, Y, Z]).

Channels (cond_channels = 6 + C + 1 for latent channels C; 15 for the 8-channel Seen2Scene VAE):
  0-2      fractions of EMPTY / UNKNOWN / BAND voxels of the current fused observation in every factor^3 cell
           (the three-state mask of Task 1, belief.observation.structure_fractions);
  3-5      change of these fractions caused by the new observation(s): current - previous (previous = all UNKNOWN
           when not given), i.e. where the new image added free space or surface;
  6..6+C-1 encoded observation latent (e.g. the VAE encoding of the observed TSDF filled onto the dense grid as
           in FM.sample, 0 elsewhere), zeros when not given;
  6+C      mask of the encoded observation latent.
"""
from typing import Optional

import torch

from seen2scene.belief.observation import ObservationState, structure_fractions


class LatentConditionBuilder:
    def __init__(self, latent_channels: int = 8):
        self.latent_channels = latent_channels

    @property
    def channels(self) -> int:
        return 6 + self.latent_channels + 1

    @torch.no_grad()
    def build(self, current: ObservationState, previous: Optional[ObservationState] = None,
              latent_observation: Optional[torch.Tensor] = None,
              latent_observation_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        frame = current.frame
        dev = current.tsdf.device
        cur = structure_fractions(current.structure, frame.latent_factor)
        if previous is None:
            prev = torch.zeros_like(cur)
            prev[1] = 1.0  # all UNKNOWN
        else:
            if previous.frame != frame:
                raise ValueError("previous observation uses another grid")
            prev = structure_fractions(previous.structure.to(dev), frame.latent_factor)
        X, Y, Z = frame.latent_shape
        lat = torch.zeros(self.latent_channels, X, Y, Z, device=dev)
        mask = torch.zeros(1, X, Y, Z, device=dev)
        if latent_observation is not None:
            if tuple(latent_observation.shape) != (self.latent_channels, X, Y, Z):
                raise ValueError(f"latent observation {tuple(latent_observation.shape)} != "
                                 f"{(self.latent_channels, X, Y, Z)}")
            lat = latent_observation.to(dev, torch.float32)
            mask = (torch.ones(1, X, Y, Z, device=dev) if latent_observation_mask is None
                    else latent_observation_mask.to(dev, torch.float32).reshape(1, X, Y, Z))
            lat = lat * mask
        return torch.cat([cur, cur - prev, lat, mask], 0)
