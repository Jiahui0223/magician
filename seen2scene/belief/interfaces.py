"""Common interfaces of belief updaters and of the frozen models they call.

- `BeliefUpdater.update(state, observation, context) -> UpdateResult(state, diagnostics)`: every updater (C0,
  C1, later ones) takes and returns the same BeliefState protocol, so planners and experiments can swap them.
- `UpdateContext`: what the updater may use besides the new image: the fused observation after (and optionally
  before) integrating it, the frozen generator / decoder, an encoded observation for latent updaters, and the
  identifiers that key the world seeds (scene, episode, step; tools.seeding.seed_world).
- `WorldGenerator` / `LatentDecoder`: structural protocols for the frozen Seen2Scene models; adapters live in
  belief.generators (official world model) or in experiment code. The core never imports experiment code.
"""
import abc
import dataclasses
from typing import List, Optional, Protocol, runtime_checkable

import torch

from seen2scene.belief.observation import ObservationState, RayBundle
from seen2scene.belief.state import BeliefState, UpdateResult
from seen2scene.tools.seeding import GLOBAL_SEED


@dataclasses.dataclass
class GenerationRequest:
    """K worlds conditioned on `observation`; world n uses seed key keys[n] at seed step `seed_step`, i.e. the
    seed seed_world(scene, episode, seed_step, keys[n]) of WorldModel.generate."""

    observation: ObservationState
    keys: List[int]
    scene: str
    episode: int
    seed_step: int
    return_latents: bool = False


@dataclasses.dataclass
class GenerationOutput:
    worlds: torch.Tensor  # [n, D, H, W]
    latents: Optional[torch.Tensor] = None  # [n, C, X, Y, Z]


@runtime_checkable
class WorldGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationOutput: ...


@runtime_checkable
class LatentDecoder(Protocol):
    def decode(self, latents: torch.Tensor, observation: ObservationState) -> torch.Tensor: ...


@dataclasses.dataclass
class UpdateContext:
    observation_state: ObservationState  # fused observation including the new image
    previous_observation_state: Optional[ObservationState] = None  # before the new frames (novel rays, C1 delta)
    generator: Optional[WorldGenerator] = None
    decoder: Optional[LatentDecoder] = None
    latent_observation: Optional[torch.Tensor] = None  # [C, X, Y, Z] encoded observation, 0 where none
    latent_observation_mask: Optional[torch.Tensor] = None  # [X, Y, Z] 1 where latent_observation is defined
    domain: Optional[torch.Tensor] = None  # [D, H, W] bool, voxels the worlds represent (None = the whole grid)
    scene: str = ""
    episode: int = 0
    step: int = 0
    base_seed: int = GLOBAL_SEED


class BeliefUpdater(abc.ABC):
    """Base class: subclasses implement `_update`; `update` validates input and output states."""

    name: str = "belief-updater"

    def update(self, state: BeliefState, observation: Optional[RayBundle],
               context: UpdateContext) -> UpdateResult:
        """observation: the rays of the new frames since the last update (None: no new image, e.g. only a constraint
        check); context.observation_state: the fused observation including them."""
        state.validate()
        if context.observation_state.frame != state.frame:
            raise ValueError("observation state and belief use different grids")
        result = self._update(state, observation, context)
        result.state.validate()
        result.diagnostics.setdefault("updater", self.name)
        return result

    @abc.abstractmethod
    def _update(self, state: BeliefState, observation: Optional[RayBundle],
                context: UpdateContext) -> UpdateResult: ...
