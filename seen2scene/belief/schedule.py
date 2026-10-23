"""Update period R (EXP4 architecture §3.1): the observation evidence is fused every step, the belief only at steps
t in {R, 2R, ...}; an update sees the rays of all frames since the last update and the observation state before
them (previous_observation_state, for novel rays and C1 deltas).

`PeriodicUpdater(updater, R)` wraps any BeliefUpdater: `step(state, rays_t, observation_state_t, t, **context)` buffers
the frame rays of step t and returns the updated belief at the update steps, the unchanged belief otherwise.
"""
import dataclasses
from typing import List, Optional

from seen2scene.belief.interfaces import BeliefUpdater, UpdateContext
from seen2scene.belief.observation import ObservationState, RayBundle
from seen2scene.belief.state import BeliefState, UpdateDiagnostics, UpdateResult


@dataclasses.dataclass
class PeriodicUpdater:
    updater: BeliefUpdater
    period: int = 1
    _buffer: List[RayBundle] = dataclasses.field(default_factory=list)
    _since: Optional[ObservationState] = None  # observation state at the last update

    def reset(self, observation_state: ObservationState):
        """Start of a sequence: the belief was generated from `observation_state`."""
        self._buffer, self._since = [], observation_state

    def step(self, state: BeliefState, rays: Optional[RayBundle], observation_state: ObservationState, t: int,
             **context) -> UpdateResult:
        if self._since is None:
            raise RuntimeError("call reset() with the observation state the belief was generated from")
        if rays is not None and rays.n_rays:
            self._buffer.append(rays)
        if t % self.period != 0:
            return UpdateResult(state, UpdateDiagnostics(action="wait", step=t, buffered_frames=len(self._buffer)))
        obs = RayBundle.cat(self._buffer) if self._buffer else None
        ctx = UpdateContext(observation_state=observation_state, previous_observation_state=self._since, step=t,
                            **context)
        result = self.updater.update(state, obs, ctx)
        result.diagnostics.set(step=t, frames=len(self._buffer), period=self.period)
        self._buffer, self._since = [], observation_state
        return result
