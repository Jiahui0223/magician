"""Seen2Scene belief update: a weighted set of K world hypotheses of one grid, updated with new depth observations.

    state.py         GridFrame, BeliefState (worlds / latents, log weights, ids, lineage, seed keys), diagnostics
    observation.py   RayBundle (new depth rays), DepthObservation (pinhole image), ObservationState (fused TSDF)
    raycast.py       first-surface ray casting of dense TSDF grids (PatchRaycaster rule, any grid, any rays)
    likelihood.py    depth likelihood of worlds (surface / pass / blocked rays, novel rays, tempering)
    constraints.py   known-observation constraint (check by default; paste / carve projections)
    resampling.py    systematic / residual / multinomial resampling
    interfaces.py    BeliefUpdater base class, UpdateContext, WorldGenerator / LatentDecoder protocols
    generators.py    WorldGenerator adapter of the official world model (active.world_model.WorldModel)
    schedule.py      PeriodicUpdater: update period R
    c0/              C0ParticleUpdater: particle update without learning
    c1/              C1LatentUpdater + C1LatentUpdateNet: learned stochastic latent update (interface; no training)
The configurations are in seen2scene/configs/belief.py. Nothing here imports experiment code.
"""
from seen2scene.belief.c0 import C0ParticleUpdater
from seen2scene.belief.interfaces import (BeliefUpdater, GenerationOutput, GenerationRequest, LatentDecoder,
                                          UpdateContext, WorldGenerator)
from seen2scene.belief.observation import DepthObservation, ObservationState, RayBundle
from seen2scene.belief.schedule import PeriodicUpdater
from seen2scene.belief.state import BeliefState, GridFrame, UpdateDiagnostics, UpdateResult

__all__ = ["BeliefState", "GridFrame", "UpdateDiagnostics", "UpdateResult", "DepthObservation", "ObservationState",
           "RayBundle", "BeliefUpdater", "UpdateContext", "WorldGenerator", "LatentDecoder", "GenerationRequest",
           "GenerationOutput", "PeriodicUpdater", "C0ParticleUpdater"]
