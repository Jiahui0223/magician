"""WorldGenerator adapters for frozen Seen2Scene world models.

`OfficialWorldModelGenerator` wraps active.world_model.WorldModel (the official completion, bit for bit): world n
of a request is WorldModel.generate(scene, observation.tsdf, episode, seed_step, ks=(keys[n],)), i.e. seed
seed_world(scene, episode, seed_step, keys[n]). Any object with the same `generate(scene, volume, episode, step,
ks)` method can be wrapped, e.g. an experiment's outdoor world model (the core does not import it).
"""
from typing import Any

import torch

from seen2scene.belief.interfaces import GenerationOutput, GenerationRequest


class OfficialWorldModelGenerator:
    def __init__(self, world_model: Any, scene_key: str = None, streaming: bool = True):
        """world_model: active.world_model.WorldModel (or the same generate signature). scene_key overrides the
        request's scene for the model's own lookup (e.g. the CSV row name). streaming: one world per call."""
        self.wm, self.scene_key, self.streaming = world_model, scene_key, streaming

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        # latents (request.return_latents) are not returned: WorldModel.generate gives dense worlds only; a caller
        # that keeps latents drops them for refreshed particles (C0 reports latents_dropped)
        scene = self.scene_key or request.scene
        vol = request.observation.tsdf
        if self.streaming:
            worlds = [self.wm.generate(scene, vol, episode=request.episode, step=request.seed_step, ks=(int(k),))[0]
                      for k in request.keys]
            return GenerationOutput(torch.stack(worlds))
        return GenerationOutput(self.wm.generate(scene, vol, episode=request.episode, step=request.seed_step,
                                                 ks=tuple(int(k) for k in request.keys)))
