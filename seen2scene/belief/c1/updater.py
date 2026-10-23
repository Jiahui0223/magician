"""C1: learned stochastic latent belief update (configs.belief.C1Latent). Interface only in this phase: no training.

One update moves every particle's latent with the C1 network (c1.network) given the observation condition
(c1.conditioning: current mask, change since the last update, encoded observation), decodes the new latents with the
frozen decoder of the context (worlds), optionally reweights with the depth likelihood (as C0) and checks the
known-observation constraint. Particles keep their ids (an update transports a hypothesis; it is not a new draw).
Per-particle noise is keyed by (scene, episode, step, particle id) through tools.seeding.seed_world and drawn on the
CPU, so it does not depend on the device. The network runs without gradients here; training code calls the network.
"""
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn

from seen2scene.belief.c1.conditioning import LatentConditionBuilder
from seen2scene.belief.constraints import ObservationConstraintModel
from seen2scene.belief.interfaces import BeliefUpdater, UpdateContext
from seen2scene.belief.likelihood import DepthLikelihoodModel
from seen2scene.belief.observation import RayBundle
from seen2scene.belief.state import BeliefState, UpdateDiagnostics, UpdateResult, normalize_log_weights
from seen2scene.configs.belief import C1Latent
from seen2scene.tools.common_utils import instantiate_from_config
from seen2scene.tools.seeding import seed_world


class C1LatentUpdater(BeliefUpdater):
    name = "C1-latent"

    def __init__(self, cfg: Optional[C1Latent] = None, network: Optional[nn.Module] = None):
        self.cfg = cfg or C1Latent()
        self.net = network if network is not None else instantiate_from_config(self.cfg.network)
        self.conditioner = LatentConditionBuilder(self.cfg.network.latent_channels)
        if self.conditioner.channels != self.net.cond_channels:
            raise ValueError(f"condition has {self.conditioner.channels} channels, network expects "
                             f"{self.net.cond_channels}")
        self.likelihood = DepthLikelihoodModel(self.cfg.likelihood)
        self.constraint = ObservationConstraintModel(self.cfg.constraint)

    def particle_noise(self, state: BeliefState, ctx: UpdateContext) -> Tuple[torch.Tensor, torch.Tensor]:
        """Noise field [K, Cn, X, Y, Z] and code [K, G] of every particle (CPU generator, then the device)."""
        net, s = self.net, self.cfg.noise_scale
        X, Y, Z = state.frame.latent_shape
        fields, codes = [], []
        for pid in state.particle_ids.tolist():
            g = torch.Generator()
            g.manual_seed(seed_world(ctx.scene, ctx.episode, ctx.step, int(pid), base=self.cfg.seed))
            fields.append(torch.randn(net.noise_channels, X, Y, Z, generator=g) * s)
            codes.append(torch.randn(net.global_noise_dim, generator=g) * s)
        return torch.stack(fields).to(state.device), torch.stack(codes).to(state.device)

    def _update(self, state: BeliefState, observation: Optional[RayBundle],
                context: UpdateContext) -> UpdateResult:
        t0 = time.time()
        if state.latents is None:
            raise ValueError("C1 needs the latents of the particles")
        K, diag = state.K, UpdateDiagnostics()
        cond = self.conditioner.build(context.observation_state, context.previous_observation_state,
                                      context.latent_observation, context.latent_observation_mask)
        cond = cond.to(state.device)[None].expand(K, -1, -1, -1, -1)
        noise, code = self.particle_noise(state, context)
        step = torch.full((K,), float(context.observation_state.version), device=state.device)
        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                out = self.net(state.latents, cond, noise, code, step)
        finally:
            self.net.train(was_training)
        z_new = out["z_new"]
        bad = ~torch.isfinite(z_new.reshape(K, -1)).all(1)
        if bad.any():  # numerical failure of a particle: keep its old latent, report it
            z_new = torch.where(bad.view(K, 1, 1, 1, 1), state.latents, z_new)
        change = (z_new - state.latents).reshape(K, -1)
        diag.set(nonfinite_particles=torch.nonzero(bad)[:, 0], latent_change_rms=change.pow(2).mean(1).sqrt(),
                 gate_mean=out["gate"].reshape(K, -1).float().mean(1),
                 new_observation_cells=float((cond[0, 3:6].abs().sum(0) > 0).float().mean()))

        worlds = None
        if context.decoder is not None:
            worlds = context.decoder.decode(z_new, context.observation_state)
            if tuple(worlds.shape) != (K, *state.frame.shape) or not torch.isfinite(worlds).all():
                raise ValueError("decoder returned invalid worlds")
        elif state.worlds is not None:
            diag.set(worlds_dropped="no decoder in the context: the old worlds would not match the new latents")

        log_w = state.log_weights
        if self.cfg.reweight and observation is not None and worlds is not None:
            prev = context.previous_observation_state
            lik = self.likelihood.score(worlds, state.frame, observation, None if prev is None else prev.structure,
                                        context.domain)
            log_w, all_invalid = normalize_log_weights(log_w + lik.loglik.cpu())
            diag.set(likelihood=lik.to_dict(), all_invalid=all_invalid)
        if worlds is not None:
            worlds, cdiag = self.constraint.apply(worlds, context.observation_state)
            diag.set(constraint=cdiag)
        new = state.replace(latents=z_new, worlds=worlds, log_weights=log_w, ancestors=state.particle_ids.clone(),
                            obs_version=context.observation_state.version, step=context.step)
        diag.set(ess_prior=state.ess(), ess_final=new.ess(), weights=new.weights(), seconds=time.time() - t0)
        return UpdateResult(new, diag)
