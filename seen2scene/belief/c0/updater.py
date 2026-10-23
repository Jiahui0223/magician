"""C0: particle belief update without learning (configs.belief.C0Particle; EXP4 architecture §3.3).

One update with the rays of the new frames since the last update:
1. reweight: log w_k += tempered depth log-likelihood of world k (belief.likelihood; only rays through previously
   UNKNOWN space with novel_only), normalised in log space;
2. collapse: every likelihood -inf / NaN, or no world explains `collapse_inlier_rate` of the scored rays -> the
   whole set contradicts the new frames -> regenerate all K worlds from the frozen generator conditioned on the
   current observation (on_collapse = "refresh_all", seeds `collapse_seeds`), uniform weights;
3. otherwise, if ESS < ess_threshold * K: resample (systematic by default) and regenerate every duplicate beyond the
   first copy of a particle (refresh = "duplicates", seeds `refresh_seeds`): resample-move with a move kernel that is
   valid for the current conditional (not for the Bayesian target; stated in the EXP4 plan);
4. a refresh that fails (generator error, wrong shape, non-finite values) keeps the particles it would have replaced
   and is reported (or raises with raise_on_refresh_failure);
5. known-observation constraint: by default only checked; the generated worlds are kept as they are.
Randomness (resampling offset) is keyed by (scene, episode, step) through tools.seeding.seed_world. Refreshed
particles get new ids; their seed key is the new id ("new", seed step = update step) or the seed key of their lineage
at seed step 0 ("persist").
"""
import time
from typing import Dict, List, Optional, Tuple

import torch

from seen2scene.belief.constraints import ObservationConstraintModel
from seen2scene.belief.interfaces import BeliefUpdater, GenerationRequest, UpdateContext
from seen2scene.belief.likelihood import DepthLikelihoodModel
from seen2scene.belief.observation import RayBundle
from seen2scene.belief.resampling import SCHEMES
from seen2scene.belief.state import BeliefState, UpdateDiagnostics, UpdateResult, normalize_log_weights
from seen2scene.configs.belief import C0Particle
from seen2scene.tools.seeding import seed_world

RNG_RESAMPLE = 1


class C0ParticleUpdater(BeliefUpdater):
    name = "C0-particle"

    def __init__(self, cfg: Optional[C0Particle] = None):
        self.cfg = cfg or C0Particle()
        self.likelihood = DepthLikelihoodModel(self.cfg.likelihood)
        self.constraint = ObservationConstraintModel(self.cfg.constraint)

    def _rng(self, ctx: UpdateContext, purpose: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(seed_world(ctx.scene, ctx.episode, ctx.step, purpose, base=self.cfg.seed))
        return g

    def _update(self, state: BeliefState, observation: Optional[RayBundle],
                context: UpdateContext) -> UpdateResult:
        t0 = time.time()
        cfg, K = self.cfg, state.K
        if state.worlds is None:
            raise ValueError("C0 needs the worlds of the particles")
        diag = UpdateDiagnostics()
        new = state.replace(ancestors=state.particle_ids.clone())

        # 1. reweight
        all_invalid, best_inlier, scored = False, 1.0, 0
        if observation is not None:
            prev = context.previous_observation_state
            lik = self.likelihood.score(state.worlds, state.frame, observation,
                                        None if prev is None else prev.structure, context.domain)
            scored = lik.counts["scored"]
            log_w, all_invalid = normalize_log_weights(state.log_weights + lik.loglik.cpu())
            best_inlier = float(lik.inlier_rate.max()) if scored else 1.0
            new = new.replace(log_weights=log_w)
            diag.set(likelihood=lik.to_dict())
        t_lik = time.time() - t0
        ess_prior, ess_post = state.ess(), new.ess()
        collapse = all_invalid or (scored > 0 and best_inlier < cfg.collapse_inlier_rate)
        diag.set(ess_prior=ess_prior, ess_posterior=ess_post, all_invalid=all_invalid, best_inlier_rate=best_inlier,
                 collapse=collapse)

        # 2./3. collapse refresh, or resampling + duplicate refresh
        action, refresh = "reweight", {"positions": [], "failed": False}
        if collapse and cfg.on_collapse == "refresh_all":
            action = "refresh_all"
            new = new.select(range(K))  # uniform weights; lineage kept for the fallback
            new, refresh = self._refresh(new, list(range(K)), context, cfg.collapse_seeds)
        elif ess_post < cfg.ess_threshold * K:
            action = "resample"
            idx = SCHEMES[cfg.resampling](torch.exp(new.log_weights), self._rng(context, RNG_RESAMPLE))
            new = new.select(idx.tolist())
            diag.set(resample_index=idx)
            if cfg.refresh == "duplicates":
                seen, dup = set(), []
                for n, i in enumerate(idx.tolist()):
                    if i in seen:
                        dup.append(n)
                    seen.add(i)
                new, refresh = self._refresh(new, dup, context, cfg.refresh_seeds)
        diag.set(action=action, refresh=refresh, n_refreshed=0 if refresh.get("failed") else len(refresh["positions"]))

        # 5. known-observation constraint (default: check only)
        worlds, cdiag = self.constraint.apply(new.worlds, context.observation_state)
        new = new.replace(worlds=worlds, obs_version=context.observation_state.version, step=context.step)
        diag.set(constraint=cdiag, ess_final=new.ess(), weights=new.weights(), particle_ids=new.particle_ids,
                 ancestors=new.ancestors, seed_keys=new.seed_keys,
                 seconds={"likelihood": t_lik, "refresh": refresh.get("seconds", 0.0), "total": time.time() - t0})
        return UpdateResult(new, diag)

    def _refresh(self, state: BeliefState, positions: List[int], ctx: UpdateContext,
                 seeds: str) -> Tuple[BeliefState, Dict]:
        """Regenerate the particles at `positions` from the frozen generator (current observation)."""
        cfg = self.cfg
        if not positions:
            return state, {"positions": [], "failed": False}
        tr = time.time()
        new_ids = list(range(state.next_particle_id, state.next_particle_id + len(positions)))
        if seeds == "new":
            keys, seed_step = new_ids, ctx.step
        else:
            keys, seed_step = [int(state.seed_keys[p]) for p in positions], 0
        info = {"positions": positions, "new_ids": new_ids, "keys": keys, "seed_step": seed_step, "seeds": seeds,
                "failed": False}
        try:
            if ctx.generator is None:
                raise RuntimeError("no generator in the update context")
            out = ctx.generator.generate(GenerationRequest(ctx.observation_state, keys, ctx.scene, ctx.episode,
                                                           seed_step, return_latents=state.latents is not None))
            w = out.worlds
            if tuple(w.shape) != (len(positions), *state.frame.shape):
                raise ValueError(f"generator returned {tuple(w.shape)}, expected "
                                 f"{(len(positions), *state.frame.shape)}")
            if not torch.isfinite(w).all():
                raise ValueError("generator returned non-finite worlds")
        except Exception as e:  # noqa: BLE001 -- any generator failure keeps the old particles
            if cfg.raise_on_refresh_failure:
                raise
            info.update(failed=True, error=repr(e), seconds=time.time() - tr)
            return state, info
        pos = torch.as_tensor(positions, dtype=torch.long)
        worlds = state.worlds.clone()
        worlds[pos.to(worlds.device)] = w.to(worlds.device, worlds.dtype)
        ids, skeys, anc = state.particle_ids.clone(), state.seed_keys.clone(), state.ancestors.clone()
        ids[pos] = torch.as_tensor(new_ids, dtype=torch.long)
        skeys[pos] = torch.as_tensor(keys, dtype=torch.long)
        anc[pos] = -1  # fresh draw from the generator, no ancestor
        latents = state.latents
        if latents is not None:
            if out.latents is not None and tuple(out.latents.shape) == (len(positions), *latents.shape[1:]) \
                    and torch.isfinite(out.latents).all():
                latents = latents.clone()
                latents[pos.to(latents.device)] = out.latents.to(latents.device, latents.dtype)
            else:  # the generator gave no (valid) latents: the refreshed particles would carry stale ones
                latents = None
                info["latents_dropped"] = True
        info["seconds"] = time.time() - tr
        return state.replace(worlds=worlds, particle_ids=ids, seed_keys=skeys, ancestors=anc, latents=latents,
                             next_particle_id=state.next_particle_id + len(positions)), info
