import math

import pytest
import torch

from conftest import MockGenerator, empty_world, observed_state_from, sphere_world, unknown_state
from seen2scene.belief.c0 import C0ParticleUpdater
from seen2scene.belief.constraints import ObservationConstraintModel
from seen2scene.belief.generators import OfficialWorldModelGenerator
from seen2scene.belief.interfaces import GenerationRequest, UpdateContext
from seen2scene.belief.likelihood import DepthLikelihoodModel
from seen2scene.belief.observation import ObservationState, RayBundle
from seen2scene.belief.resampling import SCHEMES
from seen2scene.belief.schedule import PeriodicUpdater
from seen2scene.belief.state import BeliefState
from seen2scene.configs.belief import C0Particle, DepthLikelihood, ObservationConstraint


def worlds_of(scene):
    fr, true = scene["frame"], scene["true"]
    shifted = sphere_world(fr, centre=(0.8, 0.9, 0.8))  # 2 voxels further from the camera
    blob = torch.minimum(true, sphere_world(fr, centre=(0.8, 0.25, 0.8), radius=0.12))  # extra object in free space
    return torch.stack([true, shifted, empty_world(fr), blob])


# ---------------------------------------------------------------- likelihood
def test_likelihood_orders_worlds(scene):
    lik = DepthLikelihoodModel().score(worlds_of(scene), scene["frame"], scene["rays"], scene["prior"].structure)
    ll = lik.loglik
    assert lik.counts["scored"] > 100
    assert ll[0] > ll[1] > ll[2] and ll[0] > ll[3]
    assert float(lik.inlier_rate[0]) > 0.95 and float(lik.free_violation_rate[3]) > 0.0
    assert float(lik.missing_rate[2]) == 1.0


def test_novel_only_ignores_observed_space(scene):
    lik = DepthLikelihoodModel().score(worlds_of(scene), scene["frame"], scene["rays"], scene["post"].structure)
    assert lik.counts["scored"] == 0 and torch.equal(lik.loglik, torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError):
        DepthLikelihoodModel().score(worlds_of(scene), scene["frame"], scene["rays"], None)


def test_blocked_rays_excluded(scene):
    r = scene["rays"]
    blocked = RayBundle(r.origins, r.dirs, torch.full_like(r.t_obs, 0.3), r.frame_ids)  # ends before the grid
    lik = DepthLikelihoodModel().score(worlds_of(scene), scene["frame"], blocked, scene["prior"].structure)
    assert lik.counts["scored"] == 0 and lik.counts["blocked"] > 0


def test_tempering_per_frame(scene):
    m = DepthLikelihoodModel(DepthLikelihood(effective_rays_per_frame=3.0))
    one = m.score(worlds_of(scene), scene["frame"], scene["rays"], scene["prior"].structure).loglik
    two = m.score(worlds_of(scene), scene["frame"], RayBundle.cat([scene["rays"], scene["rays"]]),
                  scene["prior"].structure).loglik
    assert torch.allclose(two, 2 * one)
    half = RayBundle(*(x[::2] for x in (scene["rays"].origins, scene["rays"].dirs, scene["rays"].t_obs,
                                        scene["rays"].frame_ids)))
    sub = m.score(worlds_of(scene), scene["frame"], half, scene["prior"].structure).loglik
    assert torch.allclose(sub, one, rtol=0.25)  # a frame is one observation, independent of the ray count


def test_invalid_world_is_minus_inf(scene):
    w = worlds_of(scene)
    w[1, 3, 3, 3] = math.nan
    lik = DepthLikelihoodModel().score(w, scene["frame"], scene["rays"], scene["prior"].structure)
    assert bool(lik.invalid[1]) and math.isinf(float(lik.loglik[1])) and torch.isfinite(lik.loglik[[0, 2, 3]]).all()


# ---------------------------------------------------------------- resampling
def test_systematic_resampling_counts_and_determinism():
    w = torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=torch.float64)
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    a, b = SCHEMES["systematic"](w, g1), SCHEMES["systematic"](w, g2)
    assert torch.equal(a, b) and (a == 0).sum() == 2 and (a == 1).sum() == 1
    for name in SCHEMES:
        idx = SCHEMES[name](w, torch.Generator().manual_seed(0))
        assert idx.numel() == 4 and torch.equal(idx, idx.sort().values)


# ---------------------------------------------------------------- C0
def ctx_for(scene, gen=None, step=5):
    post = ObservationState(scene["frame"], scene["post"].tsdf, 1)
    return UpdateContext(observation_state=post, previous_observation_state=scene["prior"], generator=gen,
                         scene="synthetic", episode=0, step=step)


def test_c0_resample_and_refresh_duplicates(wall):
    fr = wall["frame"]
    st = BeliefState.from_particles(fr, worlds=torch.stack([wall["true"]] + [empty_world(fr)] * 3))
    gen = MockGenerator(wall["true"])
    res = C0ParticleUpdater().update(st, wall["rays"], ctx_for(wall, gen))
    d, new = res.diagnostics, res.state
    assert d["action"] == "resample" and d["ess_posterior"] < 2
    assert new.ancestors.tolist().count(0) == 1 and (new.ancestors == -1).sum() == 3  # 1 survivor + 3 refreshed
    req = gen.requests[0]
    # resampled copies get provisional ids 4-6 (kept if the refresh fails); refreshed particles get fresh ids
    assert req.keys == [7, 8, 9] and req.seed_step == 5 and req.scene == "synthetic"
    assert new.particle_ids.tolist() == [0, 7, 8, 9] and new.next_particle_id == 10 and abs(new.ess() - 4) < 1e-9
    assert torch.equal(new.worlds[1], wall["true"]) and torch.equal(st.worlds[1], empty_world(fr))
    assert new.obs_version == 1 and new.step == 5


def test_c0_persist_seeds(wall):
    fr = wall["frame"]
    st = BeliefState.from_particles(fr, worlds=torch.stack([wall["true"]] + [empty_world(fr)] * 3),
                                    seed_keys=[10, 11, 12, 13])
    gen = MockGenerator(wall["true"])
    C0ParticleUpdater(C0Particle(refresh_seeds="persist")).update(st, wall["rays"], ctx_for(wall, gen))
    assert gen.requests[0].keys == [10, 10, 10] and gen.requests[0].seed_step == 0


def test_c0_collapse_refreshes_all(wall):
    fr = wall["frame"]
    st = BeliefState.from_particles(fr, worlds=torch.stack([empty_world(fr)] * 4))
    gen = MockGenerator(wall["true"])
    res = C0ParticleUpdater().update(st, wall["rays"], ctx_for(wall, gen))
    assert res.diagnostics["collapse"] and res.diagnostics["action"] == "refresh_all"
    assert gen.requests[0].keys == [0, 1, 2, 3] and gen.requests[0].seed_step == 0  # FR-persist recovery
    assert (res.state.ancestors == -1).all() and torch.equal(res.state.worlds[2], wall["true"])


@pytest.mark.parametrize("gen_kind", ["fail", "nan", "none"])
def test_c0_refresh_failure_keeps_particles(wall, gen_kind):
    fr = wall["frame"]
    st = BeliefState.from_particles(fr, worlds=torch.stack([wall["true"]] + [empty_world(fr)] * 3))
    gen = None if gen_kind == "none" else MockGenerator(wall["true"], fail=gen_kind == "fail", nan=gen_kind == "nan")
    res = C0ParticleUpdater().update(st, wall["rays"], ctx_for(wall, gen))
    assert res.diagnostics["refresh"]["failed"] and res.diagnostics["n_refreshed"] == 0
    assert torch.equal(res.state.worlds, wall["true"][None].expand(4, -1, -1, -1))  # resampled copies kept
    res.state.validate()
    with pytest.raises(Exception):
        C0ParticleUpdater(C0Particle(raise_on_refresh_failure=True)).update(st, wall["rays"], ctx_for(wall, gen))


def test_c0_keeps_generated_worlds_by_default(scene):
    st = BeliefState.from_particles(scene["frame"], worlds=worlds_of(scene))
    res = C0ParticleUpdater(C0Particle(ess_threshold=0.0)).update(st, scene["rays"], ctx_for(scene))
    assert res.diagnostics["action"] == "reweight" and torch.equal(res.state.worlds, st.worlds)
    c = res.diagnostics["constraint"]
    assert c["mode"] == "check" and c["before"]["free_violation"][3] > 0
    w = res.state.weights()
    assert w[0] > w[1] > w[2]


def test_constraint_paste_projects(scene):
    w = worlds_of(scene)
    out, d = ObservationConstraintModel(ObservationConstraint(mode="paste")).apply(w, scene["post"])
    assert max(d["after"]["free_violation"]) == 0.0 and not torch.equal(out, w)


def test_c0_deterministic(scene):
    st = BeliefState.from_particles(scene["frame"], worlds=worlds_of(scene))
    a = C0ParticleUpdater().update(st, scene["rays"], ctx_for(scene, MockGenerator(scene["true"])))
    b = C0ParticleUpdater().update(st, scene["rays"], ctx_for(scene, MockGenerator(scene["true"])))
    assert torch.equal(a.state.worlds, b.state.worlds) and torch.equal(a.state.log_weights, b.state.log_weights)
    da, db = dict(a.diagnostics), dict(b.diagnostics)
    da.pop("seconds"), db.pop("seconds")
    assert da == db


def test_periodic_updater(scene):
    fr = scene["frame"]
    st = BeliefState.from_particles(fr, worlds=worlds_of(scene))
    pu = PeriodicUpdater(C0ParticleUpdater(C0Particle(ess_threshold=0.0)), period=3)
    pu.reset(unknown_state(fr))
    acts = []
    for t in range(1, 7):
        obs_t = ObservationState(fr, scene["post"].tsdf, t)
        res = pu.step(st, scene["rays"], obs_t, t, scene="synthetic")
        acts.append((res.diagnostics["action"], res.diagnostics.get("frames")))
        st = res.state
    assert [a for a, _ in acts] == ["wait", "wait", "reweight", "wait", "wait", "reweight"]
    assert acts[2][1] == 3 and st.obs_version == 6


def test_official_generator_adapter_maps_seeds(frame):
    calls = []

    class FakeWM:
        def generate(self, scene, volume, episode, step, ks):
            calls.append((scene, episode, step, ks))
            return torch.zeros(len(ks), *frame.shape)
    adapter = OfficialWorldModelGenerator(FakeWM(), scene_key="row-name")
    out = adapter.generate(GenerationRequest(unknown_state(frame), [7, 9], "s", 2, 4))
    assert out.worlds.shape == (2, *frame.shape)
    assert calls == [("row-name", 2, 4, (7,)), ("row-name", 2, 4, (9,))]


def test_belief_package_does_not_import_experiments():
    import subprocess
    import sys
    code = ("import sys; import seen2scene.belief, seen2scene.belief.c1; "
            "bad=[m for m in sys.modules if m.split('.')[0] in ('m_common','m_world','t1_eval','g1_common','exp3b_paths')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def test_domain_masks_worlds(scene):
    fr = scene["frame"]
    blob = worlds_of(scene)[3:4]
    dom = torch.ones(fr.shape, dtype=torch.bool)
    dom[:, :12, :] = False  # the extra object (y ~ 0.25 m) lies outside the represented domain
    m = DepthLikelihoodModel()
    full = m.score(blob, fr, scene["rays"], scene["prior"].structure)
    masked = m.score(blob, fr, scene["rays"], scene["prior"].structure, dom)
    assert float(masked.free_violation_rate[0]) < float(full.free_violation_rate[0])
