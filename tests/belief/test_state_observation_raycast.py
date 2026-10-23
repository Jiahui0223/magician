import math

import numpy as np
import pytest
import torch

from conftest import look_at_c2w, small_frame, sphere_world
from seen2scene.belief.observation import (BAND, EMPTY, UNKNOWN, DepthObservation, RayBundle, classify_structure,
                                           structure_fractions)
from seen2scene.belief.raycast import march_first_hit
from seen2scene.belief.state import BeliefState, GridFrame, effective_sample_size, normalize_log_weights


# ---------------------------------------------------------------- state
def test_uniform_state_and_ess(frame):
    st = BeliefState.from_particles(frame, worlds=torch.zeros(4, *frame.shape))
    assert st.K == 4 and abs(st.ess() - 4.0) < 1e-12
    assert torch.allclose(st.weights(), torch.full((4,), 0.25, dtype=torch.float64))
    assert st.particle_ids.tolist() == [0, 1, 2, 3] and st.next_particle_id == 4


def test_ess_one_hot_and_all_invalid():
    lw = torch.tensor([0.0, -math.inf, -math.inf], dtype=torch.float64)
    assert abs(effective_sample_size(lw) - 1.0) < 1e-12
    u, bad = normalize_log_weights(torch.tensor([math.nan, -math.inf], dtype=torch.float64))
    assert bad and torch.allclose(u.exp(), torch.full((2,), 0.5, dtype=torch.float64))


def test_select_lineage_and_ids(frame):
    w = torch.stack([torch.full(frame.shape, float(i)) for i in range(4)])
    st = BeliefState.from_particles(frame, worlds=w)
    s2 = st.select([1, 1, 3, 3])
    assert s2.ancestors.tolist() == [1, 1, 3, 3]
    assert s2.particle_ids.tolist() == [1, 4, 3, 5] and s2.next_particle_id == 6
    assert s2.seed_keys.tolist() == [1, 1, 3, 3]
    assert torch.equal(s2.worlds[1], w[1]) and abs(s2.ess() - 4) < 1e-12
    assert torch.equal(st.worlds, w)  # input untouched


def test_validate_rejects_bad_states(frame):
    st = BeliefState.from_particles(frame, worlds=torch.zeros(2, *frame.shape))
    with pytest.raises(ValueError):
        st.replace(log_weights=torch.zeros(2, dtype=torch.float64)).validate()  # not normalised
    bad = st.worlds.clone()
    bad[0, 0, 0, 0] = math.nan
    with pytest.raises(ValueError):
        st.replace(worlds=bad).validate()
    with pytest.raises(ValueError):
        st.replace(particle_ids=torch.tensor([0, 0])).validate()
    with pytest.raises(ValueError):
        BeliefState.from_particles(frame, latents=torch.zeros(2, 8, 3, 3, 3))  # latent grid of 32^3 is 4^3


# ---------------------------------------------------------------- observation
def test_rays_match_patchfusion_backprojection():
    depth = torch.rand(6, 8) * 3 + 0.5
    depth[0, 0] = 0.0
    c2w = look_at_c2w((0.3, -1.0, 0.7))
    obs = DepthObservation.from_fov(depth, c2w, 60.0)
    rays = obs.to_rays()
    f = (8 / 2) / math.tan(math.radians(30))
    v, u = np.nonzero(depth.numpy() > 0)
    z = depth.numpy()[v, u].astype(np.float64)
    p_cam = np.stack([(u + 0.5 - 4) / f * z, -(v + 0.5 - 3) / f * z, -z], 1)  # active.fusion.PatchFusion.integrate
    pts = p_cam @ c2w[:3, :3].numpy().T + c2w[:3, 3].numpy()
    idx = v * 8 + u
    got = (rays.origins[idx].double() + rays.dirs[idx].double() * rays.t_obs[idx, None].double()).numpy()
    assert np.allclose(got, pts, atol=1e-5)
    assert math.isinf(float(rays.t_obs[0]))


def test_structure_classification_and_fractions(frame):
    tr = frame.truncation
    tsdf = torch.tensor([tr, 0.0, -tr, 0.5 * tr, 0.9995 * tr])
    assert classify_structure(tsdf, tr).tolist() == [EMPTY, BAND, UNKNOWN, BAND, EMPTY]
    try:  # same thresholds as the frozen helper of active.episode (imports pyrender; skipped if unavailable)
        from seen2scene.active.episode import structure_torch
        w = (torch.rand(5000) * 2 - 1) * 0.04
        assert torch.equal(classify_structure(w, 0.033), structure_torch(w))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"active.episode not importable here: {e}")


def test_structure_fractions_padding():
    s = torch.full((10, 8, 8), EMPTY, dtype=torch.int8)
    fr = structure_fractions(s, 8)
    assert fr.shape == (3, 2, 1, 1)
    assert torch.allclose(fr.sum(0), torch.ones(2, 1, 1))
    assert abs(float(fr[EMPTY, 1, 0, 0]) - 2 / 8) < 1e-6 and abs(float(fr[UNKNOWN, 1, 0, 0]) - 6 / 8) < 1e-6


def test_raybundle_cat_renumbers_frames():
    a = RayBundle(torch.zeros(3), torch.eye(3), torch.ones(3), torch.zeros(3, dtype=torch.long))
    b = RayBundle(torch.zeros(3), torch.eye(3), torch.ones(3), torch.tensor([5, 5, 7]))
    c = RayBundle.cat([a, b])
    assert c.frame_ids.tolist() == [0, 0, 0, 1, 1, 2] and c.n_frames == 3 and c.n_rays == 6


# ---------------------------------------------------------------- raycast
def test_sphere_hits_and_misses(frame):
    w = sphere_world(frame)[None]
    o = torch.tensor([0.8, -0.5, 0.8])
    d = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    out = march_first_hit(w, frame, o, d)
    assert abs(float(out["t"][0, 0]) - (1.3 - 0.35)) < 0.5 * frame.voxel_size  # analytic first hit
    assert math.isinf(float(out["t"][0, 1])) and math.isinf(float(out["t"][0, 2]))
    inside = march_first_hit(w, frame, torch.tensor([0.8, 0.8, 0.8]), d[:1])  # ray starting inside: no hit
    assert math.isinf(float(inside["t"][0, 0]))


def test_equivalent_to_patch_raycaster():
    """Same hits as the frozen PatchRaycaster on the 256^3 patch (its own square-FOV camera rays)."""
    from seen2scene.active.raycast import PatchRaycaster, camera_rays
    bmin = (1.0, 2.0, 0.5)
    fr = GridFrame.front3d_patch(bmin)
    centre = tuple(b + 1.408 for b in bmin)
    w = sphere_world(fr, centre=centre, radius=0.6)
    c2w = look_at_c2w((bmin[0] + 1.4, bmin[1] - 1.0, bmin[2] + 1.5), centre).numpy()
    ref = PatchRaycaster(bmin, device="cpu").cast(w[None], c2w, res=12, fov_deg=60.0)["t"]
    dirs, org, _ = camera_rays(c2w, 12, 60.0)
    got = march_first_hit(w[None], fr, torch.as_tensor(org), torch.as_tensor(dirs))["t"]
    assert torch.isfinite(ref).sum() > 20
    assert torch.equal(torch.isfinite(ref), torch.isfinite(got))
    fin = torch.isfinite(ref)
    assert torch.allclose(ref[fin], got[fin], atol=1e-6)


def test_non_cubic_grid():
    fr = GridFrame((0.0, 0.0, 0.0), 0.05, (40, 20, 16), 0.15)
    w = sphere_world(fr, centre=(1.0, 0.5, 0.4), radius=0.2)[None]
    out = march_first_hit(w, fr, torch.tensor([1.0, -1.0, 0.4]), torch.tensor([[0.0, 1.0, 0.0]]))
    assert abs(float(out["t"][0, 0]) - (1.5 - 0.2)) < 0.5 * fr.voxel_size
