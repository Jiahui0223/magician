import pytest
import torch

from conftest import observed_state_from, sphere_world, unknown_state
from seen2scene.belief.c1 import C1LatentUpdateNet, C1LatentUpdater, LatentConditionBuilder
from seen2scene.belief.c1.network import window_merge, window_partition
from seen2scene.belief.interfaces import UpdateContext
from seen2scene.belief.observation import ObservationState
from seen2scene.belief.state import BeliefState
from seen2scene.configs.belief import C1Latent, C1Network

TINY = dict(latent_channels=8, cond_channels=15, model_channels=32, num_blocks=2, num_heads=4, window_size=2,
            noise_channels=2, global_noise_dim=8, obs_grid=2)


def inputs(B=2, grid=(4, 4, 4), seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, 8, *grid, generator=g)
    cond = torch.rand(B, 15, *grid, generator=g)
    noise = torch.randn(B, 2, *grid, generator=g)
    code = torch.randn(B, 8, generator=g)
    return z, cond, noise, code


def test_identity_at_initialisation():
    net = C1LatentUpdateNet(**TINY)
    z, cond, noise, code = inputs()
    out = net(z, cond, noise, code, torch.tensor([3.0, 4.0]))
    assert torch.equal(out["z_new"], z) and out["delta"].abs().max() == 0
    assert out["gate"].shape == (2, 1, 4, 4, 4)


def test_parameter_count_default():
    n = sum(p.numel() for p in C1LatentUpdateNet().parameters())
    assert 5e6 < n < 1e7, n


@pytest.mark.parametrize("grid", [(4, 4, 4), (5, 3, 7)])
def test_window_partition_roundtrip(grid):
    x = torch.randn(2, *grid, 6)
    for offset in (0, 1):
        win, valid, meta = window_partition(x, 2, offset)
        assert torch.equal(window_merge(win, meta), x) and int(valid.sum()) == 2 * grid[0] * grid[1] * grid[2]


def test_backward_all_parameters():
    torch.manual_seed(0)
    net = C1LatentUpdateNet(**TINY)
    net.initialize_weights(identity=False)
    z, cond, noise, code = inputs(grid=(5, 3, 7))
    out = net(z, cond, noise, code, torch.tensor([1.0, 2.0]))
    probe = torch.randn_like(out["z_new"])  # an arbitrary scalar of the output (not a training loss)
    (out["z_new"] * probe).sum().backward()
    grads = {n: p.grad for n, p in net.named_parameters()}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    nonzero = [n for n, g in grads.items() if g.abs().sum() > 0]
    assert len(nonzero) >= 0.9 * len(grads), set(grads) - set(nonzero)


def test_directional_derivative_float64():
    """Analytic gradient == central finite difference along random directions. (torch gradcheck with eps 1e-6 is
    not meaningful here: the reused LayerNorm32 normalises in float32 by design.)"""
    torch.manual_seed(0)
    net = C1LatentUpdateNet(**{**TINY, "model_channels": 8, "num_heads": 2, "num_blocks": 1}).double()
    net.initialize_weights(identity=False, std=0.2)
    z, cond, noise, code = (x.double() for x in inputs(B=1, grid=(2, 2, 3)))
    probe = torch.randn(1, 8, 2, 2, 3, dtype=torch.float64)
    f = lambda zz: (net(zz, cond, noise, code)["z_new"] * probe).sum()  # noqa: E731
    z.requires_grad_(True)
    (g,) = torch.autograd.grad(f(z), z)
    h = 1e-3
    for i in range(3):
        v = torch.randn_like(z)
        num = (f(z.detach() + h * v) - f(z.detach() - h * v)) / (2 * h)
        assert abs(float(num) - float((g * v).sum())) <= 1e-3 * max(1.0, abs(float(num)))


def test_noise_controls_the_update():
    torch.manual_seed(0)
    net = C1LatentUpdateNet(**TINY)
    net.initialize_weights(identity=False)
    z, cond, noise, code = inputs()
    a = net(z, cond, noise, code)["z_new"]
    b = net(z, cond, noise, code)["z_new"]
    c = net(z, cond, noise + 1.0, code)["z_new"]
    d = net(z, cond, noise, code + 1.0)["z_new"]
    assert torch.equal(a, b) and not torch.allclose(a, c) and not torch.allclose(a, d)


def test_numerical_stability():
    torch.manual_seed(0)
    net = C1LatentUpdateNet(**TINY)
    net.initialize_weights(identity=False)
    z, cond, noise, code = inputs()
    assert torch.isfinite(net(z * 1e3, cond * 1e3, noise, code)["z_new"]).all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(z, cond, noise, code)
    assert torch.isfinite(out["z_new"].float()).all()


def test_condition_builder(frame):
    true = sphere_world(frame)
    cur = observed_state_from(true, frame)
    cb = LatentConditionBuilder(8)
    c = cb.build(cur, unknown_state(frame))
    assert c.shape == (15, 4, 4, 4) and cb.channels == 15
    assert torch.allclose(c[:3].sum(0), torch.ones(4, 4, 4))
    assert torch.allclose(c[3:6].sum(0), torch.zeros(4, 4, 4), atol=1e-6)  # fraction changes sum to 0
    assert float(c[4].min()) < 0  # UNKNOWN fraction dropped where observed
    lat = torch.randn(8, 4, 4, 4)
    m = (torch.rand(4, 4, 4) > 0.5).float()
    c2 = cb.build(cur, cur, lat, m)
    assert torch.equal(c2[3:6], torch.zeros(3, 4, 4, 4)) and torch.equal(c2[6:14], lat * m) and torch.equal(c2[14], m)


class MockDecoder:
    def __init__(self, frame):
        self.frame = frame

    def decode(self, latents, observation):
        return torch.tanh(latents.mean(1, keepdim=True)).repeat_interleave(8, 2).repeat_interleave(8, 3) \
            .repeat_interleave(8, 4)[:, 0] * self.frame.truncation


def test_c1_updater_protocol(frame):
    net = C1LatentUpdateNet(**TINY)
    net.initialize_weights(identity=False)
    cfg = C1Latent(network=C1Network(**{k: v for k, v in TINY.items()}))
    upd = C1LatentUpdater(cfg, network=net)
    st = BeliefState.from_particles(frame, latents=torch.randn(3, 8, 4, 4, 4))
    cur = observed_state_from(sphere_world(frame), frame)
    ctx = UpdateContext(observation_state=ObservationState(frame, cur.tsdf, 2),
                        previous_observation_state=unknown_state(frame), decoder=MockDecoder(frame), scene="x", step=2)
    a = upd.update(st, None, ctx)
    b = upd.update(st, None, ctx)
    assert torch.equal(a.state.latents, b.state.latents)  # noise keyed by (scene, episode, step, particle id)
    assert a.state.worlds.shape == (3, *frame.shape) and a.state.particle_ids.tolist() == [0, 1, 2]
    assert a.state.obs_version == 2 and not torch.equal(a.state.latents, st.latents)
    assert a.diagnostics["constraint"]["mode"] == "check"
    ident = C1LatentUpdater(cfg, network=C1LatentUpdateNet(**TINY)).update(st, None, ctx)
    assert torch.equal(ident.state.latents, st.latents)  # zero-initialised network = no update
