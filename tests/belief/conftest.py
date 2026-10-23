"""Synthetic scenes and mocks for the belief tests (CPU only; run with CUDA_VISIBLE_DEVICES= )."""
import math
import os
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from seen2scene.belief.interfaces import GenerationOutput  # noqa: E402
from seen2scene.belief.observation import EMPTY, UNKNOWN, ObservationState  # noqa: E402
from seen2scene.belief.state import GridFrame  # noqa: E402

VS = 0.05
N = 32


def small_frame() -> GridFrame:
    return GridFrame((0.0, 0.0, 0.0), VS, (N, N, N), 3 * VS)


def sphere_world(frame: GridFrame, centre=(0.8, 0.8, 0.8), radius=0.35) -> torch.Tensor:
    """Dense TSDF of a sphere on voxel centres, clamped to +-truncation."""
    ax = [(torch.arange(n, dtype=torch.float64) + 0.5) * frame.voxel_size + frame.origin[i] for i, n in enumerate(frame.shape)]
    x, y, z = torch.meshgrid(*ax, indexing="ij")
    c = torch.tensor(centre, dtype=torch.float64)
    d = torch.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) - radius
    return d.clamp(-frame.truncation, frame.truncation).float()


def empty_world(frame: GridFrame) -> torch.Tensor:
    return torch.full(frame.shape, frame.truncation)


def look_at_c2w(eye, target=(0.8, 0.8, 0.8), up=(0.0, 0.0, 1.0)) -> torch.Tensor:
    """OpenGL cam2world looking from eye at target."""
    eye, target, up = (torch.tensor(v, dtype=torch.float64) for v in (eye, target, up))
    f = target - eye
    f = f / f.norm()
    r = torch.linalg.cross(f, up)
    r = r / r.norm()
    u = torch.linalg.cross(r, f)
    m = torch.eye(4, dtype=torch.float64)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, eye
    return m


def render_depth(world: torch.Tensor, frame: GridFrame, c2w: torch.Tensor, res: int = 24, fov_deg: float = 50.0):
    """z-depth image of a world seen from c2w (pixel centres), 0 = no hit; uses the belief ray caster."""
    from seen2scene.belief.observation import DepthObservation
    from seen2scene.belief.raycast import march_first_hit
    probe = DepthObservation.from_fov(torch.ones(res, res), c2w, fov_deg)
    rays = probe.to_rays()
    t = march_first_hit(world[None], frame, rays.origins, rays.dirs)["t"][0]
    # convert ray distance back to z-depth: z = t / |d_cam|
    v, u = torch.meshgrid(torch.arange(res), torch.arange(res), indexing="ij")
    x = (u.reshape(-1).double() + 0.5 - probe.cx) / probe.fx
    y = -(v.reshape(-1).double() + 0.5 - probe.cy) / probe.fy
    norm = torch.sqrt(1 + x ** 2 + y ** 2)
    z = torch.where(torch.isfinite(t), t.double() / norm, torch.zeros_like(norm))
    return DepthObservation.from_fov(z.reshape(res, res).float(), c2w, fov_deg)


def unknown_state(frame: GridFrame) -> ObservationState:
    return ObservationState.unobserved(frame)


def observed_state_from(world: torch.Tensor, frame: GridFrame) -> ObservationState:
    """A fully observed grid with the world's TSDF (BAND where |w| < trunc, EMPTY elsewhere outside)."""
    tsdf = world.clone()
    tsdf[world >= frame.truncation * 0.999] = frame.truncation
    return ObservationState(frame, tsdf, 1)


class MockGenerator:
    """Returns copies of a template world (optionally failing / NaN) and records every request."""

    def __init__(self, template: torch.Tensor, fail: bool = False, nan: bool = False):
        self.template, self.fail, self.nan, self.requests = template, fail, nan, []

    def generate(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("mock generator failure")
        w = self.template[None].repeat(len(request.keys), 1, 1, 1).clone()
        if self.nan:
            w[0, 0, 0, 0] = math.nan
        return GenerationOutput(w)


def wall_world(frame: GridFrame, y: float = 1.1) -> torch.Tensor:
    """A plane y = const across the whole grid (every ray of the test camera hits it)."""
    ys = (torch.arange(frame.shape[1], dtype=torch.float64) + 0.5) * frame.voxel_size + frame.origin[1]
    d = (y - ys).clamp(-frame.truncation, frame.truncation).float()
    return d[None, :, None].expand(frame.shape).contiguous()


@pytest.fixture
def wall(frame):
    true = wall_world(frame)
    obs = render_depth(true, frame, look_at_c2w((0.8, -0.6, 0.8), (0.8, 0.8, 0.8)), fov_deg=40.0)
    return {"frame": frame, "true": true, "obs": obs, "rays": obs.to_rays(),
            "prior": unknown_state(frame), "post": observed_state_from(true, frame)}


@pytest.fixture
def frame():
    return small_frame()


@pytest.fixture
def scene(frame):
    true = sphere_world(frame)
    obs = render_depth(true, frame, look_at_c2w((0.8, -0.6, 0.9)))
    return {"frame": frame, "true": true, "obs": obs, "rays": obs.to_rays(),
            "prior": unknown_state(frame), "post": observed_state_from(true, frame)}


__all__ = ["EMPTY", "UNKNOWN"]
