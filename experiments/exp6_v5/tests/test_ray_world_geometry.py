"""G0-3: the V5 retrieval uses the V4 grid convention (cell (i, j, k) of F0 = [i S0, (i + 1) S0) x ..., the convention of
RayEncoder.grid_sample with align_corners=False) and the same model-frame rays as V4 / the labels."""
import math
import sys

import torch
import torch.nn.functional as F

from _common import M, V5, run, tiny_state


def test_cell_convention_matches_v4_grid_sample():
    X, Y, Z = 16, 20, 32
    F0 = torch.zeros(1, 1, X, Y, Z)
    cell = (5, 7, 11)
    F0[0, 0, cell[0], cell[1], cell[2]] = 1.0
    p = (torch.tensor(cell, dtype=torch.float32) + 0.5) * V5.S0              # the cell centre in model units
    ext = torch.tensor([X, Y, Z], dtype=torch.float32) * V5.S0
    g = 2.0 * p / ext - 1.0
    grid = torch.stack([g[2], g[1], g[0]])[None, None, None, None]
    v = F.grid_sample(F0, grid, mode="bilinear", align_corners=False)
    assert abs(float(v) - 1.0) < 1e-5, float(v)
    assert tuple(torch.floor(p / V5.S0).long().tolist()) == cell


def test_retrieval_hits_cells_on_the_ray():
    coarse = tiny_state()
    cin = M.encoder_input(coarse)
    F0 = torch.randn(1, 32, *coarse.shape[1:]); F1 = torch.randn(1, 64, *[n // 2 for n in coarse.shape[1:]])
    mem = V5.Memory(F0, F1, cin)
    o = torch.tensor([[0.3, 0.3, 0.3]])
    d = F.normalize(torch.tensor([[1.0, 0.7, 0.4]]), dim=-1)
    tcap = torch.tensor([1.2])
    idx, ok, tau, delta = V5.retrieve(mem, 0, o, d, tcap, 24)
    assert ok.any()
    # every valid token is a cell whose centre lies within one cell diagonal of the ray, at increasing depth
    assert float(delta[ok].max()) <= math.sqrt(3) * V5.S0 / 2 + 1e-6
    t = tau[ok]
    assert bool((t[1:] >= t[:-1] - V5.S0).all())
    X, Y, Z = coarse.shape[1:]
    i, r = idx[ok] // (Y * Z), idx[ok] % (Y * Z)
    cen = torch.stack([i, r // Z, r % Z], -1).float().add(0.5) * V5.S0
    rel = cen - o
    assert torch.allclose((rel * d).sum(-1), tau[ok], atol=1e-5)


def test_label_rays_equal_training_rays():
    """The label builder and v2_data.state_tensors both call RY.model_rays_torch with the shard's R, T, jitter, frame."""
    import inspect
    import build_multisurface_labels as B
    src = inspect.getsource(B.view_geometry)
    assert "model_rays_torch" in src


if __name__ == "__main__":
    sys.exit(run([test_cell_convention_matches_v4_grid_sample, test_retrieval_hits_cells_on_the_ray, test_label_rays_equal_training_rays]))
