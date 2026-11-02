"""G0-6 (training side): surface query targets. Positives within Delta/2 of ANY stored surface (the back surface is a
positive), queries outside [t_lo, t_hi] and padding rays are unsupervised, never negative."""
import sys

import torch

from _common import M, run

import train_v5 as TV  # noqa: E402

DELTA = 8 * M.VS


def test_back_surface_is_positive():
    surf = torch.tensor([[1.0, 3.0] + [float("inf")] * 6])
    g = torch.Generator().manual_seed(0)
    tq, y, v, order, first = TV.surface_queries(surf, torch.tensor([0.1]), torch.tensor([5.0]), g, 8, DELTA)
    assert y[0, 0] == 1 and y[0, 1] == 1 and v[0, 0] and v[0, 1]
    assert order[0, 0] == 1 and order[0, 1] == 2
    assert not v[0, 2:8].any()                         # empty slots unsupervised
    q = tq[0, 8:]
    gt = ((q - 1.0).abs() <= DELTA / 2) | ((q - 3.0).abs() <= DELTA / 2)
    assert torch.equal(y[0, 8:] > 0.5, gt)


def test_out_of_range_unsupervised():
    surf = torch.tensor([[1.0] + [float("inf")] * 7])
    g = torch.Generator().manual_seed(1)
    tq, y, v, _, _ = TV.surface_queries(surf, torch.tensor([0.5]), torch.tensor([0.8]), g, 8, DELTA)
    assert not v[0, 0]                                 # the surface at 1.0 is beyond t_hi -> unsupervised
    assert bool(((tq[0, 8:] >= 0.5) & (tq[0, 8:] <= 0.8) | ~v[0, 8:]).all())


def test_padding_ray_unsupervised():
    surf = torch.full((1, 8), float("inf"))
    g = torch.Generator().manual_seed(2)
    _, _, v, _, _ = TV.surface_queries(surf, torch.tensor([0.5]), torch.tensor([0.0]), g, 8, DELTA)
    assert not v.any()


def test_behind_and_near_negatives():
    surf = torch.tensor([[1.0, 3.0] + [float("inf")] * 6])
    g = torch.Generator().manual_seed(3)
    tq, y, v, _, first = TV.surface_queries(surf, torch.tensor([0.1]), torch.tensor([5.0]), g, 8, DELTA, Q_back=8)
    near = slice(8, 16); back = slice(24, 32)
    assert v[0, 8] and v[0, 9] and not v[0, 10:16].any()          # hard negatives only beside existing surfaces
    assert y[0, 8] == 0 and y[0, 9] == 0
    assert bool(v[0, back].all()) and bool((tq[0, back] > 1.0 + DELTA - 1e-6).all())
    gt = ((tq[0, back] - 3.0).abs() <= DELTA / 2)
    assert torch.equal(y[0, back] > 0.5, gt)
    lg = torch.zeros_like(tq)
    l, c = TV.surf_loss(lg, y, v, tq, first.expand_as(tq))
    assert c["n_behind"] > 0 and c["n_front"] > 0 and c["n_pos"] >= 2


def test_balanced_bce_counts():
    lg = torch.zeros(2, 4); y = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1, 0]]); v = torch.ones(2, 4, dtype=torch.bool)
    l, c = TV.surf_loss(lg, y, v)
    assert c == {"n_pos": 2, "n_neg": 6} and abs(float(l) - 0.6931) < 1e-3


if __name__ == "__main__":
    sys.exit(run([test_back_surface_is_positive, test_out_of_range_unsupervised, test_padding_ray_unsupervised, test_behind_and_near_negatives, test_balanced_bce_counts]))
