"""G0-4: retrieval masks (outside grid, outside domain, behind camera, duplicates) and keeps UNKNOWN in-domain cells;
a ray with no local token still attends to the null + coarse tokens; no dense rays x cells matrix."""
import sys

import torch
import torch.nn.functional as F

from _common import M, V5, run, tiny_state, v5_cfg


def mem_of(coarse):
    cin = M.encoder_input(coarse)
    X, Y, Z = coarse.shape[1:]
    return V5.Memory(torch.randn(1, 32, X, Y, Z), torch.randn(1, 64, X // 2, Y // 2, Z // 2), cin)


def test_outside_grid_no_local_tokens():
    mem = mem_of(tiny_state())
    o = torch.tensor([[-5.0, -5.0, -5.0]]); d = F.normalize(torch.tensor([[-1.0, 0.0, 0.0]]), dim=-1)
    for lv in (0, 1):
        _, ok, _, _ = V5.retrieve(mem, lv, o, d, torch.tensor([2.0]), 24)
        assert not ok.any()


def test_out_of_domain_masked_unknown_kept():
    c = tiny_state()
    c[0] = 0; c[1] = 0                     # everything in domain is UNKNOWN
    c[2, :, :, 16:] = 0                    # upper half out of domain
    mem = mem_of(c)
    o = torch.tensor([[0.7, 0.9, 0.5]]); d = torch.tensor([[0.0, 0.0, 1.0]])
    idx, ok, tau, _ = V5.retrieve(mem, 0, o, d, torch.tensor([2.5]), 24)
    k = idx[ok] % c.shape[3]
    assert ok.any(), "UNKNOWN in-domain cells must be kept"
    assert bool((k < 16).all()), "out-of-domain cells must be masked"


def test_behind_camera_and_duplicates_masked():
    mem = mem_of(tiny_state())
    o = torch.tensor([[0.5, 0.5, 0.5]]); d = F.normalize(torch.tensor([[1.0, 1.0, 0.2]]), dim=-1)
    idx, ok, tau, _ = V5.retrieve(mem, 0, o, d, torch.tensor([1.5]), 24)
    assert bool((tau[ok] > 0).all())
    v = idx[ok]
    assert bool((v[1:] != v[:-1]).all())


def test_decoder_with_no_local_tokens_is_finite():
    cfg = v5_cfg()
    m = V5.V5Model(cfg).eval()
    coarse = tiny_state()
    h = m.encode(M.encoder_input(coarse))
    o = torch.tensor([[-5.0, -5.0, -5.0]] * 3); d = F.normalize(torch.randn(3, 3), dim=-1)
    H, aux, _ = m.ray_features(h, o, d, torch.full((3,), 2.0), torch.rand(3, 11), return_aux=True)
    assert torch.isfinite(H).all()
    assert int(aux["valid"].sum(1).max()) == 1 + 8      # null + coarse only


if __name__ == "__main__":
    sys.exit(run([test_outside_grid_no_local_tokens, test_out_of_domain_masked_unknown_kept, test_behind_camera_and_duplicates_masked,
                  test_decoder_with_no_local_tokens_is_finite]))
