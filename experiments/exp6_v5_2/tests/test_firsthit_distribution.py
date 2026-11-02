"""G0: the first-hit pmf is normalised (sum_i p_first + p_nohit = 1) and finite for extreme hazards; analytic constant
hazards give the exact no-hit probability independently of the number of intervals; intervals cover [near, tcap];
the real V5.2 model (random hazard init, V5 decoder) gives a normalised, finite pmf, also for rays with no memory token."""
import json
import math
import sys

import torch
import torch.nn.functional as F

from _common import REPO, V52, run


def test_normalisation_extremes():
    a = torch.tensor([[0.0] * 8, [1e-9] * 8, [0.0, 0.0, 100.0, 1e6, 0.0, 0.0, 0.0, 0.0], [5.0] * 8, [1e-3, 2e-3, 0.5, 3.0, 0.0, 0.1, 0.2, 7.0]])
    f = V52.firsthit(a)
    s = f["p_first"].sum(-1) + f["p_nohit"]
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6), s
    for k in ("p_first", "p_nohit", "log_pf", "log_surv"):
        assert torch.isfinite(f[k]).all() or k == "log_pf", k
    assert not torch.isnan(f["log_pf"]).any()
    assert abs(float(f["p_nohit"][0]) - 1.0) < 1e-7 and float(f["p_nohit"][2]) < 1e-12
    assert abs(float(f["p_first"][2, 2]) - 1.0) < 1e-6                       # everything stops at the opaque interval


def test_constant_hazard_independent_of_bins():
    tcap = torch.tensor([4.0, 9.4])
    for n in (32, 64, 128, 512):
        e = V52.intervals(tcap, n, 16 * V52.VS)
        dt = e[:, 1:] - e[:, :-1]
        rho = 0.05                                                            # per S0
        f = V52.firsthit(rho * dt / V52.S0)
        L = (e[:, -1] - e[:, 0]) / V52.S0
        assert torch.allclose(f["p_nohit"], torch.exp(-rho * L), rtol=1e-4), (n, f["p_nohit"])
        assert torch.allclose(e[:, 0], tcap * 0.5 / 70.0) and torch.allclose(e[:, -1], tcap)
        assert bool((dt > 0).all())


def test_real_model_normalised_and_no_token_rays():
    from seen2scene.futureview import model as M
    cfg = json.load(open(f"{REPO}/experiments/exp6_v5_2/configs/model_v52.json"))
    torch.manual_seed(0)
    m = V52.V52Model(cfg).eval()
    coarse = (torch.rand(3, 16, 20, 32) * 255).to(torch.uint8); coarse[2] = 255
    with torch.no_grad():
        mem = m.memory(m.encode(M.encoder_input(coarse)))
        R = 64
        ext = torch.tensor([16, 20, 32]).float() * V52.S0
        o_in = (ext * 0.4)[None].expand(R, 3)
        o_out = torch.full((R, 3), -5.0)                                       # outside the grid: no local memory token
        d = F.normalize(torch.randn(R, 3), dim=-1)
        for o in (o_in, o_out):
            out = m.belief(mem, o, d, torch.full((R,), 3.0), torch.rand(R, 11))
            s = out["p_first"].sum(-1) + out["p_nohit"]
            assert torch.allclose(s, torch.ones_like(s), atol=1e-5) and torch.isfinite(out["p_first"]).all()


if __name__ == "__main__":
    sys.exit(run([test_normalisation_extremes, test_constant_hazard_independent_of_bins, test_real_model_normalised_and_no_token_rays]))
