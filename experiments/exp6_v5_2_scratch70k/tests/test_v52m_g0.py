"""G0 of V5.2-M (CPU, seen2scene env): parameters, scratch init, forward / backward, first-hit normalisation, FREE
before hazard (vs the wrong 'compute then delete' order), two walls 3 / 9 with a D100 FREE prefix, the trusted TSDF
crossing bias (thick BAND not forced), truncation at the grid exit, domain gating, the scorer replay adapter
(= the V5.2 aggregation), no flow / depth image / DistP1 call, no checkpoint read by the trainer."""
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v5_2", "tests"))
from _common import FakeGeo, Synth, run, views_x  # noqa: E402

from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.futureview import v52_model as V52  # noqa: E402
from seen2scene.futureview import v52_score as SC52  # noqa: E402
from seen2scene.futureview import v52m_model as VM  # noqa: E402
from seen2scene.futureview import v52m_score as SCM  # noqa: E402

CFG = json.load(open(os.path.join(REPO, "experiments", "exp6_v5_2_scratch70k", "configs", "model_v52m.json")))
VS, S0 = VM.VS, VM.S0


def tiny():
    torch.manual_seed(0)
    m = VM.V52MModel(CFG)
    coarse = (torch.rand(3, 16, 20, 32) * 255).to(torch.uint8); coarse[2] = 255
    return m, M.encoder_input(coarse)


def test_parameters_and_scratch():
    m, _ = tiny()
    t = VM.param_table(m)
    assert t["total"] == 26_999_143 and 20e6 <= t["total"] <= 30e6, t
    torch.manual_seed(1)
    m2 = VM.V52MModel(CFG)
    assert not torch.equal(m.spatial.inp.weight, m2.spatial.inp.weight)          # random init, nothing loaded
    src = open(os.path.join(REPO, "experiments", "exp6_v5_2_scratch70k", "scripts", "train_v52m.py")).read()
    for bad in ("v5_final", "v4_final", "v52_final", "exp6_v4/checkpoints", "load_v5_warmstart", "load_v4_warmstart"):
        assert bad not in src, bad
    assert not hasattr(m, "unet") and not any(k.startswith("unet.") for k in m.state_dict())


def test_forward_backward_all_params():
    m, cin = tiny()
    m.train()
    h = m.encode(cin)
    mem = m.memory(h)
    V, Rr = 2, M.GH * M.GW
    ext = torch.tensor([16, 20, 32]).float() * S0
    o = (ext * 0.4)
    d = F.normalize(torch.randn(V, Rr, 3), dim=-1)
    tc = torch.full((V, Rr), 2.0)
    ch = torch.rand(V, 11, Rr)
    z = m.ray_latents(mem, o[None].expand(V * Rr, 3), d.reshape(-1, 3), tc.reshape(-1), ch.transpose(1, 2).reshape(-1, 11)).reshape(V, Rr, -1)
    c = m.context(z, ch)
    sel = torch.arange(0, Rr, 97)
    n = len(sel)
    st = {"tk": torch.rand(2 * n).half(), "dlo": torch.rand(2 * n).half(), "dhi": torch.ones(2 * n).half(), "cls": torch.randint(0, 3, (2 * n,)), "tmax": torch.full((2 * n,), 2.0)}
    zz, cc = torch.cat([z[0, sel], z[1, sel]]), torch.cat([c[0, sel], c[1, sel]])
    dd = torch.cat([d[0, sel], d[1, sel]])
    out = m.decode(mem, o[None].expand(2 * n, 3), dd, torch.full((2 * n,), 2.0), zz, cc, st)
    s = out["p_first"].sum(-1) + out["p_nohit"]
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5) and torch.isfinite(out["p_first"]).all()
    (out["logit_b"].mean() + out["log_pf"].clamp(min=-30).mean()).backward()
    dead = [k for k, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs().sum()) == 0]
    assert not dead, dead[:8]


def structure(tk=1.0, dlo=1.0, dhi=1.0, tmax=9.4, tcap=9.4, n=256, dom=None):
    edges = V52.intervals(torch.tensor([tcap]), n, 16 * VS)
    st = {"tk": torch.tensor([tk]).half(), "dlo": torch.tensor([dlo]).half(), "dhi": torch.tensor([dhi]).half(), "cls": torch.tensor([0]), "tmax": torch.tensor([tmax]), "dom": dom}
    fr, feats, known = VM.ray_structure(edges, torch.tensor([tcap]), st)
    return edges, fr, known


def _walls(dlo, tcap=9.4):
    edges, fr, _ = structure(dlo=dlo / tcap, tcap=tcap)
    dt = edges[:, 1:] - edges[:, :-1]
    rho = torch.full_like(dt, 1e-4)
    for tw, op in ((1.0, 0.95), (2.8, 0.999), (8.4, 0.999)):
        inside = (edges[:, :-1] <= tw) & (edges[:, 1:] > tw)
        rho = torch.where(inside, -math.log(1 - op) * S0 / dt, rho)
    good = V52.firsthit(rho * dt * (1 - fr) / S0)
    bad = V52.firsthit(rho * dt / S0)
    iA = int(((edges[0, :-1] <= 2.8) & (edges[0, 1:] > 2.8)).nonzero()[0])
    iB = int(((edges[0, :-1] <= 8.4) & (edges[0, 1:] > 8.4)).nonzero()[0])
    return edges, fr, good, bad["p_first"] * (fr < 0.999), iA, iB


def test_free_before_hazard_and_two_walls():
    """Walls A at 2.8 (3 m) and B at 8.4 (9 m), opacity 0.999, tcap 9.4; a spurious network hit at 1.0 (opacity 0.95) inside
    D100 FREE. Case 1: FREE ends at the first UNKNOWN sample 2.6, before A. FREE-before-hazard removes the spurious hit
    BEFORE T: A gets the first-hit mass, opaque A hides B; the wrong order (compute T, then delete the FREE intervals) loses
    A's mass. Case 2: FREE ends exactly at the wall (2.8), inside A's interval: uniform density inside an interval scales
    A's optical depth by (1 - free_frac), so A and B get the analytic masses. In both cases no mass lies inside FREE."""
    tau = -math.log(1 - 0.999)
    edges, fr, good, pf_bad, iA, iB = _walls(2.6)
    assert float(fr[0, iA]) == 0.0 and float(fr[0, iA + 1:].max()) == 0.0
    assert float(good["p_first"][0, iA]) > 0.99, float(good["p_first"][0, iA])
    assert float(pf_bad[0, iA]) < 0.1, float(pf_bad[0, iA])                  # wrong order: the spurious FREE wall ate the mass
    assert float(good["p_first"][0, iB]) < 1e-2, float(good["p_first"][0, iB])
    assert float((good["p_first"] * (fr >= 0.999)).sum()) < 1e-6
    assert abs(float(good["p_first"].sum(-1) + good["p_nohit"]) - 1) < 1e-6
    edges, fr, good, pf_bad, iA, iB = _walls(2.8)
    assert 0.0 < float(fr[0, iA]) < 1.0 and float(fr[0, :iA - 1].min()) > 0.999
    aA = tau * (1 - float(fr[0, iA]))
    exp_A, exp_B = 1 - math.exp(-aA), math.exp(-aA) * 0.999
    assert abs(float(good["p_first"][0, iA]) - exp_A) < 5e-3, (float(good["p_first"][0, iA]), exp_A)
    assert abs(float(good["p_first"][0, iB]) - exp_B) < 5e-3, (float(good["p_first"][0, iB]), exp_B)
    assert float(pf_bad[0, iA]) < 0.1, float(pf_bad[0, iA])
    assert float((good["p_first"] * (fr >= 0.999)).sum()) < 1e-6
    assert abs(float(good["p_first"].sum(-1) + good["p_nohit"]) - 1) < 1e-6


def test_known_crossing_and_band_and_truncation():
    tcap = 9.4
    edges, fr, known = structure(tk=4.0 / tcap, tcap=tcap)                  # KNOWN ray: crossing at 4.0, no unknown before it
    i = int(((edges[0, :-1] <= 4.0) & (edges[0, 1:] > 4.0)).nonzero()[0])
    assert bool(known[0, i]) and int(known.sum()) == 1                       # only the crossing interval is biased
    assert float(fr[0, :i - 1].min()) > 0.999 and float(fr[0, i + 1:].max()) == 0.0
    m, _ = tiny()
    assert float(F.softplus(m.depth.known_bias)) > 0                          # non-negative bias (soft, learnable)
    e2, fr2, k2 = structure(tmax=3.0, tcap=tcap)                             # EMPTY ray leaving the grid at 3.0
    j = int(((e2[0, :-1] <= 3.0) & (e2[0, 1:] > 3.0)).nonzero()[0])
    assert float(fr2[0, :j - 1].min()) > 0.999 and float(fr2[0, j + 1:].max()) == 0.0 and int(k2.sum()) == 0


def test_domain_gating():
    tcap = 9.4
    n = 256
    dom = torch.ones(1, n); dom[0, : n // 2] = 0.0                           # first half of the intervals out of domain
    edges, fr, _ = structure(tmax=9.4, tcap=tcap, dom=dom)
    assert float(fr[0, : n // 2].max()) == 0.0 and float(fr[0, n // 2:].min()) > 0.999


def test_replay_adapter_equals_v52_aggregation():
    syn = Synth([2.8, 8.4], [0.7, 0.999], [0.9, 0.9])
    v = views_x(n_rays=64)
    g = FakeGeo()
    out = syn.belief(None, v["o"][None].expand(64, 3), v["d"], v["tcap"], v["ch"])

    class Fake:
        def memory(self, h):
            return None

        def view_belief(self, h, view, mem=None):
            return dict(out, free_frac=torch.zeros_like(out["p_first"]))
    ref = SC52.score_state(syn, None, g, [v], 1.0, 1.0, dict(SC52.DEFAULT))
    got = SCM.score_state(Fake(), None, g, [v], 1.0, 1.0, dict(SCM.DEFAULT))
    for k in ("s_vis", "p_surf", "p_space"):
        assert abs(ref[k][0] - got[k][0]) < 1e-6, (k, ref[k], got[k])


def test_no_flow_no_distp1_calls():
    from seen2scene.futureview import scoring as SCO
    calls = []
    saved = (M.sample, G.decode, SCO.new_set_from_depth, SCO.new_mask_from_depth)
    M.sample = lambda *a, **k: calls.append("sample")
    G.decode = lambda *a, **k: calls.append("decode")
    SCO.new_set_from_depth = lambda *a, **k: calls.append("distp1")
    SCO.new_mask_from_depth = lambda *a, **k: calls.append("distp1_mask")
    try:
        m, cin = tiny()
        m.eval()
        h = m.encode(cin)
        Rr = M.GH * M.GW
        ext = torch.tensor([16, 20, 32]).float() * S0
        d = F.normalize(torch.randn(Rr, 3), dim=-1)
        tc = torch.full((Rr,), 2.0)
        pr = {"cls": torch.randint(0, 3, (Rr,)), "t_known": torch.where(torch.rand(Rr) < 0.5, torch.full((Rr,), 1.0), torch.full((Rr,), float("inf"))),
              "d_lo": torch.full((Rr,), 0.5), "d_hi": torch.full((Rr,), 1.5), "n_u": torch.randint(0, 50, (Rr,)), "n_seg": torch.randint(0, 3, (Rr,)),
              "tmax": torch.full((Rr,), 1.8)}
        view = {"o": ext * 0.4, "d": d, "tcap": tc, "dz": torch.ones(Rr), "pr": pr}
        with torch.no_grad():
            out = m.view_belief(h, view)
        s = out["p_first"].sum(-1) + out["p_nohit"]
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    finally:
        M.sample, G.decode, SCO.new_set_from_depth, SCO.new_mask_from_depth = saved
    assert calls == [], calls


if __name__ == "__main__":
    sys.exit(run([test_parameters_and_scratch, test_forward_backward_all_params, test_free_before_hazard_and_two_walls, test_known_crossing_and_band_and_truncation,
                  test_domain_gating, test_replay_adapter_equals_v52_aggregation, test_no_flow_no_distp1_calls]))
