"""Experiment G3 -- unit tests of the two combination rules (protocol G3-v1 section 3 (b); CPU). Writes work_dirs/exp6_v3/experiment_G/g3/checks/unit_m_combine.json."""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m_combine as MC  # noqa: E402

INF = float("inf")
RES = {}


def check(name, ok, **kw):
    RES[name] = {"pass": bool(ok), **kw}
    print(("PASS " if ok else "FAIL ") + name, kw)


# rays (columns): r0 4 hits, r1 3 hits, r2 2-2 tie, r3 1 hit, r4 0 hits (t_known finite), r5 0 hits (t_known inf), r6 hit in sample but n_u = 0 (hit_u False)
t = torch.tensor([[1.0, 2.0, 3.0, 4.0, 9.0, INF, 5.0],
                  [2.0, 2.0, 5.0, 9.0, 9.0, INF, 5.0],
                  [3.0, 8.0, 9.0, 9.0, 9.0, INF, 5.0],
                  [6.0, 2.6, 9.0, 9.0, 9.0, INF, 5.0]], dtype=torch.float64)
hu = torch.tensor([[1, 1, 1, 1, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
tk = torch.tensor([9.0, 9.0, 9.0, 9.0, 9.0, INF, 5.0], dtype=torch.float64)
tc, hc = MC.combine_depth(t, hu, tk)
exp_h = [True, True, True, False, False, False, False]
exp_t = [3.0, (2.0 + 2.0 + 2.6) / 3, 4.0, 9.0, 9.0, INF, 5.0]
check("depth_hit_rule_ge2_of_4", hc[0].tolist() == exp_h, got=hc[0].tolist(), expected=exp_h)
check("depth_tie_2of4_is_hit", bool(hc[0, 2]) and abs(float(tc[0, 2]) - 4.0) < 1e-12, t_tie=float(tc[0, 2]))
check("depth_mean_over_hitting_samples_only", all((math.isinf(a) and math.isinf(b)) or abs(a - b) < 1e-12 for a, b in zip(tc[0].tolist(), exp_t)),
      got=tc[0].tolist(), expected=exp_t)
check("depth_single_hit_falls_back_to_t_known", float(tc[0, 3]) == 9.0 and not bool(hc[0, 3]))
check("depth_no_hit_keeps_t_known_inf", math.isinf(float(tc[0, 5])))
check("depth_shapes", tuple(tc.shape) == (1, 7) and tuple(hc.shape) == (1, 7) and hc.dtype == torch.bool)
b = torch.tensor([[0.9, -0.9, 0.2], [0.8, -1.0, -0.3], [-0.95, 1.0, 0.1], [0.97, -0.8, -0.1]])
y = torch.tensor([[-0.5, 1.0, 0.2], [0.1, 1.0, 0.4], [1.0, -0.2, 0.0], [-0.3, 1.0, 1.0]])
bb, yy = MC.combine_raw(b, y)
check("raw_is_mean_of_b_and_y", torch.allclose(bb, b.mean(0, keepdim=True)) and torch.allclose(yy, y.mean(0, keepdim=True)) and tuple(bb.shape) == (1, 3),
      b_bar=bb[0].tolist(), y_bar=yy[0].tolist())
hit_raw = ((bb > 0) & (yy < 1))[0].tolist()
check("raw_decode_rule_example", hit_raw == [True, False, False], hit=hit_raw, note="r0 3-1 hits -> hit; r1 1-3 -> no; r2 2-2 tie decided by magnitudes (sum b = -0.1) -> no")
out = {"tests": RES, "all_pass": all(r["pass"] for r in RES.values())}
od = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp6_v3/experiment_G/g3/checks"
os.makedirs(od, exist_ok=True)
json.dump(out, open(os.path.join(od, "unit_m_combine.json"), "w"), indent=1, default=str)
print("ALL PASS" if out["all_pass"] else "SOME FAILED")
sys.exit(0 if out["all_pass"] else 1)
