"""EXP6 V5 -- inference micro-benchmark (seen2scene env, 1 idle GPU): view conditioning of 10 views x 7,296 rays on a 52 x 60 x 32
coarse grid, V4 (32-point RayEncoder + X-Attn) vs V5 (ray-to-memory decoder + belief + adapter), bf16 autocast, no grad;
per-component V5 timings for one 16,384-ray chunk. Usage: python bench_cond.py [det]"""
import json, time, torch, sys
import torch.nn.functional as F
from seen2scene.futureview import model as M, v5_model as V5
dev = torch.device("cuda")
ck4 = torch.load("work_dirs/exp6_v4/checkpoints/final.pt", map_location="cpu", weights_only=False)
v4 = M.V2Model(ck4["model_cfg"]); v4.load_state_dict(ck4["ema"]); v4 = v4.to(dev).eval()
ck5 = torch.load("work_dirs/exp6_v5/checkpoints/v5_final.pt", map_location="cpu", weights_only=False)
v5 = V5.V5Model(ck5["model_cfg"]); v5.load_state_dict(ck5["ema"]); v5 = v5.to(dev).eval()
det = len(sys.argv) > 1
if det:
    torch.use_deterministic_algorithms(True, warn_only=True)
coarse = (torch.rand(3, 52, 60, 32) * 255).to(torch.uint8)
cin = M.encoder_input(coarse).to(dev)
V, R = 10, M.GH * M.GW
ext = torch.tensor([52, 60, 32], device=dev).float() * V5.S0
o = (ext * 0.4)[None].repeat(V, 1); d = F.normalize(torch.randn(V, R, 3, device=dev), dim=-1); tc = torch.full((V, R), 4.0, device=dev); ch = torch.rand(V, 11, R, device=dev)
def t(fn, n=3):
    for _ in range(1): fn()
    torch.cuda.synchronize(); a = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - a) / n
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    h4 = v4.encode(cin); h5 = v5.encode(cin)
    print("deterministic" if det else "default", "| V4 view_cond %.3fs" % t(lambda: v4.view_cond(h4, o, d, tc, ch)), "| V5 view_cond %.3fs" % t(lambda: v5.view_cond(h5, o, d, tc, ch)))
    mem = V5.Memory(h5[0], h5[1], h5[3])
    oo = o[:, None].expand(V, R, 3).reshape(-1, 3)[:16384]; dd = d.reshape(-1, 3)[:16384]; tt = tc.reshape(-1)[:16384]
    print("  retrieve F0 %.4fs" % t(lambda: V5.retrieve(mem, 0, oo, dd, tt, 24)), "F1 %.4fs" % t(lambda: V5.retrieve(mem, 1, oo, dd, tt, 12)),
          "dec chunk %.4fs" % t(lambda: v5.dec(mem, oo, dd, tt, ch.transpose(1, 2).reshape(-1, 11)[:16384])), "Memory %.4fs" % t(lambda: V5.Memory(h5[0], h5[1], h5[3])))
