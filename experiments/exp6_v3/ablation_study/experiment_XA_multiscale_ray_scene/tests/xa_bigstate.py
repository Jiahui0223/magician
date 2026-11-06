"""Experiment XA -- memory / time of one training-style forward + backward on the largest outdoor scene (dunnottar, 8 views of one state), for T1 and the XA variants on the
same state (1 GPU). Engineering check only. Usage: python xa_bigstate.py [ids ...]"""
import json
import os
import sys
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, R)
import torch  # noqa: E402

from seen2scene.v3 import data as D  # noqa: E402
from seen2scene.v3 import losses as L  # noqa: E402
from seen2scene.v3 import model as V3  # noqa: E402
from seen2scene.v3.xa import xa_model as XM  # noqa: E402

X = os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_XA_multiscale_ray_scene")
dev = torch.device("cuda")
it = sorted(x for x in D.items("outdoor", "train") if x[0] == "dunnottar")[:1]
data = D.V3Views("outdoor", it, log=lambda *a: None)
t = max(tt for (ti, tt) in data.states if len(data.state_rows[(ti, tt)]) >= 8)
st = data.state_tensors(0, t, 0, data.state_rows[(0, t)][:8], dev)
print("state", it[0], "t", t, "x4", tuple(st["x4"].shape), flush=True)
out = {}
for vid in (sys.argv[1:] or ["t1", "xa_s2d", "xa_mi"]):
    cfg = json.load(open(os.path.join(R, "experiments", "exp6_v3", "configs", "model_v3.json"))) if vid == "t1" else json.load(open(os.path.join(X, "configs", f"model_{vid}.json")))
    if vid == "t1":
        cfg["xattn"]["enabled"] = True
    torch.manual_seed(0)
    model = XM.build(cfg).to(dev).train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    ts = []
    for rep in range(3):
        torch.cuda.synchronize(); tb = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(st["x4"], st["valid4"])
            cond, mem = model.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"])
        V = cond.shape[0]
        bt = {"m_unk": st["m_unk"], "x1": st["x1"], "lmask": st["lmask"], "cond": cond, "mem": mem}
        es, nr = L.fm_terms(model, bt, torch.randn(V, 2, 64, 128, device=dev), torch.rand(V, device=dev))
        stt = [{"mu": h["mu"], "logvar": h["logvar"], "valid8": h["valid8"], "x4": st["x4"], "valid4": st["valid4"]}]
        rec, _ = L.recon_terms(model, stt)
        kl, _ = L.kl_terms(stt)
        loss = es / (2 * nr.clamp(min=1)) + rec + 0.02 * kl
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize(); ts.append(time.time() - tb)
        del h, cond, mem, bt, loss
    out[vid] = {"peak_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "seconds": min(ts), "finite": all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)}
    print(vid, out[vid], flush=True)
    del model
    torch.cuda.empty_cache()
json.dump({"state": [list(it[0]), int(t)], "x4": list(st["x4"].shape), "results": out}, open(os.path.join(X, "results", "bigstate_memory.json"), "w"), indent=1)
