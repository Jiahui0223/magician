"""G0 integration check on REAL cached data (seen2scene env, 1 GPU):
  1. cache vs live fusion: replay one outdoor trajectory, compare partial_tsdf.aggregate_fusion() of the live map with
     the cached blob at several states (bitwise), structure_lean == fu.structure() (bitwise), and the V2 coarse grid of
     the lean structure == the E2 shard coarse grid
  2. data path: V3Views of one MP3D and one outdoor trajectory, a mixed batch through build_mixed with T0 and T1,
     FM / recon / KL terms finite, gradients reach the VAE encoder (mu and logvar), ray encoder, U-Net (and xattn)
  3. ray extent: rays of a view enter the 4x grid (fraction of ray sample points inside the grid)
Output: JSON to stdout. Usage: python g0_real_data_check.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

from seen2scene import PROJECT_ROOT
from seen2scene.futureview import geometry as G
from seen2scene.futureview import model as M
from seen2scene.v3 import data as D
from seen2scene.v3 import losses as L
from seen2scene.v3 import partial_tsdf as PT
from seen2scene.v3 import trainer as TR

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v3", "shared"))
import build_tsdf_cache as B  # noqa: E402


def check_cache_vs_live(dev, scene="bannerman", tj="cg-A-0000", times=(0, 1, 10, 55, 99)):
    cache = D.TSDFCache(D.cache_path("outdoor", scene, tj))
    fr, dims, dom, win = B.TD.scene_geometry(scene)
    rec_dir = os.path.join(B.VF.DATA_V1, "trajectories", scene, tj, "records")
    files = {int(f.split("_t")[-1][:-4]): os.path.join(rec_dir, f) for f in os.listdir(rec_dir) if f.endswith(".npz")}
    shard = np.load(D.shard_path("outdoor", scene, tj))
    fu = B.PF.P1FusionFull(fr, dims, dom, device=dev)
    out = {}
    for t in range(max(times) + 1):
        fu.integrate_record(B.MC.load_service_record(files[t]))
        if t in times:
            live = PT.aggregate_fusion(fu)
            cached = PT.from_blob(cache.blobs[t], tuple(cache.block(t)["shape4"]))
            same = all(np.array_equal(live[k].cpu().numpy(), cached[k]) for k in ("sdf", "band", "free", "dom"))
            lean_eq = bool(torch.equal(PT.structure_lean(fu), fu.structure()))
            coarse_eq = bool(np.array_equal(B.coarse_of(fu), shard["coarse"][t]))
            out[t] = {"cache_eq_live": bool(same), "structure_lean_eq": lean_eq, "coarse_eq_shard": coarse_eq,
                      "band_cells": int((cached["band"] > 0).sum()), "free_cells": int((cached["free"] > 0).sum()), "valid_cells": int((cached["dom"] > 0).sum())}
    del fu
    torch.cuda.empty_cache()
    return out


def check_data_and_grads(dev):
    cfg = json.load(open(os.path.join(TR.CFG_DIR, "train_v3.json")))
    res = {}
    in_items = [x for x in D.items("indoor", "train") if os.path.exists(D.cache_path("indoor", *x))][:1]
    out_items = [x for x in D.items("outdoor", "train") if os.path.exists(D.cache_path("outdoor", *x))][:1]
    din, dout = D.V3Views("indoor", in_items), D.V3Views("outdoor", out_items)
    res["views"] = {"indoor": din.n_views, "outdoor": dout.n_views, "cache_mb": [din.cache_bytes / 1e6, dout.cache_bytes / 1e6]}
    rng = np.random.default_rng(0)
    parts = [(din, din.sample_batch(rng, 8, 1, 8, 0.5, 9, din.state_weights())), (dout, dout.sample_batch(rng, 8, 1, 8, 0.5, 9, dout.state_weights()))]
    res["batch"] = [[(int(ti), int(t), int(dlt), len(rows)) for ti, t, dlt, rows in b] for _, b in parts]
    for arm in ("t0", "t1"):
        model = TR.make_model(TR.model_cfg(arm, cfg), 0, dev).train()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        bt = D.build_mixed(model, parts, dev)
        V = bt["cond"].shape[0]
        eps = torch.randn(V, 2, M.GH, M.GW_PAD, device=dev)
        tau = torch.rand(V, device=dev)
        esum, nray = L.fm_terms(model, bt, eps, tau)
        rec, diag = L.recon_terms(model, bt["vae"])
        kl, kl_ch = L.kl_terms(bt["vae"])
        loss = esum / (2 * nray.clamp(min=1)) + cfg["lambda_rec"] * rec + cfg["kl"]["beta"] * kl
        loss.backward()
        torch.cuda.synchronize()
        gn = lambda mod: float(sum((p.grad.float() ** 2).sum() for p in mod.parameters() if p.grad is not None) ** 0.5)  # noqa: E731
        head = model.vae_enc.head[-1].weight.grad
        r = {"views": V, "fm": float(esum / (2 * nray.clamp(min=1))), "n_rays": float(nray), "rec": float(rec), "rec_diag": diag, "kl": float(kl),
             "finite": bool(torch.isfinite(loss)), "grad_vae_enc": gn(model.vae_enc), "grad_vae_dec": gn(model.vae_dec), "grad_rays": gn(model.rays),
             "grad_unet": gn(model.unet), "grad_mu_head": float(head[:16].abs().sum()), "grad_logvar_head": float(head[16:].abs().sum()),
             "mem_tokens_mean": None if bt["mem"] is None else float(bt["mem"][1].float().sum(1).mean()),
             "seconds_fwd_bwd": time.time() - t0, "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
             "grid_shapes": [list(s["x4"].shape[2:]) for s in bt["vae"]], "latent_shapes": [list(s["mu"].shape[2:]) for s in bt["vae"]]}
        if model.xa:
            r["grad_xattn"] = gn(model.unet.xattn)
        res[arm] = r
        del model, bt
        torch.cuda.empty_cache()
    # ray extent: fraction of ray sample points inside the 4x grid of their state (indoor and outdoor)
    for nm, dd in (("indoor", din), ("outdoor", dout)):
        ti, t = dd.states[len(dd.states) // 2]
        st = dd.state_tensors(ti, t, 0, dd.state_rows[(ti, t)][:4], dev)
        ext = PT.extent(st["x4"].shape[2:]).to(dev)
        n = 32
        tt = (torch.arange(n, device=dev) + 0.5) / n
        p = st["o"][:, None, None] + st["d"][:, :, None] * (st["tcap"][:, :, None] * tt[None, None])[..., None]
        inside = ((p >= 0) & (p < ext)).all(-1).float()
        res[f"ray_points_inside_{nm}"] = float(inside.mean())
        res[f"cam_inside_{nm}"] = bool(((st["o"] >= 0) & (st["o"] < ext)).all())
    return res


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    out = {"cache_vs_live_outdoor": check_cache_vs_live(dev), "data_and_grads": check_data_and_grads(dev)}
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
