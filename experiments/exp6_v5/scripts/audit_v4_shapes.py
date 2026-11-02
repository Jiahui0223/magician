"""EXP6 V5 audit (seen2scene env, 1 GPU): the real V4 checkpoint and one real Train-6 state of the E2 D100 shards ->
sha256, model_cfg, parameter counts per module, tensor shapes along the V4 path (coarse grid, encoder input, F0, F1,
32-point ray samples, ray context, cond, flow output) and along the V5 path (memory tokens per ray, z, belief probes,
CondAdapter output, cond), the scale of the V4 ray context (H_old) and the frame facts (cell sizes, extent, tcap).
Output experiments/exp6_v5/reports/audit_shapes.json. Usage: python audit_v4_shapes.py"""
import hashlib
import json
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task2", "scripts"))
sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task1", "scripts"))
import v2_data as VD  # noqa: E402
import t22_data as TD  # noqa: E402
import torch  # noqa: E402

from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.futureview import v5_model as V5  # noqa: E402

CK = os.path.join(R, "work_dirs", "exp6_v4", "checkpoints", "final.pt")
OUTD = os.path.join(R, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2", "out_views_d100")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    dev = torch.device("cuda")
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    rep = {"v4_ckpt": CK, "v4_sha256": sha(CK), "v4_step": ck["step"], "v4_arm": ck.get("arm"), "v4_model_cfg": ck["model_cfg"]}
    v4 = M.V2Model(ck["model_cfg"]); v4.load_state_dict(ck["ema"]); v4 = v4.to(dev).eval()
    rep["v4_params"] = {k: sum(p.numel() for p in getattr(v4, k).parameters()) for k in ("encoder", "rays", "ray_xattn", "unet")}
    cfg5 = json.load(open(os.path.join(R, "experiments", "exp6_v5", "configs", "model_v5.json")))
    v5 = V5.V5Model(cfg5); ws = V5.load_v4_warmstart(v5, ck["ema"]); v5 = v5.to(dev).eval()
    rep["v5_params"] = {k: sum(p.numel() for p in getattr(v5, k).parameters()) for k in ("encoder", "unet", "dec")}
    rep["v5_dec_params"] = {n: sum(p.numel() for p in getattr(v5.dec, n).parameters()) if isinstance(getattr(v5.dec, n), torch.nn.Module) else getattr(v5.dec, n).numel()
                            for n in ("q_mlp", "f_proj", "geo_mlp", "geo_bias", "null", "ln_kv", "ln_q", "wq", "wk", "wv", "wo", "ffn", "s_z", "s_t", "s_out", "sum_mlp", "adapter", "out_scale")}
    rep["warm_start"] = ws
    items, _ = VD.load_split("train6_minus_Sc")
    data = TD.CompactViewData(items[:1], root=OUTD, log=lambda *a: None)
    rng = __import__("numpy").random.default_rng(0)
    ti, t, dlt, rows = data.sample_batch(rng, 8, 1, 8, 0.5, 9, None)[0]
    st = VD.state_tensors(data, ti, t, dlt, rows, dev)
    sh = lambda x: list(x.shape)  # noqa: E731
    rep["state"] = {"scene_traj": items[0], "t": int(t), "stale": int(dlt), "views": len(rows), "coarse_u8": sh(data.coarse[ti][t - dlt]), "cin": sh(st["cin"]),
                    "o": sh(st["o"]), "d": sh(st["d"]), "tcap": sh(st["tcap"]), "ch": sh(st["ch"]), "x1": sh(st["x1"]), "m_unk": sh(st["m_unk"]),
                    "tcap_model_units": [float(st["tcap"].min()), float(st["tcap"].max())], "labelled_rays": int(st["lmask"].sum())}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h4 = v4.encode(st["cin"])
        V, Rr = st["d"].shape[:2]
        oo = st["o"][:, None].expand(V, Rr, 3).reshape(-1, 3)
        f, inside = v4.rays.point_features(h4[0], h4[1], h4[2], oo, st["d"].reshape(-1, 3), st["tcap"].reshape(-1))
        Hold = v4._ray_context_v4(h4[0], h4[1], h4[2], oo, st["d"].reshape(-1, 3), st["tcap"].reshape(-1)).float()
        c4 = v4.view_cond(h4, st["o"], st["d"], st["tcap"], st["ch"])
        v_4 = v4.velocity(torch.randn(V, 2, M.GH, M.GW_PAD, device=dev), torch.full((V,), 0.5, device=dev), c4)
        h5 = v5.encode(st["cin"])
        c5, aux, mem = v5.view_cond(h5, st["o"], st["d"], st["tcap"], st["ch"], return_aux=True)
    ext = [s * M.BLOCK * M.VS for s in h4[0].shape[2:]]
    rep["v4_path"] = {"F0": sh(h4[0]), "F1": sh(h4[1]), "dom": sh(h4[2]), "point_features": sh(f), "n_points": v4.rays.n, "inside_frac": float(inside.float().mean()),
                      "ray_context_H_old": sh(Hold), "H_old_mean": float(Hold.mean()), "H_old_std": float(Hold.std()), "H_old_rms": float(Hold.pow(2).mean().sqrt()),
                      "cond": sh(c4), "velocity": sh(v_4)}
    rep["v5_path"] = {"F0": sh(h5[0]), "F1": sh(h5[1]), "tokens_per_ray_max": int(aux["valid"].shape[1]), "valid_tokens_per_ray_mean": float(aux["valid"].float().sum(1).mean()),
                      "valid_tokens_per_ray_min": int(aux["valid"].sum(1).min()), "z": sh(aux["z"]), "probe_logits": sh(aux["probe_logit"]), "cond": sh(c5),
                      "H_new_init_rms": float(c5[:, 11:75, :, :M.GW].float().pow(2).mean().sqrt()), "coarse_tokens": sh(mem.coarse_f)}
    rep["frame"] = {"VS": M.VS, "F0_cell": M.BLOCK * M.VS, "F1_cell": 2 * M.BLOCK * M.VS, "grid_extent_model_units": ext, "rays_per_view": M.GH * M.GW, "GH_GW": [M.GH, M.GW]}
    rep["gpu"] = torch.cuda.get_device_name(0)
    out = os.path.join(R, "experiments", "exp6_v5", "reports", "audit_shapes.json")
    json.dump(rep, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("v4_sha256", "v4_params", "v5_params", "state", "v4_path", "v5_path", "frame")}, indent=1, default=str))


if __name__ == "__main__":
    main()
