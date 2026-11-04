"""EXP6 V5 / F1 -- pre-training output statistics of the two conditioning paths on the same states (seen2scene env, 1 GPU):
parent B0 ray context (RayEncoder Ray32/Conv1D, the 64 ray channels of the flow condition) vs the freshly initialised F1-B
V5RayDecoder output (out_scale = 1), on fixed training states of the S10 shards (seed 'exp6-F1|smoke|stats').
Checks: parent strict load; encoder / unet of F1-B torch.equal to the parent; projection channels and valid column of the
two conditions identical; finite outputs; mean / std / rms / per-ray norm; per-ray cosine between the two 64-d vectors.
Frozen rule (PROTOCOL.md §3): out_scale_init = rms(parent ray context) / rms(decoder output at out_scale = 1), so that F1-B
starts with the same condition scale as the parent. Also: parent FM loss on these states (V2Model) for the record.
Output results/smoke_compare.json. Usage: python smoke_compare.py [--states 96 (views)]"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_f1 as T  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=96)
    a = ap.parse_args()
    dev = torch.device("cuda")
    cfgA, _, _ = T.load_cfg("A", [])
    cfgB, _, _ = T.load_cfg("B", ["v5_overrides.out_scale_init=1.0"])
    pp = os.path.join(T.R, cfgA["parent"]["ckpt"])
    assert T.sha256_file(pp) == cfgA["parent"]["sha256"]
    parent = torch.load(pp, map_location="cpu", weights_only=False)
    mA, _, wsA = T.build_model("A", cfgA, parent)
    mB, mcfgB, wsB = T.build_model("B", cfgB, parent)
    mA, mB = mA.to(dev).eval(), mB.to(dev).eval()
    train_items, _ = T.VD.load_split(cfgA["data"]["train_set"])
    data = T.TD.CompactViewData(train_items[:12], log=lambda *x: None)
    states = data.fixed_states("exp6-F1|smoke|stats", a.states)
    rf_all, H_all, cos_all, eq_proj = [], [], [], True
    with torch.no_grad():
        for ti, t, dlt, rows in states:
            st = T.VD.state_tensors(data, ti, t, dlt, rows, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cA = mA.view_cond(mA.encode(st["cin"]), st["o"], st["d"], st["tcap"], st["ch"]).float()
                cB = mB.view_cond(mB.encode(st["cin"]), st["o"], st["d"], st["tcap"], st["ch"]).float()
            pc = mcfgB["proj_ch"]
            eq_proj &= bool(torch.equal(cA[:, :pc].bfloat16(), cB[:, :pc].bfloat16()) and torch.equal(cA[:, -1], cB[:, -1]))   # bf16 = the U-Net input conv precision
            rf = cA[:, pc:pc + 64, :, :T.M.GW].permute(0, 2, 3, 1).reshape(-1, 64)
            H = cB[:, pc:pc + 64, :, :T.M.GW].permute(0, 2, 3, 1).reshape(-1, 64)
            rf_all.append(rf.cpu()); H_all.append(H.cpu())
            cos_all.append(torch.nn.functional.cosine_similarity(rf, H, dim=1).cpu())
    rf, H, cs = torch.cat(rf_all), torch.cat(H_all), torch.cat(cos_all)
    st_ = lambda x: {"mean": float(x.mean()), "std": float(x.std()), "rms": float(x.pow(2).mean().sqrt()), "ray_norm_mean": float(x.norm(dim=1).mean()),  # noqa: E731
                     "ray_norm_p95": float(x.norm(dim=1).quantile(0.95)), "finite": bool(torch.isfinite(x).all()), "chan_std_min": float(x.std(0).min()),
                     "chan_std_max": float(x.std(0).max())}
    out = {"states": len(states), "rays": int(rf.shape[0]), "parent_ray_context": st_(rf), "f1b_decoder_out_scale1": st_(H),
           "cos_parent_vs_new": {"mean": float(cs.mean()), "abs_mean": float(cs.abs().mean())}, "projection_and_valid_channels_equal": eq_proj,
           "warm_start_A": {k: v for k, v in wsA.items()}, "warm_start_B": {k: v for k, v in wsB.items() if k != "not_copied_parent_keys"},
           "not_copied_parent_keys_B": sorted({k.split(".")[0] + "." + k.split(".")[1] for k in wsB["not_copied_parent_keys"]}),
           "params": {"parent_total": int(sum(p.numel() for p in mA.parameters())), "parent_rays": int(sum(p.numel() for n, p in mA.named_parameters() if n.startswith("rays."))),
                      "f1b_total": int(sum(p.numel() for p in mB.parameters())), "f1b_dec": int(sum(p.numel() for n, p in mB.named_parameters() if n.startswith("dec.")))}}
    out["rule_out_scale_init"] = out["parent_ray_context"]["rms"] / out["f1b_decoder_out_scale1"]["rms"]
    os.makedirs(os.path.join(T.F1, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(T.F1, "results", "smoke_compare.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("warm_start_A",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
