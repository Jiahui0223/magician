"""EXP8 -- parameter counts (CPU, magician env). MAGICIAN: the networks the official MAGICIAN test loop loads through
magician_core.load_model (MACARONS depth + SCONE occupancy / visibility modules with trained_macarons.pth); RAYSCOUT: V3-T0 EMA
checkpoint, counted over the modules used at inference (VAE encoder, ray encoder, flow U-Net; the VAE decoder is trained but not
used by the planner) and in total."""
import json, os, sys
R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task0", "magician_comparison", "adapters"))
import torch
out = {}
import magician_planner as MPL
CORE, MP = MPL.CORE, MPL.MP
p = MP.load_params(os.path.join(MP.configs_dir, "macarons_default_training_config.json"))
for k, v in dict(jitter_probability=0., symmetry_probability=0., anomaly_detection=False, jz=False, numGPU=0, WORLD_SIZE=1, batch_size=1, total_batch_size=1).items():
    setattr(p, k, v)
m = CORE.load_model(p, torch.device("cpu"))
mods = {n: v for n, v in vars(m).items() if isinstance(v, torch.nn.Module)}
sub = {n: sum(x.numel() for x in c.parameters()) for n, c in mods.items()}
tot = sum(sub.values())
out["MAGICIAN"] = {"total": tot, "children": sub, "used_in_planning": "scone occupancy (compute_scene_occupancy_probability_field); depth module unused with GT depth"}
ck = torch.load(os.path.join(R, "work_dirs", "exp6_v3", "task0_vae", "checkpoints", "final.pt"), map_location="cpu", weights_only=False)
sd = ck["ema"]
grp = {}
for k, v in sd.items():
    g = k.split(".")[0]
    grp[g] = grp.get(g, 0) + v.numel()
out["RAYSCOUT_V3-T0"] = {"total": sum(grp.values()), "by_module": grp,
                          "used_at_inference": sum(v for g, v in grp.items() if g in ("vae_enc", "rays", "unet"))}
os.makedirs(os.path.join(R, "experiments", "cvpr_results", "efficiency", "tables"), exist_ok=True)
json.dump(out, open(os.path.join(R, "experiments", "cvpr_results", "efficiency", "tables", "model_params.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
