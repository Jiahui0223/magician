"""Generate configs/<arm>.json for the unified outdoor-only benchmark from the HISTORICAL configs (read-only), applying only
the documented unified-protocol changes. Every config carries a "unified" block with the base config, its sha256 and
the exact list of changed / added keys (old -> new, reason). Run with any python (no torch):
  python experiments/exp_unified_outdoor/scripts/make_configs.py
"""
import copy
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
OUT = os.path.join(EXP, "configs")
PROTO = "experiments/exp_unified_outdoor/TRAINING_PROTOCOL_DRAFT.md"
DATA_ROOT = {"s10": "work_dirs/exp5_future_view/task1/data/views",
             "d100": "work_dirs/exp5_future_view/task3_full_depth_p1/e2/out_views_d100"}
TSDF_ROOT = {"d100": "work_dirs/exp6_v3/shared_cache/tsdf4", "s10": "work_dirs/exp_unified_outdoor/cache/tsdf4_s10"}
EMA_EVERY_2K = list(range(0, 60001, 2000))
BASES = {"v2": "experiments/exp5_future_view/task1/configs/train_main.json",
         "v4": "experiments/exp6_v4/configs/outdoor_train.json",
         "v3t1": "experiments/exp6_v3.1/configs/outdoor_train.json",
         "v4c": "experiments/exp6_v5/configs/train_v4_continue.json",
         "v5": "experiments/exp6_v5/configs/train_v5_outdoor.json"}
TRAINERS = {"v2": "experiments/exp5_future_view/task1/scripts/t12_train.py",
            "v4": "experiments/exp6_v4/scripts/train_v4.py",
            "v3t1": "experiments/exp6_v3.1/scripts/trainer_outdoor.py (+ uo_common.patch_v3t1)",
            "v4c": "experiments/exp6_v5/scripts/train_v5.py --arm v4c (+ uo_common.patch_v5)",
            "v5": "experiments/exp6_v5/scripts/train_v5.py --arm v5 (+ uo_common.patch_v5)"}


def sha(p):
    return hashlib.sha256(open(os.path.join(REPO, p), "rb").read()).hexdigest()


def build(kind, density, arm, changes):
    """changes: list of (key, new_value, reason); key None-valued new -> delete not used. Records old -> new."""
    base_p = BASES[kind]
    base = json.load(open(os.path.join(REPO, base_p)))
    cfg = copy.deepcopy(base)
    log = []
    for k, v, why in changes:
        old = base.get(k, "<absent>")
        if old == v:
            continue
        cfg[k] = v
        log.append({"key": k, "old": old, "new": v, "reason": why})
    unified = {"arm": arm, "density": density, "protocol": PROTO, "trainer": TRAINERS[kind], "base_config": base_p,
               "base_config_sha256": sha(base_p), "changed_keys": log}
    return cfg, unified


def main():
    os.makedirs(OUT, exist_ok=True)
    written = []
    for d in ("s10", "d100"):
        dr = DATA_ROOT[d]
        # ---------------- V2 Original (t12_train format)
        arm = f"v2_{d}"
        cfg, u = build("v2", d, arm, [
            ("version", f"uo_{arm}_v1", "new identity"),
            ("protocol", PROTO, "unified protocol"),
            ("data_root", dr, "density variant (read by uo_common.patch_v2; t12_train itself always reads the S10 shards)"),
            ("ckpt_every", 2000, "unified: full-state checkpoint >= every 2k steps and EMA at 10k/20k/40k/48k/60k (t12_train saves ema_<step> at every ckpt_every)"),
        ])
        cfg["unified"] = u
        written.append((arm, cfg))
        # ---------------- V4 (train_v4 format)
        arm = f"v4_{d}"
        cfg, u = build("v4", d, arm, [
            ("version", f"uo_{arm}_v1", "new identity"),
            ("protocol", PROTO, "unified protocol"),
            ("data_root", dr, "density variant"),
            ("data_doc", f"Macarons++ Train-6 minus S-c (frozen_values.json split), {d.upper()} view shards; zero MP3D samples", "doc"),
            ("steps", 60000, "unified main budget 60k (historical 40k)"),
            ("ema_save_steps", EMA_EVERY_2K, "unified: EMA every 2k (incl. 10k/20k/40k/48k/60k; 48k = parent of V4C / V5)"),
            ("checkpoint_policy", "fixed_ema_step_60000", "fixed EMA at the end of the 60k budget"),
            ("final_checkpoint", "EMA@60000", "fixed EMA at the end of the 60k budget"),
        ])
        cfg["unified"] = u
        written.append((arm, cfg))
        # ---------------- V3-T1 outdoor-only (trainer_outdoor format)
        arm = f"v3t1_{d}"
        cfg, u = build("v3t1", d, arm, [
            ("version", f"uo_{arm}_v1", "new identity"),
            ("note", "Unified outdoor-only V3-T1: exp6_v3.1 outdoor_train.json recipe with the T1 model (xattn enabled at construction, "
                     "as seen2scene.v3.trainer.model_cfg) and the unified changes listed in unified.changed_keys", "doc"),
            ("model", "experiments/exp6_v3/configs/model_v3.json", "the V3 model config shared by T0 / T1 (T1 = xattn switched on at construction); repo-relative (trainer_outdoor.CFG_DIR patched to the repo root)"),
            ("train_set", "train6_minus_Sc", "split name for uo_common v2_pool loader (v2_data.load_split); not read by trainer_outdoor"),
            ("data_root", dr, "density variant (seen2scene.v3.data.shard_path patched)"),
            ("tsdf_cache_root", TSDF_ROOT[d], "density variant of the 4x partial-TSDF cache (seen2scene.v3.data.CACHE patched); S10 cache must be built first"),
            ("steps", 60000, "unified main budget 60k (historical 40k)"),
            ("lr", 2e-4, "unified peak LR (V2 / V4 value); historical V3 1e-4 was an a-priori draft value, never compared with 2e-4 (see protocol §4)"),
            ("ema_save_steps", EMA_EVERY_2K, "unified: EMA every 2k (the trainer also saves EMA at every eval, eval_every 5000)"),
            ("checkpoint_policy", "fixed_ema_step_60000", "fixed EMA at the end of the 60k budget"),
            ("selection", {"rule": "fixed EMA @ 60000 (frozen before training); S-c metrics are monitoring only; no evaluation scene, no early stopping"},
             "fixed EMA at the end of the 60k budget"),
        ])
        u["v3_arm"] = "t1"
        u["sampler"] = "v2_pool"
        u["sampler_doc"] = ("every rank loads all 138 Train-6-minus-S-c trajectories in the V2 split order; one V2-rule draw of "
                            "batch_views / world = 16 views per rank and step (identical (state, views, staleness) to V2 / V4 for the "
                            "same seed / rank), split at a state boundary into accum = 2 micro-batches (memory); historical V3 "
                            "sharded trajectories sorted[rank::world] and drew 2 x 8 views")
        cfg["unified"] = u
        written.append((arm, cfg))
        # ---------------- V4-Continue 12k from the unified V4@48k (train_v5 --arm v4c format)
        parent = {"arm": f"v4_{d}", "step": 48000, "ckpt": f"work_dirs/exp_unified_outdoor/runs/v4_{d}/ema_048000.pt"}
        arm = f"v4c12k_from48k_{d}"
        cfg, u = build("v4c", d, arm, [
            ("version", f"uo_{arm}_v1", "new identity"),
            ("arm", "V4-Continue12k-from-V4@48k", "doc"),
            ("data_root", dr, "density variant (train_v5.OUTD patched)"),
            ("init", f"{parent['ckpt']} (unified V4 {d.upper()} EMA@48k; sha256 recorded in <out>/PARENT.json at the first launch)",
             "parent = new unified V4@48k instead of the historical V4 EMA@40k (train_v5.V4CK / V4SHA patched)"),
            ("stages", "all V4 parameters trained from the unified V4 EMA@48k, same data / batch / EMA / step count as V5", "doc"),
        ])
        u["parent"] = parent
        cfg["unified"] = u
        written.append((arm, cfg))
        # ---------------- V5-Warm 12k from the unified V4@48k (train_v5 --arm v5 format)
        arm = f"v5warm12k_from48k_{d}"
        cfg, u = build("v5", d, arm, [
            ("version", f"uo_{arm}_v1", "new identity"),
            ("arm", "V5-Warm12k-from-V4@48k", "doc"),
            ("model", "experiments/exp6_v5/configs/model_v5.json", "same model config file, repo-relative (train_v5.CFG patched to the repo root)"),
            ("data_root", dr, "density variant (train_v5.OUTD patched)"),
            ("init", f"{parent['ckpt']} (unified V4 {d.upper()} EMA@48k; sha256 recorded in <out>/PARENT.json at the first launch)",
             "parent = new unified V4@48k instead of the historical V4 EMA@40k (train_v5.V4CK / V4SHA patched)"),
        ])
        u["parent"] = parent
        u["labels"] = "work_dirs/exp6_v5/labels/v1 (density independent: built from shard R / T / jitter / k_M / dims / tstar / t / a only)"
        cfg["unified"] = u
        written.append((arm, cfg))
    for arm, cfg in written:
        p = os.path.join(OUT, f"{arm}.json")
        json.dump(cfg, open(p, "w"), indent=1)
        print(f"{arm:26s} {len(cfg['unified']['changed_keys'])} changed keys -> {os.path.relpath(p, REPO)}")


if __name__ == "__main__":
    main()
