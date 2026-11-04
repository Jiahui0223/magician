"""EXP6 V5 / F1 -- CVPR Table 1 registry entries (CPU; generated): table1_export/f1_model_registry_entry.yaml.
One entry per arm for the checkpoint selected by the pre-registered Dev-3 rule (results/summary.json, extended group, official
AUC@20; 2k vs 5k), plus the non-selected 2k / 5k as 'not_selected' records (provenance only). Status
PENDING_TABLE1_EVALUATION unless a Table 1 evaluation of that checkpoint exists (then the status names it). The F1 Dev-3
pilot numbers are attached as development results only; Table 1 reads provenance, never these floats."""
import hashlib
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(F1)))
W = os.path.join(R, "work_dirs", "exp6_v5", "F1")
STEP = {"A2k": ("A", 2000), "A5k": ("A", 5000), "B2k": ("B", 2000), "B5k": ("B", 5000)}


def main():
    S = json.load(open(os.path.join(F1, "results", "summary.json")))
    sel = S["official"]["selection"]["selected"]
    ext = S["official"]["groups"]["extended"]["models"]
    parent_sha = "343f41ab8d566c52007c881b5f32375c3253dd8f1c7a0d8a906d412a165e4516"
    t1 = yaml.safe_load(open(os.path.join(R, "experiments", "cvpr_results", "table1", "MODEL_REGISTRY.yaml")))
    t1_ids = {m["id"] for m in t1["models"]}
    out = {"generated_by": "experiments/exp6_v5/F1/scripts/export_table1.py", "protocol": "experiments/exp6_v5/F1/PROTOCOL.md (f1-v1)",
           "selection_rule": S["official"]["selection"]["rule"], "entries": []}
    for m, (arm, step) in STEP.items():
        p = os.path.join(W, "runs", arm, f"ema_{step:06d}.pt")
        if not os.path.exists(p):
            continue
        mid = f"F1-{'GQ' if arm == 'B' else 'CONT'}-v1@{step // 1000}k"
        e = {"model_id": mid, "model_family": "V5-F1", "arm": f"F1-{arm}", "selected_on_dev3": sel.get(arm) == m,
             "parent_model": "V2-Mix-B0", "parent_ckpt_sha256": parent_sha,
             "checkpoint": os.path.relpath(p, R), "checkpoint_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "weights": "EMA 0.999",
             "training_data": "Outdoor Train-6 minus S-c / S10 shards (no MP3D)", "training_steps_added": step,
             "inference_density": "S10", "scoring_density": "S10", "refresh": "R10", "flow_k": 4, "flow_euler_steps": 10, "planner": "Greedy-NV",
             "architecture_difference": ("V5 Geometry Query only (V5RayDecoder replaces Ray32/Conv1D); no L_surf / L_bridge" if arm == "B"
                                         else "none (parent V2 architecture continued with L_FM)"),
             "selection_split": "Dev (Dev-3, 15 episodes, official AUC@20)",
             "launch": {"command": f"CUDA_VISIBLE_DEVICES=$G V4_P1_MODE=sparse $MG experiments/exp6_v5/scripts/sim_out_v5.py --v4-ckpt $R/{os.path.relpath(p, R)} --group V2M --scene $S --start $K --root $T1W/closed_loop/outdoor/<ID>/V2M",
                        "chain_regression": "B0 weights through this chain reproduce Table 1 B0 barts_s0 bitwise (F1 CHECKPOINT_AUDIT §4)"},
             "dev3_development_result_official": {k: ext[m][k] for k in ("auc20", "auc100", "final", "cov20")} | {"episodes": ext[m]["n_episodes"]},
             "status": ("REGISTERED_IN_TABLE1" if mid in t1_ids else "PENDING_TABLE1_EVALUATION") if sel.get(arm) == m else "NOT_SELECTED (provenance only)"}
        out["entries"].append(e)
    os.makedirs(os.path.join(F1, "table1_export"), exist_ok=True)
    yaml.safe_dump(out, open(os.path.join(F1, "table1_export", "f1_model_registry_entry.yaml"), "w"), sort_keys=False, allow_unicode=True, width=200)
    print("selected", sel, "entries", [(e["model_id"], e["status"]) for e in out["entries"]])


if __name__ == "__main__":
    main()
