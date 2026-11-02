#!/bin/bash
# G0 resume (1 GPU: A 40 vs B 20 + 20, hazard 20 -> joint at the resume point) on GPU 0, then a 4-rank DDP short run (300 steps).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
cd $R; export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
PY="experiments/exp6_v5_2/scripts/train_v5_2.py"; W=work_dirs/exp6_v5_2/runs/g0; L=work_dirs/exp6_v5_2/logs
rm -rf $W/resume_A $W/resume_B $W/ddp_short
OV="hazard_steps=20 ckpt_every=20"
CUDA_VISIBLE_DEVICES=0 $S2E/bin/python $PY --smoke resume --out $W/resume_A --override steps=40 $OV > $L/g0_resume_A.log 2>&1; echo "A exit $?"
CUDA_VISIBLE_DEVICES=0 $S2E/bin/python $PY --smoke resume --out $W/resume_B --override steps=20 $OV > $L/g0_resume_B1.log 2>&1; echo "B1 exit $?"
CUDA_VISIBLE_DEVICES=0 $S2E/bin/python $PY --smoke resume --out $W/resume_B --override steps=40 $OV > $L/g0_resume_B2.log 2>&1; echo "B2 exit $?"
CUDA_VISIBLE_DEVICES="" $S2E/bin/python - <<'PYEOF'
import json, torch
W = "work_dirs/exp6_v5_2/runs/g0"
a = torch.load(f"{W}/resume_A/ckpt_last.pt", map_location="cpu", weights_only=False); b = torch.load(f"{W}/resume_B/ckpt_last.pt", map_location="cpu", weights_only=False)
out = {"step_A": a["step"], "step_B": b["step"]}
for k in ("model", "ema"):
    out[f"max_abs_diff_{k}"] = max(float((a[k][n].float() - b[k][n].float()).abs().max()) for n in a[k])
out["status"] = "PASS" if out["step_A"] == out["step_B"] == 40 and out["max_abs_diff_model"] < 1e-5 else "CHECK"
print(json.dumps(out)); json.dump(out, open(f"{W}/resume_check.json", "w"), indent=1)
PYEOF
CUDA_VISIBLE_DEVICES=0,1,2,3 $S2E/bin/torchrun --standalone --nproc_per_node 4 $PY --smoke short --out $W/ddp_short > $L/g0_ddp_short.log 2>&1; echo "DDP exit $?"
grep -h "short {" $L/g0_ddp_short.log
echo G0-RESUME-DDP-DONE
