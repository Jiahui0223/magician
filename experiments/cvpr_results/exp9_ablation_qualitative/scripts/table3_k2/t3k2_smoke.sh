#!/bin/bash
# T3K2 smoke / regression of the kd9b chain (magician env; uses the GPU in CUDA_VISIBLE_DEVICES). Copy of table3_k/t9_smoke.sh with sim_out_kd9b.py
# and a noise offset argument. Usage: t3k2_smoke.sh K DEPTH NOISE SCENE START MOVES NAME [REF_MODEL]
#   REF_MODEL given -> compare executed keys (and simulator coverage) step by step with the canonical Table 1 / Table 3 record.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
K=$1; DEP=$2; NZ=$3; S=$4; ST=$5; MV=$6; NAME=$7; REF=$8
O=$R/work_dirs/cvpr_results/exp9_ablation_qualitative/t3k2/regression/$NAME
mkdir -p $O
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT /bsuscratch/jiahuizhang/envs/magician/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date '+%F %T') $(hostname -s) gpu $CUDA_VISIBLE_DEVICES worker $WORKER_ID: K=$K depth=$DEP noise=$NZ $S s$ST moves=$MV"
t0=$(date +%s)
$MG $R/experiments/cvpr_results/exp9_ablation_qualitative/scripts/table3_k2/sim_out_kd9b.py --v3-ckpt $R/work_dirs/exp6_v3/task0_vae/checkpoints/final.pt \
  --k $K --depth $DEP --noise $NZ --group V2M --scene $S --start $ST --moves $MV --root $O > $O/sim.log 2>&1
rc=$?
echo "sim rc=$rc $(( $(date +%s) - t0 ))s"
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python - <<PY
import json, glob, os
o="$O"; ep=json.load(open(glob.glob(o+"/episodes/ep_*.json")[0]))
h=ep["planner_hello"][0] if isinstance(ep.get("planner_hello"), list) else ep.get("planner_hello", {})
res={"name":"$NAME","K":$K,"depth":$DEP,"noise":$NZ,"scene":"$S","start":$ST,"moves":$MV,"n_steps":len(ep["steps"]),
     "kd":h.get("planner",{}).get("kd"), "planner_K":h.get("planner",{}).get("K")}
ref="$REF"
if ref:
    scene, start = "$S", int("$ST")
    r=json.load(open(f"$R/work_dirs/cvpr_results/table1/episodes/{ref}/{scene}_s{start}.json"))
    n=len(ep["executed_keys"]); res["ref"]=ref
    res["executed_keys_equal"]=f'{sum(a==b for a,b in zip(ep["executed_keys"], r["executed_keys"][:n]))}/{n}'
    re_=json.load(open(r["episode_file"]))
    res["sim_coverage_max_abs_diff"]=max(abs(a-b) for a,b in zip(ep["coverage"], re_["coverage"][:len(ep["coverage"])]))
    res["sim_coverage_bitwise_equal"]=ep["coverage"]==re_["coverage"][:len(ep["coverage"])]
json.dump(res, open(o+"/result.json","w"), indent=1); print(json.dumps(res))
PY
