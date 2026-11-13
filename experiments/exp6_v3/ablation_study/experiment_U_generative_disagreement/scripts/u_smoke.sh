#!/bin/bash
# Experiment U smoke / regression of one short closed loop (queue task; magician env sim + seen2scene planner on CUDA_VISIBLE_DEVICES).
# Usage: u_smoke.sh POLICY SCENE START MOVES NAME [REF_MODEL]
#   closed loop (sim_out_u.py, MOVES moves) -> official coverage replay (u_out_cov.py = t25_out_cov unchanged, must equal the stored
#   curve bitwise) -> check: scores logged, U fields present, finite; REF_MODEL given -> executed keys / coverage prefix vs the canonical
#   Table 1 / Table 3 record of that model (C1 must reproduce V3-T0-K4D1).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
P=$1; S=$2; ST=$3; MV=$4; NAME=$5; REF=$6
U=$R/experiments/exp6_v3/ablation_study/experiment_U_generative_disagreement
O=$R/work_dirs/exp6_v3/experiment_U/smoke/$NAME
[ -d $O ] && mv $O ${O}_old_$(date +%s)
mkdir -p $O
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT /bsuscratch/jiahuizhang/envs/magician/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date '+%F %T') $(hostname -s) gpu $CUDA_VISIBLE_DEVICES worker $WORKER_ID: policy=$P $S s$ST moves=$MV ref=$REF"
t0=$(date +%s)
$MG $U/scripts/sim_out_u.py --v3-ckpt $R/work_dirs/exp6_v3/task0_vae/checkpoints/final.pt --policy $P --group V2M --scene $S --start $ST --moves $MV --root $O > $O/sim.log 2>&1
rc=$?; echo "sim rc=$rc $(( $(date +%s) - t0 ))s"; [ $rc = 0 ] || { tail -30 $O/sim.log; exit 1; }
t1=$(date +%s)
$MG $U/scripts/u_out_cov.py --root $O --out $O/metrics --scene $S --start $ST > $O/cov.log 2>&1
rc=$?; echo "replay rc=$rc $(( $(date +%s) - t1 ))s"; [ $rc = 0 ] || { tail -30 $O/cov.log; exit 2; }
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python - <<PY
import json, glob, math
o = "$O"; ep = json.load(open(glob.glob(o + "/episodes/ep_*.json")[0]))
mt = json.load(open(glob.glob(o + "/metrics/V2M/*.json")[0]))
steps = [json.loads(l) for f in glob.glob(o + "/planner_logs/*.steps.jsonl") for l in open(f)]
pol = "$P"
ok_fields = all(all(k in s["all_scores"] for k in ("u_c1", "u_u1", "u_u2", "u_u3", "u_metrics")) for s in steps if s["n"] > 0)
fin = all(all(math.isfinite(x) for x in s["scores"]) for s in steps)
match = all(s["scores"] == (s["all_scores"]["u_c1"] if pol == "c1" else s["all_scores"]["u_" + pol]) for s in steps if s["n"] > 0)
res = {"name": "$NAME", "policy": pol, "scene": "$S", "start": int("$ST"), "moves": int("$MV"), "n_steps": len(steps),
       "replay_official_equal_stored": mt.get("official_equal_stored"), "cov_final": mt["cov_official"][-1],
       "u_fields_logged": ok_fields, "scores_finite": fin, "scores_equal_policy_metric": match,
       "fallbacks": sum(1 for s in steps if (s.get("choice") or {}).get("fallback")),
       "decision_s_mean": sum(s["decision_s"] for s in steps) / max(1, len(steps)),
       "u_metrics_s_mean": sum((s.get("timing") or {}).get("u_metrics_s", 0) for s in steps) / max(1, len(steps)),
       "peak_mem_gb": max(s.get("peak_mem_gb") or 0 for s in steps)}
ref = "$REF"
if ref:
    r = json.load(open(f"$R/work_dirs/cvpr_results/table1/episodes/{ref}/$S" + "_s$ST.json"))
    n = len(ep["executed_keys"]); res["ref"] = ref
    res["executed_keys_equal"] = f'{sum(a == b for a, b in zip(ep["executed_keys"], r["executed_keys"][:n]))}/{n}'
    res["coverage_max_abs_diff_vs_ref"] = max(abs(a - b) for a, b in zip(ep["coverage"], r["coverage_official"][:len(ep["coverage"])]))
json.dump(res, open(o + "/result.json", "w"), indent=1); print(json.dumps(res))
PY
echo "end $(date '+%F %T')"
