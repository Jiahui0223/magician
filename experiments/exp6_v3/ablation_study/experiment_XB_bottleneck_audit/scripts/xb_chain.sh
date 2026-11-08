#!/bin/bash
# Experiment XB / Task C -- one node, 4 GPUs: scratch training of V3-T1 on the reduced outdoor scene set (resumable, retried) -> checkpoint selection ->
# Outdoor D1 All-14 (70 x 100 steps, official coverage) + Indoor H0 (6 x 150 steps) -> aggregation (the XA aggregator with the XB roots). No step depends on a metric.
# Usage: xb_chain.sh PORT ID CONFIG        (inside the node's Slurm allocation; setsid nohup)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
X=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit; W=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp6_v3/ablation_study/experiment_XB_bottleneck_audit; XA=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
PORT=$1; id=$2; cfg=$3; host=$(hostname -s); LOG=$X/logs/chain_$host.log
say() { echo "$(date '+%F %T') $host $*" >> $LOG; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCENES="bannerman fushimi liberty pisa redeemer barts colosseum eiffel pantheon sestino_museum alhambra bridge neuschwanstein"
RUN=$W/runs/$id; mkdir -p $RUN $W/eval/$id $W/jobs
say "$id START (config $cfg)"
n=0; oom=0; ov=""
while [ ! -f $RUN/ema_040000.pt ]; do
  n=$((n+1)); [ $n -gt 8 ] && break
  say "$id train attempt $n ($([ -f $RUN/ckpt_last.pt ] && echo resume || echo scratch)) $ov"
  ( export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
    echo "start $host alloc ${SLURM_JOB_ID:-in-allocation} git $(git rev-parse --short HEAD) $(date '+%F %T') $ov" >> $RUN/train.log
    CUDA_VISIBLE_DEVICES=0,1,2,3 $S2E/bin/torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT --nnodes 1 --nproc_per_node 4 $X/scripts/train_xb.py --config $cfg --out $RUN $ov >> $RUN/train.log 2>&1
    rc=$?; echo "EXIT $rc $host $(date '+%F %T')" >> $RUN/train.log; exit $rc )
  rc=$?
  if [ $rc -ne 0 ]; then
    if tail -400 $RUN/train.log | grep -qi "out of memory"; then oom=$((oom+1)); [ $oom -ge 2 ] && ov="--override accum=2"; fi
    say "$id train exit $rc (oom $oom)"; sleep 90
  fi
done
if [ ! -f $RUN/ema_040000.pt ]; then say "$id INCOMPLETE training failed after $n attempts"; exit 1; fi
say "$id TRAINED"
PYTHONPATH=$R LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib CUDA_VISIBLE_DEVICES="" $S2E/bin/python $X/scripts/xb_select_final.py $id $cfg >> $LOG 2>&1
CK=work_dirs/exp6_v3/ablation_study/experiment_XB_bottleneck_audit/checkpoints/$id/final.pt
if [ ! -f $R/$CK ]; then say "$id INCOMPLETE selection failed"; exit 1; fi
: > $W/jobs/${id}_big.txt; : > $W/jobs/${id}_small.txt
for k in 0 1 2 3 4; do echo "out $id $CK dunnottar $k" >> $W/jobs/${id}_big.txt; done
for e in "Z6MFQCViBuw 0 6" "Z6MFQCViBuw 0 17" "oLBMNvg9in8 1 36" "oLBMNvg9in8 1 78" "QUCTc6BB5sX 1 34" "QUCTc6BB5sX 1 135"; do echo "in $id $CK $e" >> $W/jobs/${id}_big.txt; done
for s in $SCENES; do for k in 0 1 2 3 4; do echo "out $id $CK $s $k" >> $W/jobs/${id}_small.txt; done; done
for round in 1 2 3; do
  nd=$(ls $W/eval/$id/metrics/V2M/*.json 2>/dev/null | wc -l); ni=$(ls $W/eval/$id/indoor/V2M/*/episode.json 2>/dev/null | wc -l)
  [ $nd -ge 70 ] && [ $ni -ge 6 ] && break
  say "$id eval round $round (outdoor $nd/70, indoor $ni/6)"
  rm -f $W/eval/$id/claims/*.claim 2>/dev/null
  for g in 0 1 2 3; do bash $X/scripts/xb_eval_queue.sh $g $W/jobs/${id}_big.txt & done; wait
  for g in 0 1 2 3 0 1 2 3 0 1 2 3; do bash $X/scripts/xb_eval_queue.sh $g $W/jobs/${id}_small.txt & sleep 2; done; wait
done
nd=$(ls $W/eval/$id/metrics/V2M/*.json 2>/dev/null | wc -l); ni=$(ls $W/eval/$id/indoor/V2M/*/episode.json 2>/dev/null | wc -l)
say "$id EVALUATED outdoor $nd/70 indoor $ni/6"
XA_TEST_IDS=$id XA_TEST_W=$W XA_TEST_OUT=$X PYTHONPATH=$R CUDA_VISIBLE_DEVICES="" flock -w 900 $X/logs/.aggregate.lock $S2E/bin/python $XA/scripts/aggregate.py >> $LOG 2>&1
if [ $nd -ge 70 ] && [ $ni -ge 6 ] && grep -q "\"complete\": true" $X/results/$id/status.json 2>/dev/null; then say "$id COMPLETE"; else say "$id INCOMPLETE outdoor $nd/70 indoor $ni/6"; fi
say "CHAIN END $id"
