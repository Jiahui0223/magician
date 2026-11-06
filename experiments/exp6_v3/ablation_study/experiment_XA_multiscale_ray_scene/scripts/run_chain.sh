#!/bin/bash
# Experiment XA -- one chain on the 4 GPUs of this node: for each variant ID in order: scratch training 40K (resumable, retried) -> checkpoint selection ->
# Outdoor D1 All-14 (70 x 100 steps, official coverage) + Indoor H0 (6 x 150 steps) -> aggregation -> next ID. No step depends on a metric.
# Engineering failures are retried (training: resume from ckpt_last.pt; after 2 CUDA OOM failures gradient accumulation 2 with the same global batch;
# episodes: resume / restart with the same seeds); a variant that cannot be completed is recorded INCOMPLETE and the chain goes on (never COMPLETE without 70 / 70 + 6 / 6).
# Usage: run_chain.sh PORT ID [ID ...]      (inside the node's Slurm allocation; setsid nohup)
R=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene; cd $R
X=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene; W=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
PORT=$1; shift; host=$(hostname -s); LOG=$X/logs/chain_$host.log
say() { echo "$(date '+%F %T') $host $*" >> $LOG; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCENES="bannerman fushimi liberty pisa redeemer barts colosseum eiffel pantheon sestino_museum alhambra bridge neuschwanstein"
for id in "$@"; do
  RUN=$W/runs/$id; mkdir -p $RUN $W/eval/$id $W/jobs
  say "$id START"
  # ---------------------------------------------------------------- 1. training
  n=0; oom=0; ov=""
  while [ ! -f $RUN/ema_040000.pt ]; do
    n=$((n+1)); [ $n -gt 8 ] && break
    say "$id train attempt $n ($([ -f $RUN/ckpt_last.pt ] && echo resume || echo scratch)) $ov"
    ( export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
      echo "start $host alloc ${SLURM_JOB_ID:-in-allocation} git $(git rev-parse --short HEAD) $(date '+%F %T') $ov" >> $RUN/train.log
      CUDA_VISIBLE_DEVICES=0,1,2,3 $S2E/bin/torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT --nnodes 1 --nproc_per_node 4 $X/scripts/train_xa.py --id $id --out $RUN $ov >> $RUN/train.log 2>&1
      rc=$?; echo "EXIT $rc $host $(date '+%F %T')" >> $RUN/train.log; exit $rc )
    rc=$?
    if [ $rc -ne 0 ]; then
      if tail -400 $RUN/train.log | grep -qi "out of memory"; then oom=$((oom+1)); [ $oom -ge 2 ] && ov="--override accum=2"; fi
      say "$id train exit $rc (oom $oom)"; sleep 90
    fi
  done
  if [ ! -f $RUN/ema_040000.pt ]; then say "$id INCOMPLETE training failed after $n attempts"; continue; fi
  say "$id TRAINED"
  # ---------------------------------------------------------------- 2. selection
  PYTHONPATH=$R LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib CUDA_VISIBLE_DEVICES="" $S2E/bin/python $X/scripts/select_final.py $id >> $LOG 2>&1
  CK=work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/checkpoints/$id/final.pt
  if [ ! -f $R/$CK ]; then say "$id INCOMPLETE selection failed"; continue; fi
  # ---------------------------------------------------------------- 3. closed loops
  : > $W/jobs/${id}_big.txt; : > $W/jobs/${id}_small.txt
  for k in 0 1 2 3 4; do echo "out $id $CK dunnottar $k" >> $W/jobs/${id}_big.txt; done
  for e in "Z6MFQCViBuw 0 6" "Z6MFQCViBuw 0 17" "oLBMNvg9in8 1 36" "oLBMNvg9in8 1 78" "QUCTc6BB5sX 1 34" "QUCTc6BB5sX 1 135"; do echo "in $id $CK $e" >> $W/jobs/${id}_big.txt; done
  for s in $SCENES; do for k in 0 1 2 3 4; do echo "out $id $CK $s $k" >> $W/jobs/${id}_small.txt; done; done
  for round in 1 2 3; do
    nd=$(ls $W/eval/$id/metrics/V2M/*.json 2>/dev/null | wc -l); ni=$(ls $W/eval/$id/indoor/V2M/*/episode.json 2>/dev/null | wc -l)
    [ $nd -ge 70 ] && [ $ni -ge 6 ] && break
    say "$id eval round $round (outdoor $nd/70, indoor $ni/6)"
    rm -f $W/eval/$id/claims/*.claim 2>/dev/null
    for g in 0 1 2 3; do bash $X/scripts/xa_eval_queue.sh $g $W/jobs/${id}_big.txt & done; wait        # dunnottar and indoor: one job per GPU
    for g in 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3; do bash $X/scripts/xa_eval_queue.sh $g $W/jobs/${id}_small.txt & sleep 2; done; wait
  done
  nd=$(ls $W/eval/$id/metrics/V2M/*.json 2>/dev/null | wc -l); ni=$(ls $W/eval/$id/indoor/V2M/*/episode.json 2>/dev/null | wc -l)
  say "$id EVALUATED outdoor $nd/70 indoor $ni/6"
  # ---------------------------------------------------------------- 4. aggregation
  PYTHONPATH=$R CUDA_VISIBLE_DEVICES="" flock -w 900 $X/logs/.aggregate.lock $S2E/bin/python $X/scripts/aggregate.py >> $LOG 2>&1   # the two chains share the report files
  if [ $nd -ge 70 ] && [ $ni -ge 6 ] && grep -q "\"complete\": true" $X/results/$id/status.json 2>/dev/null; then say "$id COMPLETE"; else say "$id INCOMPLETE outdoor $nd/70 indoor $ni/6 (see results/$id/status.json)"; fi
done
say "CHAIN END $*"
