#!/bin/bash
# Experiment G -- formal D0 training (protocol G-v1 section 2): run_train.sh det on the 4 GPUs of this node until the trainer writes
# run_none.json (40,000 steps or the frozen early-stop rule). A crashed attempt is resumed from ckpt_last.pt by the unchanged trainer (its
# rule: generators re-seeded with (seed, rank, step); logged as "resumed at step"); at most 4 attempts; no recipe change on retry.
# Usage: setsid nohup bash train_d0_chain.sh > /dev/null 2>&1 < /dev/null &
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
GD=$R/experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic
W=$R/work_dirs/exp6_v3/experiment_G/d0; RUN=$W/runs/main; LOG=$W/logs/train.log; CH=$W/logs/chain.log
mkdir -p $RUN $W/logs
say() { echo "$(date '+%F %T') $(hostname -s) $*" >> $CH; }
say "CHAIN START pid $$ (protocol G-v1, sha $(sha256sum $GD/reports/G0_PROTOCOL.md | cut -c1-16))"
n=0
while [ ! -f $RUN/run_none.json ]; do
  n=$((n+1)); [ $n -gt 4 ] && { say "GAVE UP after 4 attempts"; exit 1; }
  say "attempt $n ($([ -f $RUN/ckpt_last.pt ] && echo resume from ckpt_last.pt || echo scratch))"
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash $GD/scripts/run_train.sh det $RUN $LOG
  rc=$(tail -1 $LOG | awk '{print $2}')
  say "attempt $n exit $rc"
  [ -f $RUN/run_none.json ] || sleep 120
done
say "CHAIN DONE: run_none.json written; next: bash $GD/scripts/post_train.sh"
