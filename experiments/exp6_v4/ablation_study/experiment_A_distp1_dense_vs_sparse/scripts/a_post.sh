#!/bin/bash
# a_post.sh : A1 G3 closed loops (4 workers on GPUs 0-3 of this node) -> outdoor coverage pass -> a_analyze_g3.py
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; A=$R/experiments/exp6_v4/ablation_study/experiment_A_distp1_dense_vs_sparse; W=$R/work_dirs/exp6_v4/ablation_study/experiment_A
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
say() { echo "$(date '+%F %T') $*"; }
cd $R; mkdir -p $W/logs $W/eval
say "A1 G3 queue start on $(hostname -s) job ${SLURM_JOB_ID:-local} ($(wc -l < $W/g3_jobs_A1.txt) jobs)"
for g in 0 1 2 3; do bash $A/scripts/a_run_queue.sh $g $W/g3_jobs_A1.txt & done; wait
say "closed loops: $(ls $W/eval/closed_loop/indoor/A1/V2M/*/episode.json 2>/dev/null | wc -l) indoor, $(ls $W/eval/closed_loop/outdoor/A1/V2M/episodes/*.json 2>/dev/null | wc -l) outdoor -> coverage pass"
for g in 0 1 2 3; do bash $A/scripts/a_run_queue.sh $g $W/g3_cov_A1.txt & done; wait
say "metrics: $(ls $W/eval/metrics/outdoor/A1/V2M/*.json 2>/dev/null | wc -l)"
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $A/scripts/a_analyze_g3.py > $W/logs/a_analyze_g3.log 2>&1; say "a_analyze_g3 exit $?"
say "A POST DONE"
