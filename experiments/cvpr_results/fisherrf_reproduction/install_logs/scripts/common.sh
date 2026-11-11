#!/bin/bash
# shared settings for the fisherrf env install (sourced by every step)
export ENV=/bsuscratch/jiahuizhang/envs/fisherrf
export CONDA_PKGS_DIRS=/bsuscratch/jiahuizhang/cache/conda/pkgs
export PIP_CACHE_DIR=/bsuhome/jiahuizhang/.claude/jobs/fde6a1ae/tmp/pipcache
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export AM=/bsuhome/jiahuizhang/scratch/projects/occupancy/FisherRF/FisherRF-active-mapping
source /cm/shared/apps/conda/23.1.0/etc/profile.d/conda.sh
