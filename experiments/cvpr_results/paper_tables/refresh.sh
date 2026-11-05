#!/bin/bash
# Regenerate the CVPR paper tables / figures from the latest VALIDATED CVPR Table 1 results.
#   1. Table 1: validate.py + aggregate.py (official coverage)  -> experiments/cvpr_results/table1/results/*
#   2. paper_tables: make_tables.py + make_figures.py           -> tables/, figures/, MANIFEST.json (checks + input sha256)
# Any failed consistency check aborts with a non-zero exit code; nothing in table1/ or work_dirs/ is modified by step 2.
set -e
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python
cd $R/experiments/cvpr_results/table1/scripts
$PY validate.py > /dev/null
$PY aggregate.py > /dev/null
cd $R
$PY experiments/cvpr_results/paper_tables/scripts/make_tables.py
$PY experiments/cvpr_results/paper_tables/scripts/make_figures.py | head -1
