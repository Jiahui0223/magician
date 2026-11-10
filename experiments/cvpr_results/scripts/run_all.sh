#!/bin/bash
# Regenerate every CVPR overnight table / figure / report input from the canonical records (CPU only, ~2 min).
# Order: manifest (+ record validation) -> Tables 1-3 -> Fig. 3 / 5 -> Fig. 4 (if the scan outputs exist) -> quality gates -> checksums.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python; S=experiments/cvpr_results/scripts
set -e
$PY $S/build_manifest.py
$PY $S/make_tables.py
$PY $S/make_figures.py
ls work_dirs/cvpr_results/overnight_20261110/intermediates/fig4/*.jsonl > /dev/null 2>&1 && $PY $S/fig4/fig4_plot.py
$PY $S/validate.py
( cd experiments/cvpr_results && find fig manifest report scripts -type f ! -name sha256.txt ! -path "*/__pycache__/*" | sort | xargs sha256sum > manifest/sha256.txt )
echo "sha256.txt: $(wc -l < experiments/cvpr_results/manifest/sha256.txt) files"
