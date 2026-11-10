#!/bin/bash
# Regenerate every CVPR overnight table / figure / report from the canonical records (CPU only, ~3 min).
# Order: manifest (+ record validation) -> Tables 1-3 -> Fig. 3 / 5 -> K = 1 pairing check -> Fig. 4 (only when the scan and the
# extended back-face check of all six pool scenes exist) -> quality gates -> reports -> checksums.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python; S=experiments/cvpr_results/scripts; D=work_dirs/cvpr_results/overnight_20261110/intermediates/fig4
set -e
$PY $S/build_manifest.py
$PY $S/make_tables.py
$PY $S/make_table1_v2.py
$PY $S/make_table2_v2.py
$PY $S/table1_metric_profile.py
$PY $S/make_figures.py
$PY $S/table3/k1_pairing_check.py
ok=1; for s in bannerman redeemer colosseum pantheon alhambra neuschwanstein; do
  [ -f $D/$s.jsonl ] && [ "$(wc -l < $D/backface_$s.jsonl 2>/dev/null)" = "30" ] && grep -q back_share_hist_max $D/backface_$s.jsonl || ok=0; done
if [ $ok = 1 ]; then $PY $S/fig4/fig4_plot.py; else echo "Fig. 4 NOT regenerated: scan / extended back-face check incomplete"; fi
$PY $S/validate.py
$PY $S/make_reports.py
cp experiments/cvpr_results/OVERNIGHT_AUDIT.md experiments/cvpr_results/report/AUDIT.md
( cd experiments/cvpr_results && find fig manifest report scripts OVERNIGHT_AUDIT.md OVERNIGHT_RUN_PLAN.md -type f ! -name sha256.txt ! -path "*/__pycache__/*" | sort | xargs sha256sum > manifest/sha256.txt )
echo "sha256.txt: $(wc -l < experiments/cvpr_results/manifest/sha256.txt) files"
