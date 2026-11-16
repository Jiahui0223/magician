#!/bin/bash
# T3K2 collection (CPU only, ~1 min): (1) back up manifest/episodes.csv + provenance.json once, (2) add the validated V3-T0-K2D{1,2}-N2 rows to the
# manifest (t3k2_manifest.py; existing rows asserted unchanged), (3) noise-pairing check of all new episodes (fig/table3/k2_noise_pairing.csv),
# (4) Table 3 v4 = make_table3_v2.py --k12-mean (Unseen-6; needs the 30 Unseen-6 episodes of both ids; the All-14 files are written only when
# all 70 + 70 exist). Usage: bash t3k2_collect.sh
export CUDA_VISIBLE_DEVICES=""
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python
S=experiments/cvpr_results/exp9_ablation_qualitative/scripts; BK=experiments/cvpr_results/exp9_ablation_qualitative/manifests/backup_20261116_t3k2
set -e
for f in episodes.csv provenance.json; do [ -f $BK/$f ] || cp -p experiments/cvpr_results/manifest/$f $BK/$f; done
( cd $BK && sha256sum *.py *.sh *.yaml episodes.csv provenance.json > SHA256SUMS )
echo "records: K2D1-N2 $(ls work_dirs/cvpr_results/table1/episodes/V3-T0-K2D1-N2 2>/dev/null | grep -c json)/70, K2D2-N2 $(ls work_dirs/cvpr_results/table1/episodes/V3-T0-K2D2-N2 2>/dev/null | grep -c json)/70"
$PY $S/table3_k2/t3k2_manifest.py
$PY $S/table3_k2/k2_pairing_check.py
$PY $S/table3_fig5/make_table3_v2.py --k12-mean
