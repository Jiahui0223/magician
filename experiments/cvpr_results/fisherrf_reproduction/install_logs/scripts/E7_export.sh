#!/bin/bash
source "$(dirname "$0")/common.sh"
O=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/cvpr_results/fisherrf_reproduction
echo "[start] $(date -Is)"
conda list -p "$ENV" --export > "$O/env_fisherrf_conda_list.txt"; echo "conda list exit=$?"
"$ENV/bin/python" -m pip freeze > "$O/env_fisherrf_pip_freeze.txt"; echo "pip freeze exit=$?"
"$ENV/bin/python" -m pip check; echo "pip check exit=$?"
du -sh "$ENV"
echo "[end] $(date -Is)"
