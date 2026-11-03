# EXP6 V2 common environment (sourced by the B2 / E2 entry scripts)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
X2=$R/experiments/exp6_v2
EXP5=$R/experiments/exp5_future_view
E2P=$EXP5/task3_full_depth_p1/e2_pilot/scripts
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PY=$S2E/bin/python
