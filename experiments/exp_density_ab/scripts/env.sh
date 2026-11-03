# EXP Density A/B -- seen2scene env for the fixed-state scripts (fvdb needs the env's libtbb, as in the historical launchers)
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PY=$S2E/bin/python
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
