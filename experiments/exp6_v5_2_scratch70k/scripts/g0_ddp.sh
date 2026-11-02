#!/bin/bash
# G0 DDP: gradient probe (GPU 0) || 2-rank DDP resume consistency on GPUs 1,2 (A 30 vs B 15 + 15, then a world-size change 2 -> 4 resume of B to 45)
# ; then 4-rank throughput (200 steps, real per-rank batch of a 4-rank run).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
cd $R; export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
PY=experiments/exp6_v5_2_scratch70k/scripts/train_v52m.py; W=work_dirs/exp6_v5_2_scratch70k/runs/g0; L=work_dirs/exp6_v5_2_scratch70k/logs
rm -rf $W/gradscale $W/ddp2_A $W/ddp2_B $W/ddp4_short
CUDA_VISIBLE_DEVICES=0 $S2E/bin/python $PY --smoke gradscale --out $W/gradscale > $L/g0_gradscale.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1,2
  $S2E/bin/torchrun --standalone --nproc_per_node 2 $PY --smoke resume --out $W/ddp2_A --override steps=30 ckpt_every=15 > $L/g0_ddp2_A.log 2>&1; echo "ddp2 A exit $?"
  $S2E/bin/torchrun --standalone --nproc_per_node 2 $PY --smoke resume --out $W/ddp2_B --override steps=15 ckpt_every=15 > $L/g0_ddp2_B1.log 2>&1; echo "ddp2 B1 exit $?"
  $S2E/bin/torchrun --standalone --nproc_per_node 2 $PY --smoke resume --out $W/ddp2_B --override steps=30 ckpt_every=15 > $L/g0_ddp2_B2.log 2>&1; echo "ddp2 B2 exit $?" ) &
wait
CUDA_VISIBLE_DEVICES="" $S2E/bin/python -c "
import torch
W = '$W'
a = torch.load(f'{W}/ddp2_A/checkpoints/last.pt', map_location='cpu', weights_only=False); b = torch.load(f'{W}/ddp2_B/checkpoints/last.pt', map_location='cpu', weights_only=False)
print('DDP2 resume: steps', a['step'], b['step'], 'max |dmodel|', max(float((a['model'][k].float() - b['model'][k].float()).abs().max()) for k in a['model']))
" 2>&1 | grep -v -i warn
CUDA_VISIBLE_DEVICES=0,1,2,3 $S2E/bin/torchrun --standalone --nproc_per_node 4 $PY --smoke resume --out $W/ddp2_B --override steps=45 ckpt_every=15 > $L/g0_ddp_world_change.log 2>&1; echo "world 2->4 resume exit $?"
grep -h "resumed" $L/g0_ddp_world_change.log
CUDA_VISIBLE_DEVICES=0,1,2,3 $S2E/bin/torchrun --standalone --nproc_per_node 4 $PY --smoke short --out $W/ddp4_short --override smoke.emulate_world=4 > $L/g0_ddp4_short.log 2>&1; echo "ddp4 short exit $?"
grep -h "short {" $L/g0_ddp4_short.log
echo G0-DDP-DONE
