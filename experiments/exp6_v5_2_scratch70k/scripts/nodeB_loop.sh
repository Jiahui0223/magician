#!/bin/bash
# Control loop of node B (inside its sbatch allocation). ctl/round.json = {"round": id, "nnodes": 2, "master_addr", "master_port", "args"}
# -> torchrun --nnodes 2 --node_rank 1 ...; ctl/eval_task.json = {"id", "cmd"} -> run the evaluation command; ctl/SHUTDOWN -> exit.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; C=$R/work_dirs/exp6_v5_2_scratch70k/ctl; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
cd $R; mkdir -p $C
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
END=$(squeue -j $SLURM_JOB_ID -h -o %e)
done_round=""; done_eval=""
# a round / eval task announced before this loop started is stale (its head ranks are gone): never join it
[ -f $C/round.json ] && done_round=$($S2E/bin/python -c "import json; print(json.load(open('$C/round.json'))['round'])")
[ -f $C/eval_task.json ] && done_eval=$($S2E/bin/python -c "import json; print(json.load(open('$C/eval_task.json'))['id'])")
status() { echo "{\"host\": \"$(hostname -s)\", \"job\": \"$SLURM_JOB_ID\", \"state\": \"$1\", \"round\": \"$done_round\", \"eval\": \"$done_eval\", \"end\": \"$END\", \"time\": \"$(date '+%F %T')\", \"unix\": $(date +%s)}" > $C/nodeB_status.json.tmp && mv $C/nodeB_status.json.tmp $C/nodeB_status.json; }
while true; do
  [ -f $C/SHUTDOWN ] && { status shutdown; echo "$(date '+%F %T') SHUTDOWN"; exit 0; }
  status idle
  if [ -f $C/round.json ]; then
    rid=$($S2E/bin/python -c "import json; print(json.load(open('$C/round.json'))['round'])")
    if [ "$rid" != "$done_round" ]; then
      eval $($S2E/bin/python -c "
import json; r = json.load(open('$C/round.json'))
print('NN=%d MA=%s MP=%d' % (r['nnodes'], r['master_addr'], r['master_port'])); print('ARGS=\"%s\"' % ' '.join(r['args']))")
      if [ "$NN" = "2" ]; then
        done_round=$rid; status training
        echo "$(date '+%F %T') round $rid: torchrun node_rank 1 -> $MA:$MP"
        $S2E/bin/torchrun --nnodes 2 --node_rank 1 --nproc_per_node 4 --master_addr $MA --master_port $MP $ARGS >> $R/work_dirs/exp6_v5_2_scratch70k/logs/train_nodeB_round$rid.log 2>&1
        ec=$?; echo "$(date '+%F %T') round $rid exit $ec"; echo $ec > $C/nodeB_round${rid}.exit
      else
        done_round=$rid
      fi
    fi
  fi
  if [ -f $C/eval_task.json ]; then
    eid=$($S2E/bin/python -c "import json; print(json.load(open('$C/eval_task.json'))['id'])")
    if [ "$eid" != "$done_eval" ]; then
      done_eval=$eid; status evaluating
      cmd=$($S2E/bin/python -c "import json; print(json.load(open('$C/eval_task.json'))['cmd'])")
      echo "$(date '+%F %T') eval task $eid: $cmd"; bash -c "$cmd" >> $R/work_dirs/exp6_v5_2_scratch70k/logs/eval_nodeB_$eid.log 2>&1; echo "$(date '+%F %T') eval $eid exit $?"
      echo done > $C/nodeB_eval_${eid}.done
    fi
  fi
  sleep 10
done
