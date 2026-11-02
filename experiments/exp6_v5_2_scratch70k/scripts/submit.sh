#!/bin/bash
# EXP6 V5.2-M Scratch70k -- submission / continuation entry point (run from anywhere).
#   bash submit.sh status      : RUN_STATUS summary, last checkpoint step, own jobs
#   bash submit.sh nodeB       : sbatch the 4-GPU node-B worker (joins a running driver as ranks 4-7 / evaluation node)
#   bash submit.sh continue1   : sbatch a 1-node continuation (driver + 4 ranks; expands if a node-B job is alive)
#   bash submit.sh continue2   : sbatch a 2-node continuation (8 GPU / 60 CPU; needs no other own GPU job under the QOS)
#   bash submit.sh stop        : ask the running driver to checkpoint and stop (ctl/STOP_DRIVER)
# The original run (2026-11-02) used the interactive allocation 3443137 (gpu115) as head node + nodeB.sbatch (job 3449325),
# started with: setsid nohup python scripts/overnight_driver.py --head-job 3443137 > work_dirs/.../logs/driver.log 2>&1 &
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S=$R/experiments/exp6_v5_2_scratch70k/scripts; W=$R/work_dirs/exp6_v5_2_scratch70k
cd $R; mkdir -p $W/logs $W/ctl
case ${1:-status} in
  status) cat $W/checkpoints/last_manifest.json 2>/dev/null; echo; python3 -c "import json; s = json.load(open('$R/experiments/exp6_v5_2_scratch70k/RUN_STATUS.json')); print(s['state'], s.get('phase'), s.get('steps_completed')); print('\n'.join(s['events'][-8:]))"; squeue -u $USER ;;
  nodeB) rm -f $W/ctl/SHUTDOWN; sbatch $S/slurm/nodeB.sbatch | tee -a $W/ctl/submissions.log ;;
  continue1) pgrep -f overnight_driver.py >/dev/null && { echo "a driver is already running on this host"; exit 1; }; sbatch $S/slurm/continue_1node.sbatch | tee -a $W/ctl/submissions.log ;;
  continue2) sbatch $S/slurm/continue_2node.sbatch | tee -a $W/ctl/submissions.log ;;
  stop) touch $W/ctl/STOP_DRIVER; echo "stop requested: the training round saves at the next 50-step check and the driver stops" ;;
  *) sed -n 2,9p $0 ;;
esac
