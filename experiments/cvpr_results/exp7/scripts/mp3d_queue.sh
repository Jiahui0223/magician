#!/bin/bash
# EXP7 MP3D-150 queue worker. One worker = one GPU slot; it walks a job list (METHOD:SCAN:LEVEL:START per line) and runs
# every unit not finished and not claimed. Claims are directories (mkdir is atomic on the shared file system, safe across
# nodes); a unit gets at most 3 tries; a failed try's output directory is moved to raw/_failed/<method>/<tag>_try<k>_<host>
# (never deleted); a finished episode.json is never overwritten (t23_in_sim asserts it).
# Usage: bash mp3d_queue.sh GPU JOBLIST [POOL_LABEL]
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
cd $R
G=$1; JL=$2; POOL=${3:-A}
W7=$R/work_dirs/cvpr_results/exp7
RAW=$W7/episodes/mp3d150/raw
CL=$W7/jobs/claims
mkdir -p $CL $W7/logs/episodes $RAW/_failed
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=$R/work_dirs/exp6_v3/task0_vae/checkpoints/final.pt
WLOG=$W7/logs/worker_${POOL}_$(hostname -s)_g${G}_$(basename $JL .txt).log
echo "start $(date '+%F %T') host $(hostname -s) gpu $G joblist $JL pid $$" >> $WLOG
while read -r it; do
  [ -z "$it" ] && continue
  IFS=: read m scan lv st <<< "$it"
  tag=${scan}_L${lv}_S${st}
  case $m in
    OURS)    od=$RAW/OURS/V2M/$tag ;;
    OURS-H0) od=$RAW/OURS-H0/V2M/$tag ;;
    MAG-B)   od=$RAW/MAG-B/MAG-B/$tag ;;
    RANDOM)  od=$RAW/RANDOM/G-RAND/$tag ;;
    *) echo "unknown method $m" >> $WLOG; continue ;;
  esac
  [ -f $od/episode.json ] && continue
  mkdir $CL/${m}_${tag} 2>/dev/null || continue
  echo "$(hostname -s):$G:$$:$POOL $(date '+%F %T')" > $CL/${m}_${tag}/owner
  for try in 1 2 3; do
    if [ -d $od ] && [ ! -f $od/episode.json ]; then
      mkdir -p $RAW/_failed/$m; mv $od $RAW/_failed/$m/${tag}_try${try}_$(hostname -s)_$(date +%s)
    fi
    EL=$W7/logs/episodes/${m}_${tag}.log
    echo "=== try $try $(date '+%F %T') host $(hostname -s) gpu $G pool $POOL" >> $EL
    t0=$(date +%s)
    case $m in
      OURS)    CUDA_VISIBLE_DEVICES=$G /usr/bin/time -v $MG $R/experiments/cvpr_results/exp7/scripts/sim_in_d.py --v3-ckpt $CK --scan $scan --level $lv --start $st --group V2M --out $RAW/OURS >> $EL 2>&1 ;;
      OURS-H0) CUDA_VISIBLE_DEVICES=$G /usr/bin/time -v $MG $R/experiments/exp6_v3/shared/sim_in_v3.py --v3-ckpt $CK --scan $scan --level $lv --start $st --group V2M --out $RAW/OURS-H0 >> $EL 2>&1 ;;
      MAG-B)   CUDA_VISIBLE_DEVICES=$G /usr/bin/time -v $MG $R/experiments/exp5_future_view/task2/scripts/t23_in_sim.py --scan $scan --level $lv --start $st --group MAG-B --out $RAW/MAG-B >> $EL 2>&1 ;;
      RANDOM)  CUDA_VISIBLE_DEVICES=$G /usr/bin/time -v $MG $R/experiments/exp5_future_view/task0/magician_comparison/scripts/t12_sim.py --scan $scan --level $lv --start $st --method RAND-NV --moves 150 --out $RAW/RANDOM >> $EL 2>&1 ;;
    esac
    rc=$?
    echo "exit $rc $it try $try wall $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $WLOG
    [ -f $od/episode.json ] && { echo "done $(date '+%F %T')" >> $CL/${m}_${tag}/owner; break; }
  done
  [ -f $od/episode.json ] || echo "FAILED_3 $it" >> $WLOG
done < $JL
echo "end $(date '+%F %T')" >> $WLOG
