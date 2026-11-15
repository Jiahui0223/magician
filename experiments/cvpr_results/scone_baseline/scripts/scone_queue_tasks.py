"""EXP9-SCONE -- writes task files for the shared exclusive-GPU queue (work_dirs/cvpr_results/gpu_queue/pending, name-ordered;
one task per GPU; CUDA_VISIBLE_DEVICES set by experiments/cvpr_results/scripts/gpu_queue_worker.sh). Tasks are named scone_*.
Modes:
  smoke   Phase C: 1 outdoor (colosseum start 0, 20 moves) + 1 indoor (oLBMNvg9in8_L1 start 36, 20 moves), each followed by
          scone_smoke_check.py (evaluation side).
  full    Phase D: 70 outdoor (All-14 x 5 official starts, 100 moves; closed loop -> official-coverage replay t25 -> done marker)
          + 44 indoor (EXP7 MP3D-150 manifest, 150 moves; closed loop -> C_obs replay validation). Resumable: a task whose
          final output exists exits 0 immediately; a partial episode directory is moved to _failed/ before a rerun.
Default is a dry run (prints the task list); --write writes the files.
Usage: python scone_queue_tasks.py smoke|full [--write]"""
import csv
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
Q = os.path.join(R, "work_dirs", "cvpr_results", "gpu_queue", "pending")
SB = os.path.join(R, "experiments", "cvpr_results", "scone_baseline", "scripts")
W = os.path.join(R, "work_dirs", "cvpr_results", "scone_baseline")
PY = "env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /bsuscratch/jiahuizhang/envs/magician/bin/python"
SCENES = ["dunnottar", "colosseum", "bannerman", "pantheon", "redeemer", "liberty", "pisa", "fushimi", "alhambra", "neuschwanstein",
          "eiffel", "bridge", "barts", "sestino_museum"]
HEAD = "#!/bin/bash\n# {doc}\ncd {R}\nset -o pipefail\necho \"start $(date '+%F %T') host $(hostname -s) gpu $CUDA_VISIBLE_DEVICES\"\n"


def smoke():
    o = os.path.join(W, "smoke", "outdoor")
    i = os.path.join(W, "smoke", "indoor")
    t_out = HEAD.format(doc="EXP9-SCONE Phase C smoke: outdoor colosseum s0, 20 moves + evaluation-side checks", R=R) + f"""
mkdir -p {o}
[ -f {o}/episodes/ep_scone_colosseum_s0_r0.json ] || {PY} {SB}/scone_sim_out.py --scene colosseum --start 0 --seed 0 --moves 20 --out {o} --diag-dir {o}/diag --diag-steps 0 1 5 10 15 19 || exit 1
{PY} {SB}/scone_smoke_check.py out --scene colosseum --start 0 --root {o} --work {o}/check --diag-dir {o}/diag --eval-steps 1 5 10 15 --out {o}/smoke_check_out.json || exit 2
echo "end $(date '+%F %T')"
"""
    t_in = HEAD.format(doc="EXP9-SCONE Phase C smoke: indoor oLBMNvg9in8_L1 start 36, 20 moves + evaluation-side checks", R=R) + f"""
mkdir -p {i}
[ -f {i}/SCONE/oLBMNvg9in8_L1_S36/episode.json ] || {PY} {SB}/scone_sim_in.py --scan oLBMNvg9in8 --level 1 --start 36 --moves 20 --out {i} --diag-steps 0 1 5 10 15 19 || exit 1
{PY} {SB}/scone_smoke_check.py in --scan oLBMNvg9in8 --level 1 --start 36 --ep-dir {i}/SCONE/oLBMNvg9in8_L1_S36 --eval-steps 1 5 10 15 --out {i}/smoke_check_in.json || exit 2
echo "end $(date '+%F %T')"
"""
    return {"scone_0smoke_a_out_colosseum_s0.sh": t_out, "scone_0smoke_b_in_oLBMNvg9in8_L1_S36.sh": t_in}


def full():
    tasks = {}
    o = os.path.join(W, "outdoor")
    for n, s in enumerate(SCENES):
        for k in range(5):
            tag = f"scone_{s}_s{k}_r0"
            ep = f"{o}/closed_loop/episodes/ep_{tag}.json"
            mt = f"{o}/metrics/scone/{tag}.json"
            tasks[f"scone_d_out_{n:02d}_{s}_s{k}.sh"] = HEAD.format(doc=f"EXP9-SCONE Phase D outdoor {s} start {k} (100 moves)", R=R) + f"""
[ -f {mt} ] && {{ echo "done already: {mt}"; exit 0; }}
mkdir -p {o}/closed_loop {o}/_failed {o}/logs
if [ ! -f {ep} ]; then
  [ -d {o}/closed_loop/frames/{tag} ] && mv {o}/closed_loop/frames/{tag} {o}/_failed/{tag}_frames_$(date +%s)
  {PY} {SB}/scone_sim_out.py --scene {s} --start {k} --seed 0 --moves 100 --out {o}/closed_loop > {o}/logs/sim_{tag}.log 2>&1 || {{ tail -3 {o}/logs/sim_{tag}.log; exit 1; }}
fi
{PY} {SB}/scone_out_cov.py --tag {tag} --root {o}/closed_loop --out {o}/metrics --work {o}/cov_tmp > {o}/logs/cov_{tag}.log 2>&1 || {{ tail -3 {o}/logs/cov_{tag}.log; exit 2; }}
tail -1 {o}/logs/cov_{tag}.log; echo "end $(date '+%F %T')"
"""
    i = os.path.join(W, "indoor")
    rows = [r for r in csv.DictReader(open(os.path.join(R, "experiments", "cvpr_results", "exp7", "manifests", "mp3d150_manifest.csv")))
            if r["method"] == "MAG-B"]
    assert len(rows) == 44, len(rows)
    for n, r in enumerate(rows):
        tag = f"{r['scene_id']}_S{r['start_id']}"
        epd = f"{i}/closed_loop/SCONE/{tag}"
        tasks[f"scone_d_in_{n:02d}_{tag}.sh"] = HEAD.format(doc=f"EXP9-SCONE Phase D indoor {tag} (150 moves)", R=R) + f"""
[ -f {epd}/validation.json ] && grep -q '"valid": true' {epd}/validation.json && {{ echo "validated already: {epd}"; exit 0; }}
mkdir -p {i}/closed_loop {i}/_failed {i}/logs
if [ ! -f {epd}/episode.json ]; then
  [ -d {epd} ] && mv {epd} {i}/_failed/{tag}_partial_$(date +%s)
  {PY} {SB}/scone_sim_in.py --scan {r['scan']} --level {r['level']} --start {r['start_id']} --moves 150 --out {i}/closed_loop > {i}/logs/{tag}.log 2>&1 || {{ tail -3 {i}/logs/{tag}.log; exit 1; }}
fi
{PY} {SB}/scone_validate_in.py --ep-dir {epd} --scene-id {r['scene_id']} --start-id {r['start_id']} --moves 150 > {i}/logs/{tag}_val.log 2>&1 || {{ tail -2 {i}/logs/{tag}_val.log; exit 2; }}
tail -1 {i}/logs/{tag}_val.log; echo "end $(date '+%F %T')"
"""
    return tasks


if __name__ == "__main__":
    mode, write = sys.argv[1], "--write" in sys.argv
    tasks = smoke() if mode == "smoke" else full()
    for name, body in sorted(tasks.items()):
        p = os.path.join(Q, name)
        if write:
            assert not os.path.exists(p), p
            with open(p + ".tmp", "w") as f:
                f.write(body)
            os.replace(p + ".tmp", p)
        print(("wrote " if write else "dry ") + p)
    print(len(tasks), "tasks")
