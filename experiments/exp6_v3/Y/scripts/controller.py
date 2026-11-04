"""EXP6 V3 Y / ablation D -- one controller pass (seen2scene env, CPU, run every 60 s by controller_loop.sh on the node that owns
the track's processes). Read-mostly; idempotent; never deletes historical files.
  --track Y : reads work_dirs/exp6_v3/Y/runs/main (train.log, log.jsonl, eval.jsonl, ckpt_last.pt, ema_*.pt)
      * every new ckpt_last step that is a multiple of 1000 -> atomic copy ckpt_step<step>.pt, sha256 into
        checkpoint_manifest.json; rolling: keep the newest 3 step copies (milestone EMA files ema_<step>.pt are never touched)
      * process check (torchrun / train_y.py on this node); dead without EXIT 0 and below the stop step -> restart
        train_y.sh (resume from ckpt_last.pt), at most 2 automatic restarts (control.json max_restarts)
      * control.json {"stop_at": N}: once ckpt_last.pt has step >= N and ema_<N>.pt exists, stop the run cleanly (SIGTERM
        to the torchrun process group) and record it
  --track D : counts the Table 1 canonical records of the D models (and Y models evaluated in the D queue) per split,
      failures / retries / gave-ups from the Table 1 queue logs; if no t1_queue worker of the D job file is alive on
      this node while jobs remain -> restart control.json["workers"] workers per GPU (at most 2 automatic restarts)
Both: STATUS.json, STATUS.md (work dir + tracked experiments copy), EVENTS.jsonl (append-only), heartbeat."""
import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import subprocess
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
YW = os.path.join(R, "work_dirs", "exp6_v3", "Y")
DW = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_D_h3_lookahead")
YX = os.path.join(R, "experiments", "exp6_v3", "Y")
DX = os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_D_h3_lookahead")
T1E = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
T1L = os.path.join(R, "work_dirs", "cvpr_results", "table1", "g3", "logs")
SPLIT = {"train": ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer"], "dev": ["barts", "colosseum", "eiffel", "pantheon", "sestino_museum"],
         "test": ["alhambra", "bridge", "neuschwanstein"]}


def now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def event(root, track, etype, **kw):
    with open(os.path.join(root, "EVENTS.jsonl"), "a") as f:
        f.write(json.dumps({"ts": now(), "track": track, "event": etype, "host": os.uname()[1].split(".")[0], **kw}, default=str) + "\n")


def jread(p, d):
    try:
        return json.load(open(p))
    except Exception:                         # noqa: BLE001
        return d


def jwrite(p, o):
    tmp = p + ".tmp"
    json.dump(o, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, p)


def procs(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if int(x) != os.getpid()]


def track_y():
    run = os.path.join(YW, "runs", "main")
    ctl = jread(os.path.join(YW, "control.json"), {"stop_at": 50000, "max_restarts": 2})
    st = jread(os.path.join(YW, "STATUS.json"), {})
    man_p = os.path.join(run, "checkpoint_manifest.json")
    man = jread(man_p, {"parent": {"ckpt_last_sha256": "3d81978206c88068c5dd5eda9cb221c8023948854549184b8aa35274c5941ce9", "step": 40000}, "steps": {}, "milestones": {}})
    logs = [json.loads(l) for l in open(os.path.join(run, "log.jsonl"))] if os.path.exists(os.path.join(run, "log.jsonl")) else []
    last = logs[-1] if logs else {}
    tlog = open(os.path.join(run, "train.log")).read().splitlines() if os.path.exists(os.path.join(run, "train.log")) else []
    exits = [l for l in tlog if l.startswith("EXIT ")]
    alive = procs("train_y.py --out .*Y/runs/main")
    log_age = time.time() - os.path.getmtime(os.path.join(run, "train.log")) if os.path.exists(os.path.join(run, "train.log")) else 1e9
    ck_step = None
    ck = os.path.join(run, "ckpt_last.pt")
    if os.path.exists(ck):
        ck_step = st.get("ckpt_step_cache", {}).get(str(os.path.getmtime(ck)))
        if ck_step is None:
            import torch
            ck_step = int(torch.load(ck, map_location="cpu", weights_only=False)["step"])
            st["ckpt_step_cache"] = {str(os.path.getmtime(ck)): ck_step}
    # rolling durable copies + manifest
    if ck_step and ck_step % 1000 == 0 and ck_step > 40000 and str(ck_step) not in man["steps"]:
        dst = os.path.join(run, f"ckpt_step{ck_step:06d}.pt")
        shutil.copyfile(ck, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        h = sha(dst)
        import torch
        ok = int(torch.load(dst, map_location="cpu", weights_only=False)["step"]) == ck_step
        man["steps"][str(ck_step)] = {"file": os.path.basename(dst), "sha256": h, "bytes": os.path.getsize(dst), "time": now(), "verified_step": ok}
        keep = sorted(int(s) for s in man["steps"] if os.path.exists(os.path.join(run, man["steps"][s]["file"])))[-3:]
        for s in list(man["steps"]):
            f = os.path.join(run, man["steps"][s]["file"])
            if int(s) not in keep and os.path.exists(f):
                os.remove(f)                      # rolling copies made by this controller only
                man["steps"][s]["rolled_out"] = now()
        event(YW, "Y", "checkpoint_saved", step=ck_step, sha256=h, verified=ok)
    for f in sorted(glob.glob(os.path.join(run, "ema_*.pt"))):
        s = os.path.basename(f)[4:10]
        if s not in man["milestones"]:
            man["milestones"][s] = {"file": os.path.basename(f), "sha256": sha(f), "time": now()}
            event(YW, "Y", "milestone_ema", step=int(s), sha256=man["milestones"][s]["sha256"])
    jwrite(man_p, man)
    phase = "RUNNING" if alive else ("COMPLETED" if exits and exits[-1].startswith("EXIT 0") else "STOPPED")
    stop_at = int(ctl.get("stop_at", 55000))
    if alive and ck_step is not None and ck_step >= stop_at and os.path.exists(os.path.join(run, f"ema_{stop_at:06d}.pt")):
        for pid in procs("torchrun.*train_y.py --out .*Y/runs/main"):
            try:
                os.killpg(os.getpgid(pid), 15)
            except Exception:                 # noqa: BLE001
                pass
        event(YW, "Y", "stopped_at_target", step=ck_step, stop_at=stop_at, reason="control.json stop_at (budget rule; resumable to 55000)")
        phase = "STOPPED_AT_TARGET"
        st["stopped_at_target"] = ck_step
    restarts = int(st.get("restarts", 0))
    if not alive and log_age > 300 and not st.get("stopped_at_target") and (ck_step or 0) < stop_at and tlog and not (exits and exits[-1].startswith("EXIT 0")):
        if restarts < int(ctl.get("max_restarts", 2)) and ctl.get("auto_restart", True):
            err = "\n".join(tlog[-15:])
            subprocess.Popen(["setsid", "nohup", "bash", os.path.join(YX, "scripts", "train_y.sh"), run], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                             cwd=R, env=dict(os.environ, Y_PORT=str(29611 + restarts + 1)))
            st["restarts"] = restarts + 1
            event(YW, "Y", "auto_restart", from_step=ck_step, attempt=restarts + 1, last_log=err[-1500:])
            phase = "RECOVERING"
        else:
            phase = "BLOCKED_FATAL"
    st.update({"run_id": "Y_20261104", "track": "Y", "phase": phase, "hostname": os.uname()[1].split(".")[0], "job_id": "3461041 (gpu114)",
               "parent_checkpoint_sha256": man["parent"]["ckpt_last_sha256"], "last_heartbeat_utc": now(), "steps_completed": last.get("step"),
               "ckpt_last_step": ck_step, "stop_at": stop_at, "target_steps": 55000, "lr": last.get("lr"), "loss": last.get("loss"), "fm": last.get("fm"),
               "rec": last.get("rec"), "kl": last.get("kl"), "grad_norm": last.get("grad_norm"), "s_per_step": last.get("s_per_step"), "mem_gb": last.get("mem_gb"),
               "eta_stop_at_h": (stop_at - (last.get("step") or 40000)) * (last.get("s_per_step") or 1.5) / 3600 if last else None,
               "milestones": sorted(man["milestones"]), "last_good_checkpoint": (max(man["steps"], key=int) if man["steps"] else "parent 40000"),
               "resume_command": f"bash experiments/exp6_v3/Y/scripts/train_y.sh {os.path.relpath(run, R)}"})
    jwrite(os.path.join(YW, "STATUS.json"), st)
    ev = [json.loads(l) for l in open(os.path.join(run, "eval.jsonl"))] if os.path.exists(os.path.join(run, "eval.jsonl")) else []
    md = [f"# STATUS — EXP6 V3 / Y（{now()}，controller 自动更新）", "",
          f"- **阶段**：{phase}",
          f"- **节点 / 作业**：{st['hostname']} / {st['job_id']}",
          f"- **已完成步数**：{st['steps_completed']}（ckpt_last 步数 {ck_step}；计划在 {stop_at} 停止，配置上限为 55000）",
          f"- **训练指标**：loss {st['loss']}，fm {st['fm']}，rec {st['rec']}，kl {st['kl']}，梯度范数 {st['grad_norm']}，学习率 {st['lr']}",
          f"- **速度与显存**：{st['s_per_step']} s/步，峰值显存 {st['mem_gb']} GB / rank；预计还需 {st['eta_stop_at_h'] and round(st['eta_stop_at_h'], 2)} h 到达 {stop_at} 步",
          f"- **里程碑 EMA**：{', '.join(st['milestones']) or '—'}；最近的滚动 checkpoint：{st['last_good_checkpoint']}；自动重启次数：{st.get('restarts', 0)}",
          f"- **续跑命令**：`{st['resume_command']}`", "",
          "| eval step | FM in_val | FM out_sc | fair ES in_val | fair ES out_sc |", "|---|---|---|---|---|"]
    for e in ev:
        md.append(f"| {e['step']} | {e.get('fm_in_val', 0):.4f} | {e.get('fm_out_sc', 0):.4f} | {e.get('samp_in_val', {}).get('fair_es_arclength_vox', float('nan')):.3f} | "
                  f"{e.get('samp_out_sc', {}).get('fair_es_arclength_vox', float('nan')):.3f} |")
    for p in (os.path.join(YW, "STATUS.md"), os.path.join(YX, "STATUS.md")):
        open(p, "w").write("\n".join(md) + "\n")


def track_d():
    ctl = jread(os.path.join(DW, "control.json"), {"jobfile": None, "workers": 2, "gpus": [0, 1, 2, 3], "max_restarts": 2})
    st = jread(os.path.join(DW, "STATUS.json"), {})
    models = ctl.get("models", ["V3-D1-H3-NORISK", "V3-D2-DEPTH3-B3", "V3-D3-DEPTH3-B5", "V3-D4-DEPTH4-B3", "V3-Y45K-H0", "V3-Y50K-H0", "V3-YBEST-H3"])
    cnt = {}
    for m in models:
        d = os.path.join(T1E, m)
        have = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(d, "*.json"))}
        cnt[m] = {g: sum(f"{s}_s{k}" in have for s in sc for k in range(5)) for g, sc in SPLIT.items()}
        cnt[m]["all14"] = sum(cnt[m][g] for g in SPLIT)
    prev = st.get("counts", {})
    for m in cnt:
        if cnt[m] != prev.get(m):
            event(DW, "D", "progress", model=m, counts=cnt[m])
    fails = []
    for f in glob.glob(os.path.join(T1L, "queue_*.log")):
        for l in open(f):
            if l.startswith("exit ") and not l.startswith("exit 0") and any(m + ":" in l for m in models):
                fails.append(l.strip())
    gave = [l.strip() for l in open(os.path.join(T1L, "gave_up.log"))] if os.path.exists(os.path.join(T1L, "gave_up.log")) else []
    gave = [l for l in gave if any(m + ":" in l for m in models)]
    if len(fails) > len(st.get("failures", [])):
        event(DW, "D", "episode_failures", new=fails[len(st.get("failures", [])):])
    jf = ctl.get("jobfile")
    alive = procs("t1_queue.sh [0-9] .*" + os.path.basename(jf)) if jf else []
    qlog_age = min([time.time() - os.path.getmtime(f) for f in glob.glob(os.path.join(T1L, "queue_*.log"))] or [1e9])
    remaining = None
    if jf and os.path.exists(os.path.join(R, jf)):
        rem = 0
        for line in open(os.path.join(R, jf)):
            line = line.strip()
            if line:
                m, s, k = line.split(":")
                rem += not os.path.exists(os.path.join(T1E, m, f"{s}_s{k}.json"))
        remaining = rem
    restarts = int(st.get("restarts", 0))
    phase = "RUNNING" if alive else ("COMPLETED" if remaining == 0 else "IDLE")
    if jf and not alive and qlog_age > 300 and remaining and ctl.get("auto_restart", True) and os.uname()[1].startswith(ctl.get("host", "gpu115")):
        if restarts < int(ctl.get("max_restarts", 2)):
            for g in ctl.get("gpus", [0, 1, 2, 3]):
                for _ in range(int(ctl.get("workers", 2))):
                    subprocess.Popen(["setsid", "nohup", "bash", os.path.join(R, "experiments", "cvpr_results", "table1", "scripts", "t1_queue.sh"), str(g), os.path.join(R, jf)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, cwd=R)
            st["restarts"] = restarts + 1
            event(DW, "D", "auto_restart_workers", remaining=remaining, attempt=restarts + 1)
            phase = "RECOVERING"
    st.update({"run_id": "D_20261104", "track": "D", "phase": phase, "hostname": os.uname()[1].split(".")[0], "job_id": ctl.get("job_id", "3443137 (gpu115)"),
               "model_sha256": "639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb", "last_heartbeat_utc": now(), "counts": cnt,
               "failures": fails, "gave_up": gave, "workers_alive_this_node": len(alive), "jobfile": jf, "jobs_remaining": remaining,
               "resume_command": f"bash experiments/cvpr_results/table1/scripts/t1_queue.sh <GPU> {jf}"})
    jwrite(os.path.join(DW, "STATUS.json"), st)
    md = [f"# STATUS — EXP6 V3 ablation D（{now()}，controller 自动更新）", "",
          f"- **阶段**：{phase}",
          f"- **作业**：{st['job_id']}；本节点存活的 worker：{len(alive)}；任务文件 `{jf}` 剩余 {remaining} 个任务",
          f"- **失败尝试**：{len(fails)}；放弃：{len(gave)}（队列会自动重试最多 3 次）；自动重启次数：{st.get('restarts', 0)}",
          f"- **续跑命令**：`{st['resume_command']}`", "",
          "| 模型 | Train-6 | Dev-5 | Test-3 | All-14 |", "|---|---|---|---|---|"]
    for m, c in cnt.items():
        md.append(f"| {m} | {c['train']}/30 | {c['dev']}/25 | {c['test']}/15 | {c['all14']}/70 |")
    for p in (os.path.join(DW, "STATUS.md"), os.path.join(DX, "STATUS.md")):
        open(p, "w").write("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["Y", "D"], required=True)
    a = ap.parse_args()
    os.makedirs(YW if a.track == "Y" else DW, exist_ok=True)
    try:
        track_y() if a.track == "Y" else track_d()
    except Exception as e:                    # noqa: BLE001
        import traceback
        event(YW if a.track == "Y" else DW, a.track, "controller_error", error=repr(e), tb=traceback.format_exc()[-2000:])


if __name__ == "__main__":
    main()
