"""EXP5 Task 1.4 -- closed-loop orchestrator (stdlib only; one node, protocol §5.4).

  smoke        12-move smoke of E1 (barts s0, pantheon s0, rep 0) + resume test (barts s0 interrupted after t = 10)
  smoke-check  legality, guard, common-evaluator replays, resume bitwise equality, memory -> smoke/smoke_check.json
  run          queue of 100-step episodes of the given arms (order RUN_ORDER x scene x start x rep) + post jobs
               (common-evaluator replay + oracle diagnosis) on --gpus with --per-gpu episode slots per GPU;
               an episode that exits non-zero is resumed from its checkpoint (up to 3 attempts); INVALID / LeakError
               in its log -> STOP marker for that arm (its remaining episodes are skipped), other arms continue
  status       closed_loop/STATUS.json
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as LC  # noqa: E402

T6SIM = os.path.join(LC.EXP3B, "m_t6_sim.py")
ODIAG = os.path.join(LC.T3_SCRIPTS, "t3_loop_oracle_diag.py")
STOP_MARKERS = ("INVALID", "LeakError", "guard: forbidden")


def jdump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f".{os.getpid()}.tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, p)


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def ep_job(eid, s, k, rep, root, moves, interrupt=None, suffix=""):
    return {"kind": "episode", "name": LC.tag(eid, s, k, rep) + suffix, "exp": eid, "scene": s, "start": k, "rep": rep,
            "root": root, "moves": moves, "interrupt": interrupt}


def post_job(eid, s, k, rep, root):
    return {"kind": "post", "name": "post_" + LC.tag(eid, s, k, rep), "exp": eid, "scene": s, "start": k, "rep": rep,
            "root": root}


def episode_path(job):
    return LC.dirs(job["root"], LC.tag(job["exp"], job["scene"], job["start"], job["rep"]))["episode"]


def post_outputs(job):
    tg = LC.tag(job["exp"], job["scene"], job["start"], job["rep"])
    return os.path.join(job["root"], "replay", f"rep_{tg}.json"), os.path.join(job["root"], "oracle_diag", f"diag_{tg}.json")


def job_done(job):
    if job["kind"] == "episode":
        return os.path.exists(episode_path(job))
    return all(os.path.exists(x) for x in post_outputs(job))


def cmd_of(job):
    if job["kind"] == "episode":
        return [LC.MAGICIAN_PYTHON, os.path.join(HERE, "v2_sim.py"), "--exp", job["exp"], "--scene", job["scene"],
                "--start", str(job["start"]), "--rep", str(job["rep"]), "--root", job["root"], "--moves", str(job["moves"])]
    tg = LC.tag(job["exp"], job["scene"], job["start"], job["rep"])
    D = LC.dirs(job["root"], tg)
    rep, diag = post_outputs(job)
    for d in (os.path.dirname(rep), os.path.dirname(diag)):
        os.makedirs(d, exist_ok=True)
    parts = []
    if not os.path.exists(rep):
        parts.append(f"{LC.MAGICIAN_PYTHON} {T6SIM} replay --episode {D['episode']} --out {rep}")
    if not os.path.exists(diag):
        parts.append(f"{LC.MAGICIAN_PYTHON} {ODIAG} --episode {D['episode']} --planner-log {D['plog']}/pl_{tg}.json --out {diag}")
    parts.append(f"rm -rf {os.path.dirname(rep)}/replay_frames/rep_{tg} {os.path.dirname(diag)}/frames_diag_{tg}")
    return ["bash", "-c", " && ".join(parts)]


def launch(job, gpu, logs, attempt):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES=str(gpu))
    env.pop("ABL_INTERRUPT_AFTER", None)
    if job.get("interrupt") is not None and attempt == 0:
        env["ABL_INTERRUPT_AFTER"] = str(job["interrupt"])
    os.makedirs(logs, exist_ok=True)
    lp = os.path.join(logs, f"{job['name']}.a{attempt}.log")
    return subprocess.Popen(cmd_of(job), cwd=LC.REPO, env=env, stdout=open(lp, "w"), stderr=subprocess.STDOUT), lp


def run_queue(jobs, gpus, per_gpu, logs, acct_dir, max_attempts=3):
    """Episodes first-come over slots; the post job of an episode is queued when it finishes."""
    pending = [j for j in jobs if not job_done(j)]
    for j in jobs:
        if j["kind"] == "episode" and job_done(j) and not job_done(post_job(j["exp"], j["scene"], j["start"], j["rep"], j["root"])):
            pending.append(post_job(j["exp"], j["scene"], j["start"], j["rep"], j["root"]))
    slots = {(g, i): None for g in gpus for i in range(per_gpu)}
    attempts = {}
    stopped = set()
    while pending or any(v is not None for v in slots.values()):
        for sl, cur in list(slots.items()):
            if cur is None:
                continue
            p, job, lp, t0 = cur
            rc = p.poll()
            if rc is None:
                continue
            slots[sl] = None
            a = attempts.get(job["name"], 0)
            acc = load(os.path.join(acct_dir, f"{job['name']}.json")) or {"attempts": []}
            acc["attempts"].append({"attempt": a, "rc": rc, "start": t0, "end": time.time(), "gpu": sl[0], "log": lp})
            jdump(acc, os.path.join(acct_dir, f"{job['name']}.json"))
            attempts[job["name"]] = a + 1
            txt = open(lp).read()[-20000:]
            if any(m in txt for m in STOP_MARKERS):
                stopped.add(job["exp"])
                open(os.path.join(acct_dir, f"STOP_{job['exp']}"), "a").write(f"{job['name']} {lp}\n")
                print(f"STOP {job['exp']}: {job['name']} ({lp})", flush=True)
                continue
            if job_done(job):
                print(f"done {job['name']} rc={rc} {time.time() - t0:.0f}s", flush=True)
                if job["kind"] == "episode" and not job.get("interrupt"):
                    pending.append(post_job(job["exp"], job["scene"], job["start"], job["rep"], job["root"]))
            elif rc == LC.EXIT_INTERRUPT or attempts[job["name"]] < max_attempts:
                pending.insert(0, job)                                     # resume from the checkpoint
            else:
                print(f"GIVEUP {job['name']} after {attempts[job['name']]} attempts", flush=True)
                open(os.path.join(acct_dir, "GIVEUP"), "a").write(f"{job['name']}\n")
        for sl in slots:
            if slots[sl] is None and pending:
                job = pending.pop(0)
                if job["exp"] in stopped or job_done(job):
                    continue
                p, lp = launch(job, sl[0], logs, attempts.get(job["name"], 0))
                slots[sl] = (p, job, lp, time.time())
        time.sleep(5)
    return stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "smoke-check", "run", "status"])
    ap.add_argument("--exps", default="E1")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--per-gpu", type=int, default=2)
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    if a.mode == "smoke":
        root = LC.exp_root("E1", smoke=True)
        jobs = [ep_job("E1", s, k, 0, root, LC.SMOKE_MOVES) for s, k in LC.SMOKE_EPISODES]
        jobs.append(ep_job("E1", "barts", 0, 0, root + "_resume", LC.SMOKE_MOVES, interrupt=LC.SMOKE_INTERRUPT_AFTER,
                           suffix="_resume"))
        run_queue(jobs, gpus, 1, os.path.join(root, "logs"), os.path.join(root, "acct"))
        smoke_check(gpus)
    elif a.mode == "smoke-check":
        smoke_check(gpus)
    elif a.mode == "run":
        exps = a.exps.split(",")
        jobs = [ep_job(e, s, k, r, LC.exp_root(e), LC.MOVES) for e in exps for s in LC.SCENES for k in LC.STARTS
                for r in LC.REPS]
        stopped = run_queue(jobs, gpus, a.per_gpu, os.path.join(LC.LOOP, "logs"), os.path.join(LC.LOOP, "acct"))
        print("finished; stopped arms:", sorted(stopped), flush=True)
    else:
        status()


def smoke_check(gpus):
    root = LC.exp_root("E1", smoke=True)
    out, ok = {"time": time.ctime(), "episodes": {}}, []
    for s, k in LC.SMOKE_EPISODES:
        tg = LC.tag("E1", s, k, 0)
        e, pl = load(LC.dirs(root, tg)["episode"]), load(os.path.join(root, "planner_logs", f"pl_{tg}.json"))
        b0 = load(os.path.join(LC.B0_ROOT, "episodes", f"ep_A2_{s}_s{k}_r0.json"))
        r = {"present": e is not None and pl is not None}
        if r["present"]:
            r.update({"cov0_equal_B0": e["cov0"] == b0["cov0"], "guard_denied": pl["guard"]["denied"],
                      "picks_legal": all(x["pick"] is None or 0 <= x["pick"] < x["n_legal"] for x in e["steps"][:-1]),
                      "refresh_ts": [x["t"] for x in pl["refreshes"]],
                      "fallbacks": sum(1 for x in pl["steps"] if x["choice"].get("fallback")),
                      "decision_s_mean": sum(x["decision_s"] for x in pl["steps"]) / len(pl["steps"]),
                      "peak_mem_gb": pl["peak_mem_gb_this_process"], "cov12": e["coverage"][-1],
                      "cov12_B0": b0["coverage"][LC.SMOKE_MOVES]})
            r["pass"] = (r["cov0_equal_B0"] and not r["guard_denied"] and r["picks_legal"]
                         and r["refresh_ts"] == [x for x in LC.R10 if x < LC.SMOKE_MOVES])
        else:
            r["pass"] = False
        out["episodes"][tg] = r
        ok.append(r["pass"])
    procs, reps = [], []
    for i, (s, k) in enumerate(LC.SMOKE_EPISODES):
        tg = LC.tag("E1", s, k, 0)
        o = os.path.join(root, "replay", f"rep_{tg}.json")
        os.makedirs(os.path.dirname(o), exist_ok=True)
        reps.append(o)
        if not os.path.exists(o):
            procs.append(subprocess.Popen([LC.MAGICIAN_PYTHON, T6SIM, "replay", "--episode", LC.dirs(root, tg)["episode"],
                                           "--out", o], cwd=LC.REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[i % len(gpus)])),
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    shutil.rmtree(os.path.join(root, "replay", "replay_frames"), ignore_errors=True)
    rr = [load(x) for x in reps]
    out["replay_pass"] = all(x and x["bitwise_equal"] and x["all_legal"] for x in rr)
    ok.append(out["replay_pass"])
    tg = LC.tag("E1", "barts", 0, 0)
    er, eu = load(LC.dirs(root + "_resume", tg)["episode"]), load(LC.dirs(root, tg)["episode"])
    plr, plu = load(os.path.join(root + "_resume", "planner_logs", f"pl_{tg}.json")), load(os.path.join(root, "planner_logs", f"pl_{tg}.json"))
    acct = load(os.path.join(root + "_resume", "acct", f"{tg}_resume.json")) or load(os.path.join(root, "acct", f"{tg}_resume.json"))
    r = {"present": er is not None and eu is not None and plr is not None}
    if r["present"]:
        r.update({"attempt_rcs": [x["rc"] for x in (acct or {}).get("attempts", [])],
                  "resumed_at": (er.get("resume") or {}).get("T"),
                  "sim_replay_bitwise_equal": (er.get("resume") or {}).get("replay_bitwise_equal"),
                  "coverage_equal": er["coverage"] == eu["coverage"], "keys_equal": er["executed_keys"] == eu["executed_keys"],
                  "scores_equal": [x["planner"]["scores"] for x in er["steps"][:-1]] == [x["planner"]["scores"] for x in eu["steps"][:-1]],
                  "all_scores_equal": [x["all_scores"] for x in plr["steps"]] == [x["all_scores"] for x in plu["steps"]]})
        r["pass"] = (r["attempt_rcs"] == [LC.EXIT_INTERRUPT, 0] and r["resumed_at"] == LC.SMOKE_INTERRUPT_AFTER + 1
                     and r["sim_replay_bitwise_equal"] and r["coverage_equal"] and r["keys_equal"] and r["scores_equal"]
                     and r["all_scores_equal"])
    else:
        r["pass"] = False
    out["resume_test"] = r
    ok.append(r["pass"])
    out["pass"] = all(ok)
    jdump(out, os.path.join(root, "smoke_check.json"))
    print(json.dumps({"pass": out["pass"], "episodes": {k: v["pass"] for k, v in out["episodes"].items()},
                      "replay": out["replay_pass"], "resume": r["pass"]}), flush=True)
    return out["pass"]


def status():
    rows = {}
    for e in LC.RUN_ORDER:
        root = LC.exp_root(e)
        eps = glob.glob(os.path.join(root, "episodes", "ep_*.json"))
        diag = glob.glob(os.path.join(root, "oracle_diag", "diag_*.json"))
        rows[e] = {"name": LC.ARMS[e]["name"], "episodes_done": len(eps), "post_done": len(diag),
                   "of": len(LC.SCENES) * len(LC.STARTS) * len(LC.REPS),
                   "stopped": os.path.exists(os.path.join(LC.LOOP, "acct", f"STOP_{e}"))}
    jdump({"time": time.ctime(), "arms": rows}, os.path.join(LC.LOOP, "STATUS.json"))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
