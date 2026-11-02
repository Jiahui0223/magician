"""EXP6 V5.2-M Scratch70k -- unattended driver (stdlib only). Runs on the head node of an allocation (setsid / nohup inside
the interactive allocation, or inside slurm/continue_*.sbatch) and executes the frozen chain of FROZEN_PROTOCOL.md §6:

  1. training rounds until 70,000 synchronized optimizer steps (train_v52m.py resumes from checkpoints/last.pt):
     2-node 8-rank DDP whenever the node-B worker job (nodeB_loop.sh, its own sbatch job) is alive and idle, else 1-node
     4-rank DDP; a running 1-node round is stopped at a checkpoint (stop file -> save -> exit 85) to expand to 2 nodes;
     a lost node B -> the next round is 1-node; per-round deadline = earliest Slurm end of the used jobs - 10 min.
  2. read-only milestone G2 at 20k / 40k (step_XXXXXX.pt, EMA) on the head node, concurrently with training.
  3. after 70k: final G1 / G2 of last70k (= step_070000.pt) and best_train_holdout.pt (head node + node B),
     frozen gate (§5) on the primary checkpoint best_train_holdout, G3 smoke (always), G3 full (gate + valid smoke).
  4. reports (make_reports_v52m.py), RUN_STATUS.json / AUTO_DECISION.json, STATUS.md block, git commit + push of the
     experiment folder only, node-B release (ctl/SHUTDOWN).
Never fabricates: every stage writes its own status; a failed stage is reported as failed and the chain continues where
it can. NaN / OOM-exhausted training -> state "blocked", evaluations of existing checkpoints are not started.
Usage: python overnight_driver.py --head-job JOBID [--phase all|final|reports]"""
import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
E = os.path.join(R, "experiments", "exp6_v5_2_scratch70k")
SC = os.path.join(E, "scripts")
W = os.path.join(R, "work_dirs", "exp6_v5_2_scratch70k")
CTL, CK, LOG, EV = (os.path.join(W, x) for x in ("ctl", "checkpoints", "logs", "eval"))
S2E = "/bsuscratch/jiahuizhang/envs/seen2scene"
PY, TORCHRUN = S2E + "/bin/python", S2E + "/bin/torchrun"
STEPS, MILESTONES = 70000, (20000, 40000)
ENV = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=R,
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="6")
STATUS_P, DECISION_P = os.path.join(E, "RUN_STATUS.json"), os.path.join(E, "AUTO_DECISION.json")
GATE = {"outdoor_top1_min": 0.46, "mp3d_top1_min": 0.40}


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def jload(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


def jsave(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(o, open(p + ".tmp", "w"), indent=1, default=str)
    os.replace(p + ".tmp", p)


def sh(cmd, timeout=120):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def job_state(job):
    return sh(f"squeue -h -j {job} -o %T") if job else ""


def job_end(job):
    try:
        return time.mktime(time.strptime(sh(f"squeue -h -j {job} -o %e"), "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return None


def master_addr():
    for line in sh("ip -4 addr show ib0").splitlines():
        if "inet " in line:
            return line.split()[1].split("/")[0]
    return sh("hostname -i").split()[0]


class Driver:
    def __init__(self, a):
        self.a = a
        self.st = jload(STATUS_P, None) or {"experiment": "EXP6 V5.2-M Scratch70k", "state": "planned", "rounds": [], "milestones": {}, "final": {}, "g3": {}}
        self.st.setdefault("events", [])
        self.st["head"] = {"host": sh("hostname -s"), "job": a.head_job, "driver_pid": os.getpid()}
        self.st["git_sha_launch"] = sh(f"git -C {R} rev-parse HEAD")
        self.milestone_procs = {}
        for k, v in self.st["milestones"].items():                 # restarted driver: relaunch unfinished milestone evals (finished parts are skipped)
            if v.get("state") == "running" and not os.path.exists(os.path.join(v.get("out", ""), "DONE")):
                v["state"] = "pending"
        self.save()

    # ------------------------------------------------------------ status
    def save(self, **kw):
        self.st.update(kw)
        self.st["updated"] = time.strftime("%F %T")
        lm = jload(os.path.join(CK, "last_manifest.json"), {})
        self.st["steps_completed"] = lm.get("step", 0)
        jsave(STATUS_P, self.st)

    def event(self, msg):
        log(msg)
        self.st["events"].append(f"{time.strftime('%F %T')} {msg}")
        self.save()

    def step_done(self):
        return jload(os.path.join(CK, "last_manifest.json"), {}).get("step", 0)

    def nodeb(self):
        s = jload(os.path.join(CTL, "nodeB_status.json"), {})
        if not s or job_state(s.get("job")) != "RUNNING":
            return None
        return s

    # ------------------------------------------------------------ training
    def train(self):
        fails, last_fail_step = 0, -1
        while self.step_done() < STEPS:
            nb = self.nodeb()
            two = nb is not None and nb.get("state") == "idle" and time.time() - nb.get("unix", 0) < 120
            ends = [e for e in (job_end(self.a.head_job), job_end(nb["job"]) if two else None) if e]
            deadline = (min(ends) - 600) if ends else 0
            if deadline and time.time() > deadline - 900:
                self.event(f"not enough allocation time for another round (deadline {time.ctime(deadline)}); stopping at step {self.step_done()}")
                self.save(state="checkpointed")
                return False
            k = len(self.st["rounds"])
            rid, port = f"r{k:03d}_{int(time.time())}", 29600 + (k % 300)
            stop = os.path.join(CTL, "STOP_TRAIN")
            if os.path.exists(stop):
                os.remove(stop)
            args = [os.path.join(SC, "train_v52m.py"), "--out", W, "--stop-file", stop, "--deadline", f"{deadline:.0f}"]
            s0 = self.step_done()
            rd = {"round": rid, "nnodes": 2 if two else 1, "start": time.strftime("%F %T"), "step_before": s0, "deadline": time.ctime(deadline) if deadline else None,
                  "jobs": [self.a.head_job] + ([nb["job"]] if two else []), "hosts": [self.st["head"]["host"]] + ([nb["host"]] if two else [])}
            self.st["rounds"].append(rd)
            self.save(state="running", phase="train")
            lf = open(os.path.join(LOG, f"train_head_{rid}.log"), "a")
            if two:
                ma = master_addr()
                jsave(os.path.join(CTL, "round.json"), {"round": rid, "nnodes": 2, "master_addr": ma, "master_port": port, "args": args})
                cmd = [TORCHRUN, "--nnodes", "2", "--node_rank", "0", "--nproc_per_node", "4", "--master_addr", ma, "--master_port", str(port)] + args
            else:
                cmd = [TORCHRUN, "--standalone", "--nproc_per_node", "4"] + args
            env = dict(ENV, CUDA_VISIBLE_DEVICES="0,1,2,3")
            self.event(f"round {rid}: {'2-node 8-rank' if two else '1-node 4-rank'} from step {s0}, deadline {rd['deadline']}")
            p = subprocess.Popen(cmd, cwd=R, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            stop_sent, b_exit_seen, lost_n = False, None, 0
            while p.poll() is None:
                time.sleep(30)
                self.check_milestones()
                if not two and not stop_sent and self.step_done() < STEPS - 4000:
                    nb2 = self.nodeb()
                    if nb2 and nb2.get("state") == "idle" and time.time() - nb2.get("unix", 0) < 120 and (job_end(nb2["job"]) or 0) - time.time() > 3 * 3600:
                        self.event(f"node B {nb2['host']} (job {nb2['job']}) available: stopping round {rid} at a checkpoint to expand to 8 ranks")
                        open(stop, "w").close()
                        stop_sent = True
                if two:
                    ef = os.path.join(CTL, f"nodeB_round{rid}.exit")
                    js = job_state(rd["jobs"][1])
                    lost_n = 0 if js == "RUNNING" else lost_n + 1               # 4 consecutive observations (2 min) against squeue hiccups
                    if os.path.exists(ef) and b_exit_seen is None:
                        b_exit_seen = time.time()
                    if lost_n >= 4 or (b_exit_seen and time.time() - b_exit_seen > 300):
                        self.event(f"round {rid}: node B lost (job state '{job_state(rd['jobs'][1])}', exit file {os.path.exists(ef)}); killing the head ranks")
                        os.killpg(p.pid, 15)
                        time.sleep(30)
                        if p.poll() is None:
                            os.killpg(p.pid, 9)
                if os.path.exists(os.path.join(CTL, "STOP_DRIVER")):
                    open(stop, "w").close()
            ec = p.returncode
            rd.update({"end": time.strftime("%F %T"), "exit": ec, "step_after": self.step_done()})
            if two:
                rd["nodeB_exit"] = (open(os.path.join(CTL, f"nodeB_round{rid}.exit")).read().strip() if os.path.exists(os.path.join(CTL, f"nodeB_round{rid}.exit")) else None)
            self.save()
            self.event(f"round {rid} exit {ec}, step {s0} -> {rd['step_after']}")
            if ec == 87:
                self.save(state="blocked", blocker=f"non-finite loss (dump in {W}); see logs/train_head_{rid}.log")
                return False
            if ec == 86:
                self.save(state="blocked", blocker="CUDA OOM after the 2 frozen fallbacks")
                return False
            if os.path.exists(os.path.join(CTL, "STOP_DRIVER")):
                self.save(state="checkpointed", blocker="stopped by ctl/STOP_DRIVER")
                return False
            if ec not in (0, 85):
                if rd["step_after"] > last_fail_step:
                    fails, last_fail_step = 0, rd["step_after"]
                fails += 1
                if fails > 3:
                    self.save(state="blocked", blocker=f"4 consecutive failed rounds without progress at step {rd['step_after']} (last exit {ec})")
                    return False
                time.sleep(60)
            elif ec == 85 and not stop_sent and deadline and time.time() > deadline - 900:
                self.save(state="checkpointed")
                return False
        self.check_milestones()
        self.save(phase="train_done")
        return True

    # ------------------------------------------------------------ evaluations
    def eval_script(self, name, ck, gpus, sets):
        """Shell script: S-c floors of ck -> fixed-state G1/G2 (parts over the given GPUs) -> analysis."""
        od = os.path.join(EV, name)
        os.makedirs(od, exist_ok=True)
        cfg = os.path.join(od, "score_v52m.json")
        env = f"cd {R}; export LD_PRELOAD={S2E}/lib/libtbb.so LD_LIBRARY_PATH={S2E}/lib PYTHONPATH={R} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4"
        L = ["#!/bin/bash", env, f"echo \"$(date '+%F %T') eval {name} on $(hostname -s) GPUs {gpus}\"",
             f"[ -f {cfg} ] || CUDA_VISIBLE_DEVICES={gpus[0]} {PY} {SC}/calibrate_floors_v52m.py --ckpt {ck} --out-cfg {cfg} --out-json {od}/calibration_floors.json > {od}/calibrate.log 2>&1"]
        jobs = []
        for s, st, n in sets:
            for i in range(n):
                jobs.append(f"--set {s} --stratum {st} --part {i}/{n}")
        for gi, g in enumerate(gpus):
            mine = jobs[gi::len(gpus)]
            if mine:
                L.append("( " + "; ".join(f"CUDA_VISIBLE_DEVICES={g} {PY} {SC}/eval_fixed_v52m.py {j} --ckpt {ck} --score-cfg {cfg} --out {od}/fixed >> {od}/fixed_gpu{g}.log 2>&1" for j in mine) + " ) &")
        L += ["wait", f"{PY} {SC}/analyze_g12_v52m.py --rows {od}/fixed --name {name} --out-json {od}/g12.json --out-md {od}/g12.md > {od}/analyze.log 2>&1",
              f"echo \"$(date '+%F %T') eval {name} done exit $?\"", f"touch {od}/DONE"]
        p = os.path.join(od, "run_eval.sh")
        open(p, "w").write("\n".join(L) + "\n")
        return p, od

    SETS_HEAD = (("valsel", "B0", 3), ("dev3", "B0", 3), ("dev3", "B2", 2))   # 8 parts on 4 GPUs (2 per GPU, sequential)

    def check_milestones(self):
        for m in MILESTONES:
            ck = os.path.join(CK, f"step_{m:06d}.pt")
            key = str(m)
            ms = self.st["milestones"].setdefault(key, {"state": "pending"})
            if ms["state"] == "pending" and os.path.exists(ck):
                p, od = self.eval_script(f"milestone_{m // 1000}k", ck, [0, 1, 2, 3], self.SETS_HEAD)
                self.milestone_procs[key] = subprocess.Popen(["bash", p], cwd=R, stdout=open(os.path.join(od, "run.log"), "a"), stderr=subprocess.STDOUT, start_new_session=True)
                ms.update({"state": "running", "start": time.strftime("%F %T"), "ckpt": ck, "out": od})
                self.event(f"milestone G2 {m // 1000}k started (read-only, head node, concurrent with training)")
            elif ms["state"] == "running":
                pr = self.milestone_procs.get(key)
                if pr is not None and pr.poll() is not None or pr is None and os.path.exists(os.path.join(ms["out"], "DONE")):
                    ok = os.path.exists(os.path.join(ms["out"], "g12.json"))
                    ms.update({"state": "done" if ok else "failed", "end": time.strftime("%F %T")})
                    self.event(f"milestone G2 {m // 1000}k {ms['state']}")
                    self.reports()
        self.save()

    def wait_milestones(self):
        while any(v["state"] == "running" for v in self.st["milestones"].values()):
            time.sleep(60)
            self.check_milestones()

    def run_parallel(self, local_cmd, remote_cmd, tag, timeout_h=12):
        """local_cmd on the head node and remote_cmd on node B (if alive); wait for both."""
        procs = []
        if local_cmd:
            procs.append(subprocess.Popen(["bash", "-c", local_cmd], cwd=R, env=ENV, stdout=open(os.path.join(LOG, f"{tag}_head.log"), "a"), stderr=subprocess.STDOUT, start_new_session=True))
        nb = self.nodeb()
        done_f = None
        if remote_cmd and nb:
            eid = f"{tag}_{int(time.time())}"
            jsave(os.path.join(CTL, "eval_task.json"), {"id": eid, "cmd": remote_cmd})
            done_f = os.path.join(CTL, f"nodeB_eval_{eid}.done")
            self.event(f"{tag}: node B {nb['host']} task {eid}")
        elif remote_cmd:
            self.event(f"{tag}: node B not available, remote part runs on the head node after the local part")
        t0 = time.time()
        while time.time() - t0 < timeout_h * 3600:
            local_ok = all(p.poll() is not None for p in procs)
            remote_ok = done_f is None or os.path.exists(done_f) or (self.nodeb() is None)
            if local_ok and remote_ok:
                break
            time.sleep(60)
        if remote_cmd and done_f is None:
            subprocess.run(["bash", "-c", remote_cmd], cwd=R, env=ENV, stdout=open(os.path.join(LOG, f"{tag}_head_remote_part.log"), "a"), stderr=subprocess.STDOUT)
        elif done_f is not None and not os.path.exists(done_f):
            self.event(f"{tag}: node B part did not finish (node lost?); rerunning it on the head node (claims / done files skip finished items)")
            subprocess.run(["bash", "-c", remote_cmd], cwd=R, env=ENV, stdout=open(os.path.join(LOG, f"{tag}_head_remote_part.log"), "a"), stderr=subprocess.STDOUT)

    def final(self):
        self.wait_milestones()
        fin = self.st["final"]
        last = os.path.join(CK, "step_070000.pt")
        best = os.path.join(CK, "best_train_holdout.pt")
        bj = jload(os.path.join(CK, "best_train_holdout.json"), {})
        fin.update({"last70k": last, "best_train_holdout": best, "best_step": bj.get("step"), "best": bj})
        for p in (last, best):
            if os.path.exists(p):
                fin.setdefault("sha256", {})[os.path.basename(p)] = sh(f"sha256sum {p}", 600).split()[0]
        same = bj.get("step") == STEPS
        fin["best_is_last"] = same
        self.save(phase="final_g2")
        # G2 / G1 of the two checkpoints: last70k on the head node, best on node B (or both on the head node)
        p1, od1 = self.eval_script("final_last70k", last, [0, 1, 2, 3], self.SETS_HEAD)
        cmds = [f"bash {p1}"]
        rem = None
        if not same:
            p2, od2 = self.eval_script("final_best", best, [0, 1, 2, 3], self.SETS_HEAD)
            rem = f"bash {p2}"
        self.event(f"final G2: last70k{'' if same else ' + best_train_holdout (step %s)' % bj.get('step')}")
        self.run_parallel(" ; ".join(cmds), rem, "final_g2")
        prim = "final_last70k" if same else "final_best"
        fin["primary"] = {"name": prim, "ckpt": last if same else best, "step": STEPS if same else bj.get("step")}
        self.save()
        self.reports()
        # frozen gate (§5)
        g = jload(os.path.join(EV, prim, "g12.json"))
        dec = self.gate(g)
        jsave(DECISION_P, dec)
        self.event(f"gate: {dec['g2_gate']} -> {dec['decision']}")
        # G3 smoke (always)
        ck, cfg = fin["primary"]["ckpt"], os.path.join(EV, prim, "score_v52m.json")
        self.g3("smoke", ck, cfg, os.path.join(W, "g3_smoke_jobs.txt"), os.path.join(W, "g3_smoke_cov.txt"), workers=2)
        self.reports()
        sm = jload(os.path.join(E, "reports", "g3_smoke_v52m.json"), {})
        dec["g3_smoke_valid"] = bool(sm.get("valid"))
        if dec["g2_gate"] == "PASS" and dec["g3_smoke_valid"]:
            dec["decision"] = "G3 smoke + G3 full"
            jsave(DECISION_P, dec)
            self.g3("full", ck, cfg, os.path.join(W, "g3_full_jobs.txt"), os.path.join(W, "g3_full_cov.txt"), workers=2)
        else:
            dec["decision"] = "G3 smoke only" + ("" if dec["g2_gate"] == "PASS" else " (G2 gate not met)") + ("" if dec["g3_smoke_valid"] else " (smoke invalid)")
            self.st["g3"]["full"] = {"state": "skipped", "reason": dec["decision"]}
        jsave(DECISION_P, dec)
        self.save(phase="reports")
        self.reports()

    def gate(self, g):
        d = {"rule": "FROZEN_PROTOCOL §5: G3 full iff Outdoor Dev-3 G2 Top-1 (λ=0, 275 states) ≥ 0.46 AND MP3D VALSEL G2 Top-1 (λ=0, 90 states) ≥ 0.40 AND "
                     "UNKNOWN-ray high-confidence false no-hit share (rays with p_nohit in [0.9, 1] that truly hit) below old V5.2 in both domains; AND the G3 smoke is valid",
             "thresholds": GATE, "time": time.strftime("%F %T")}
        if not g or "summary" not in g:
            d.update({"g2_gate": "NO-DATA", "decision": "G3 smoke only (G2 missing)"})
            return d
        S = g["summary"]
        t_out = S.get("dev3", {}).get("all", {}).get("means", {}).get("V52ML0_top1", {}).get("mean")
        t_in = S.get("valsel", {}).get("all", {}).get("means", {}).get("V52ML0_top1", {}).get("mean")
        fn = {}
        for dset in ("dev3", "valsel"):
            new = g["g1_new"].get(dset, {}).get("unknown", {})
            old = g["g1_old_v52"].get(dset, {}).get("unknown", {})
            nf = None if new.get("bin_09_1_rays", 0) == 0 else 1 - new["bin_09_1_actual_nohit"]
            of = None if not old.get("bin_09_1_rays") else 1 - old["bin_09_1_actual_nohit"]
            fn[dset] = {"new_false_nohit_share_p09": nf, "new_rays_p09": new.get("bin_09_1_rays"), "old_false_nohit_share_p09": of, "old_rays_p09": old.get("bin_09_1_rays"),
                        "new_hc_false_nohit_rate_p095": new.get("hc_false_nohit_rate"), "improved": (of is not None) and (nf is None or nf < of)}
        ok = (t_out is not None and t_out >= GATE["outdoor_top1_min"]) and (t_in is not None and t_in >= GATE["mp3d_top1_min"]) and all(v["improved"] for v in fn.values())
        d.update({"outdoor_top1_l0": t_out, "mp3d_top1_l0": t_in, "false_nohit": fn, "g2_gate": "PASS" if ok else "FAIL",
                  "decision": "G3 smoke, then full if the smoke is valid" if ok else "G3 smoke only"})
        return d

    def g3(self, kind, ck, cfg, jobs, cov, workers=2):
        root = os.path.join(EV, "g3")
        self.st["g3"][kind] = {"state": "running", "start": time.strftime("%F %T"), "ckpt": ck, "score_cfg": cfg}
        self.save(phase=f"g3_{kind}")
        self.event(f"G3 {kind} started ({workers} workers / GPU, head + node B)")
        q = os.path.join(SC, "run_queue_v52m.sh")
        body = (f"export V52M_CKPT={ck} V52M_SCORE_CFG={cfg} G3_ROOT={root}; "
                f"for g in 0 1 2 3; do for i in $(seq {workers}); do bash {q} $g {jobs} & sleep 2; done; done; wait; "
                f"for g in 0 1 2 3; do for i in $(seq {workers}); do bash {q} $g {cov} & sleep 2; done; done; wait")
        self.run_parallel(body, body, f"g3_{kind}", timeout_h=10)
        # a second local pass picks up items whose claim holder died (stale claims are cleared only for missing outputs)
        self.st["g3"][kind].update({"state": "done", "end": time.strftime("%F %T")})
        self.save()

    # ------------------------------------------------------------ reports
    def reports(self):
        r = subprocess.run([PY, os.path.join(SC, "make_reports_v52m.py")], cwd=R, env=ENV, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            log("make_reports failed:", r.stderr[-2000:])
        return r.returncode == 0

    def finish(self):
        lm = self.step_done()
        state = self.st.get("state")
        if state not in ("blocked", "checkpointed"):
            state = "completed" if lm >= STEPS and self.st["g3"].get("smoke", {}).get("state") == "done" else "partial"
        self.save(state=state, phase="done", finished=time.strftime("%F %T"))
        self.reports()
        msg = f"[V5.2-M][report] Scratch70k unattended chain: state {state}, steps {lm}"
        out = sh(f"cd {R} && git add experiments/exp6_v5_2_scratch70k && git commit -q -m {shlex.quote(msg)} && git push -q origin seen2scene 2>&1; git rev-parse HEAD", 600)
        self.st["git_sha_reports"] = out.splitlines()[-1] if out else None
        self.save()
        if not self.a.keep_nodeb:
            open(os.path.join(CTL, "SHUTDOWN"), "w").close()
            self.event("node B released (ctl/SHUTDOWN)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-job", required=True)
    ap.add_argument("--phase", choices=["all", "final", "reports"], default="all")
    ap.add_argument("--keep-nodeb", action="store_true")
    a = ap.parse_args()
    os.makedirs(CTL, exist_ok=True)
    d = Driver(a)
    d.event(f"driver start pid {os.getpid()} phase {a.phase} on {d.st['head']['host']} (job {a.head_job})")
    if a.phase == "reports":
        d.reports()
        return
    ok = True
    if a.phase == "all":
        ok = d.train()
    if ok and d.step_done() >= STEPS:
        d.final()
    else:
        d.wait_milestones()
    d.finish()


if __name__ == "__main__":
    main()
