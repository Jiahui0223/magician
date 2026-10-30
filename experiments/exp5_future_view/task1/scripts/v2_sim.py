"""EXP5 Task 1.4 -- simulator / evaluation side (MAGICIAN env). A copy of the frozen abl_2 simulator
(experiments/exp4_multiworld_belief/ablation_study/abl_2/common/abl2_sim.py, sha256 92571175...) with exactly these
changes: definitions from v2_common, the planner child is v2_planner.py (with --rep), episode tags carry the noise
replicate (r<rep>). Everything else -- seeded observation, common legal list, hover, shared coverage code, AUC@100 =
trapz(cov[0..100]) / 100, per-step checkpoint, bitwise resume replay, ABL_INTERRUPT_AFTER -- is unchanged.
Usage: $MAGICIAN_PYTHON v2_sim.py --exp E1..E6 --scene S --start K --rep R --root DIR [--moves N] [--planner-gpu G]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_common as LC  # noqa: E402

sys.path.insert(1, LC.EXP3B)
import m_t6_sim as T6S  # noqa: E402  (frozen T6 simulator: EpisodeSim)
import m_common as MC  # noqa: E402
import m_t6_common as T6  # noqa: E402
import m_t7_common as T7  # noqa: E402

np, torch = T6S.np, T6S.torch


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "v2_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t)]
        env = T6.seen2scene_env(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if a.planner_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True,
                                  bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")


def read_ckpt(path):
    if not os.path.exists(path):
        return None, []
    head, steps = None, []
    for i, line in enumerate(open(path)):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            break
        if i == 0:
            head = r
        else:
            if r["t"] != len(steps):
                break
            steps.append(r)
    return head, steps


def svc_equal(p, rec):
    old = np.load(p)
    return set(old.files) == set(rec) and all(np.array_equal(old[x], np.asarray(rec[x])) for x in rec)


def run(a):
    t_start = time.time()
    s, k, moves = a.scene, a.start, a.moves
    tg = LC.tag(a.exp, s, k, a.rep)
    D = LC.dirs(a.root, tg)
    assert not os.path.exists(D["episode"]), D["episode"]
    for x in ("episodes", "planner_logs", "ckpt", "svc", "frames"):
        os.makedirs(os.path.join(a.root, x), exist_ok=True)
    head, done = read_ckpt(D["ckpt"])
    me = {"tag": tg, "exp": a.exp, "method": a.exp, "scene": s, "start": k, "rep": a.rep, "moves": moves}
    if head is not None and {x: head.get(x) for x in me} != me:
        raise RuntimeError(f"INVALID: checkpoint header {head} != {me}")
    if not done:                                  # nothing executed yet: clean start
        for p in (D["svc"], D["frames"]):
            shutil.rmtree(p, ignore_errors=True)
        for x in ("steps.jsonl", "refresh.jsonl", "json", "stderr"):
            p = os.path.join(D["plog"], f"pl_{tg}.{x}")
            if os.path.exists(p):
                os.remove(p)
        with open(D["ckpt"], "w") as f:
            f.write(json.dumps(dict(me, created=time.ctime())) + "\n")
    else:
        shutil.rmtree(D["frames"], ignore_errors=True)          # rendered frames are re-rendered by the replay
        lines = open(D["ckpt"]).read().splitlines()[:1 + len(done)]
        with open(D["ckpt"], "w") as f:
            f.write("\n".join(lines) + "\n")
    T = len(done)
    os.makedirs(D["svc"], exist_ok=True)
    sim = T6S.EpisodeSim(s, k, D["frames"])
    torch.cuda.reset_peak_memory_stats(sim.dev)
    resume = None
    if T:
        tr = time.time()
        bad = []
        for t in range(T):
            ob = sim.observe(t)
            legal, _ = sim.legal()
            keys = [T6.key_of(x[0]) for x in legal]
            st = done[t]
            if (ob["cov"] != st["cov"] or sim.key() != st["pose_key"] or keys != st["legal_keys"]
                    or int(len(ob["sample_flat"])) != st["n_samples"]):
                bad.append({"t": t, "what": "observation"})
            rec = sim.service_record(t, ob, legal)
            if not svc_equal(os.path.join(D["svc"], f"svc_{s}_s{k}_t{t}.npz"), rec):
                bad.append({"t": t, "what": "service_record"})
            sim.move(st["action_key"])
        if bad:
            raise RuntimeError(f"INVALID: resume replay differs from the checkpoint: {bad[:5]}")
        resume = {"T": T, "replay_s": time.time() - tr, "replay_bitwise_equal": True}
    planner = Planner(a, T, os.path.join(D["plog"], f"pl_{tg}.stderr"))
    hellos = [planner.hello]
    out = {"method": a.exp, "exp": a.exp, "name": LC.ARMS[a.exp]["name"], "scene": s, "start": k, "rep": a.rep, "seed": 0, "tag": tg, "kind": "service",
           "planner_gpu": a.planner_gpu, "sim_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "host": os.uname()[1]}
    cov = [x["cov"] for x in done]
    steps = list(done)
    interrupt = int(os.environ.get("ABL_INTERRUPT_AFTER", "-1"))
    try:
        for t in range(T, moves + 1):
            ts = time.time()
            ob = sim.observe(t)
            cov.append(ob["cov"])
            legal, flags = sim.legal()
            keys = [T6.key_of(x[0]) for x in legal]
            step = {"t": t, "cov": ob["cov"], "pose_key": sim.key(), "legal_keys": keys, "n_legal": len(keys),
                    "n_samples": int(len(ob["sample_flat"])), "observe_s": time.time() - ts}
            if t == moves:
                steps.append(step)
                break
            td = time.time()
            rec = sim.service_record(t, ob, legal)
            p = os.path.join(D["svc"], f"svc_{s}_s{k}_t{t}.npz")
            np.savez_compressed(p, **rec)
            ans = planner.ask(t, p)
            pick = ans["pick"]
            if pick is not None and not (0 <= pick < len(keys)):
                raise RuntimeError(f"INVALID: illegal pick {pick} of {len(keys)} at t={t}")
            step.update({"pick": pick, "action_key": keys[pick] if pick is not None else None, "hover": pick is None,
                         "decide_wall_s": time.time() - td, "planner": ans})
            sim.move(None if pick is None else keys[pick])
            steps.append(step)
            with open(D["ckpt"], "a") as f:
                f.write(json.dumps(step, default=float) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if t == interrupt:
                planner.p.kill()
                print(f"ABL {tg}: injected interruption after t={t}", flush=True)
                sys.exit(LC.EXIT_INTERRUPT)
        bye = planner.close()
    except BaseException:
        if planner.p.poll() is None:
            planner.p.kill()
        raise
    out.update({"steps": steps, "planner_hello": hellos, "planner_bye": bye, "resume": resume, "moves": moves,
                "coverage": cov, "auc": T7.auc(cov, moves), "cov_final": cov[-1], "cov0": cov[0],
                "cov_at": {str(x): cov[x] for x in (10, 20, 50, 100) if x <= moves},
                "executed_keys": [x["pose_key"] for x in steps],
                "n_hover": sum(1 for x in steps[:-1] if x["hover"]),
                "sim_peak_mem_gb_this_process": torch.cuda.max_memory_allocated(sim.dev) / 2 ** 30,
                "seconds_this_process": time.time() - t_start})
    json.dump(out, open(D["episode"], "w"), indent=1, default=float)
    os.chmod(D["episode"], 0o444)
    shutil.rmtree(D["frames"])
    print(f"ABL episode {tg}: done, cov@{moves} {cov[-1]:.4f}, auc {out['auc']:.4f}, "
          f"{out['seconds_this_process']:.0f} s (resumed at {T})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=LC.RUN_ORDER)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--moves", type=int, default=LC.MOVES)
    ap.add_argument("--planner-gpu", type=int, default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
