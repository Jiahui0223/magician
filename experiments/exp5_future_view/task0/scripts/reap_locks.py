"""EXP5 Task 0 -- stale-lock reaper for the atomic claims of t04_gen.py (fixed states) and run_cl.sh (closed-loop
episodes). A lock without its output is stale when its owner (content "host:pid") is this host and the pid is dead,
or when the lock is empty (old format) or owned by another node and older than --max-age hours (episodes 3, states 1.5). A stale claim is released for one
retry (<id>.attempts counts the failures); at --max-attempts it is kept and <id>.failed.json records the failure
(reported in the analysis, never silently dropped).
Usage: reap_locks.py states <dir> | reap_locks.py episodes <dir>     (dir = .../<model>)
"""
import json
import os
import socket
import sys
import time


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    kind, d = sys.argv[1], sys.argv[2]
    max_age = float(sys.argv[3]) if len(sys.argv) > 3 else (3.0 if kind == "episodes" else 1.5)
    max_attempts = 2
    host = socket.gethostname().split(".")[0]
    if not os.path.isdir(d):
        return
    for f in sorted(os.listdir(d)):
        if not f.endswith(".lock"):
            continue
        sid = f[:-5]
        out = os.path.join(d, sid, "episode.json") if kind == "episodes" else os.path.join(d, f"{sid}.json")
        lk = os.path.join(d, f)
        if os.path.exists(out) or os.path.exists(os.path.join(d, f"{sid}.failed.json")):
            continue
        owner = open(lk).read().strip()
        age_h = (time.time() - os.path.getmtime(lk)) / 3600.0
        if owner:
            h, pid = owner.split(":")
            if h.split(".")[0] == host:
                stale = not alive(int(pid))
            else:                               # another node (e.g. an allocation that has ended): by age only
                stale = age_h > max_age
        else:
            stale = age_h > max_age
        if not stale:
            continue
        ap = os.path.join(d, f"{sid}.attempts")
        n = (int(open(ap).read().strip() or 0) if os.path.exists(ap) else 0) + 1
        open(ap, "w").write(str(n))
        if n >= max_attempts:
            json.dump({"id": sid, "status": "FAILED", "attempts": n, "reason": "worker died (CUDA out of memory in "
                       "the frozen world model / fvdb, or the allocation ended) on every attempt", "owner": owner,
                       "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(d, f"{sid}.failed.json"), "w"), indent=1)
            print("FAILED", d, sid, n, flush=True)
        else:
            os.remove(lk)
            print("released", d, sid, "attempt", n, flush=True)


if __name__ == "__main__":
    main()
