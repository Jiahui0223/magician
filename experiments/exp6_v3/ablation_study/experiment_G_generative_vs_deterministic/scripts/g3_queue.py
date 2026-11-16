"""Experiment G3 -- task files for the shared exclusive-GPU queue (work_dirs/cvpr_results/gpu_queue/pending/expG3_*.sh; protocol G3-v1).
  smoke   expG3_a_*  10-move smokes: outdoor G4R (hooks off, K = 4, must equal V3-T0-D1 liberty s0), MRAW, MDEPTH; indoor G4RIN (must equal
                     EXP7 OURS EU6Fwq7SyZv_L0 S34), MRAWIN, MDEPTHIN
  fixed   expG3_b_*  fixed-state workers (g3_fixed.py over confirm then explore; trajectory claims)
  closed  expG3_c_*  outdoor MRAW / MDEPTH x All-14 x 5 (dunnottar first)   +   expG3_d_* indoor MRAWIN / MDEPTHIN x 44 (longest first)
Dry run unless --write. Usage: python g3_queue.py smoke|fixed|closed [--workers 8] [--only NAMES] [--write]"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g2_queue as Q2  # noqa: E402

R, GD, Q = Q2.R, Q2.GD, Q2.Q
SCENE_ORDER = ["dunnottar", "colosseum", "bannerman", "pantheon", "redeemer", "liberty", "pisa", "fushimi", "alhambra", "neuschwanstein",
               "eiffel", "bridge", "barts", "sestino_museum"]
WS = os.path.join(R, "work_dirs/exp6_v3/experiment_G/g3/smoke")
CHK = f"{Q2.PYENV} && $S2E/bin/python {GD}/scripts/g3_smoke_check.py"
def task(name, comment, cmd):
    return name, f"#!/bin/bash\n# Experiment G3 (protocol G3-v1): {comment}\n{cmd}\n"


Q2.task = task
OK = "$S2E/bin/python -c \"import json,sys; sys.exit(0 if json.load(open('{p}'))['pass'] else 7)\""


def smoke(a):
    out = []
    ref_out = os.path.join(R, "work_dirs/cvpr_results/table1/g3/closed_loop/outdoor/V3-T0-D1/V2M")
    for arm, mode in (("G4R", "off"), ("MRAW", "raw"), ("MDEPTH", "depth")):
        d = os.path.join(WS, "out", "closed_loop", arm, "liberty_s0")
        res = os.path.join(WS, f"out_{arm}.json")
        out.append(Q2.task(f"expG3_a_out_{arm}", f"outdoor smoke {arm} liberty s0 10 moves",
                           f"export G_MOVES=10 G_WORKDIR={WS}/out\nbash {GD}/scripts/g3_episode.sh {arm} liberty 0 || exit $?\n"
                           f"{CHK} --domain out --dir {d} --moves 10 --mode {mode} " + (f"--ref-dir {ref_out} " if arm == "G4R" else "") + f"--out {res} && " + OK.format(p=res)))
    r = sorted(Q2.ours_rows(), key=lambda r: r["wall"])[0]
    tag = f"{r['scene_id']}_S{r['start_id']}"
    ref_in = os.path.join(Q2.W7, "episodes/mp3d150/raw/OURS/V2M", tag)
    for arm, mode in (("G4RIN", "off"), ("MRAWIN", "raw"), ("MDEPTHIN", "depth")):
        d = os.path.join(WS, "in", "raw", arm, "V2M", tag)
        res = os.path.join(WS, f"in_{arm}.json")
        out.append(Q2.task(f"expG3_a_in_{arm}", f"indoor smoke {arm} {tag} 10 moves",
                           f"export G_MOVES=10 G_WORKDIR={WS}/in\nbash {GD}/scripts/g3_in_episode.sh {arm} {r['scan']} {r['level']} {r['start_id']} || exit $?\n"
                           f"{CHK} --domain in --dir {d} --moves 10 --mode {mode} " + (f"--ref-dir {ref_in} " if arm == "G4RIN" else "") + f"--out {res} && " + OK.format(p=res)))
    return out


def fixed(a):
    return [Q2.task(f"expG3_b_fixed_w{w}", f"fixed-state worker {w} (confirm then explore)",
                    f"{Q2.PYENV} && $S2E/bin/python {GD}/scripts/g3_fixed.py --set confirm && $S2E/bin/python {GD}/scripts/g3_fixed.py --set explore")
            for w in range(1, a.workers + 1)]


def closed(a):
    out = []
    for i, s in enumerate(SCENE_ORDER):
        for k in range(5):
            for arm in ("MRAW", "MDEPTH"):
                out.append(Q2.task(f"expG3_c_{i:02d}_{s}_s{k}_{arm}", f"outdoor closed loop {arm} {s} start {k}, 100 moves + official replay",
                                   f"bash {GD}/scripts/g3_episode.sh {arm} {s} {k}"))
    for i, r in enumerate(Q2.ours_rows()):
        for arm in ("MRAWIN", "MDEPTHIN"):
            out.append(Q2.task(f"expG3_d_in_{i:02d}_{arm}_{r['scene_id']}_S{r['start_id']}", f"MP3D-150 {arm} {r['scene_id']} S{r['start_id']}",
                               f"bash {GD}/scripts/g3_in_episode.sh {arm} {r['scan']} {r['level']} {r['start_id']}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("smoke", "fixed", "closed"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    T = {"smoke": smoke, "fixed": fixed, "closed": closed}[a.what](a)
    if a.only:
        T = [t for t in T if t[0] in a.only.split(",")]
    for name, body in T:
        if os.path.exists(os.path.join(Q, "pending", name + ".sh")) or any(f.startswith(name + "__") for d in ("running", "done", "failed") for f in os.listdir(os.path.join(Q, d))):
            print("exists", name)
            continue
        if a.write:
            p = os.path.join(Q, "pending", name + ".sh")
            with open(p + ".tmp", "w") as f:
                f.write(body)
            os.replace(p + ".tmp", p)
        print(("queued " if a.write else "dry-run ") + name)
    print(len(T), "tasks", "written" if a.write else "(dry run; add --write)")


if __name__ == "__main__":
    main()
