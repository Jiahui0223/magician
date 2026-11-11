"""EXP8-MP3D -- indoor replay driver (magician env; one planner child per trajectory). Same idea as eff_replay.py (outdoor): the RECORDED
service records of a real EXP7 MP3D-150 closed-loop trajectory (work_dirs/cvpr_results/exp7/episodes/mp3d150/raw/OURS/V2M/<tag>/svc,
read-only, linked into a fresh folder) are fed step by step to a frozen indoor planner child with the t23_in_sim protocol
({t, svc} -> answer). Both methods see the same observation sequence; nothing is simulated, rendered or evaluated here.
  OURS-PROF  eff_in_prof_planner.py (= exp7 d_in_planner.py chain + synchronised component timers)
  OURS       exp7 d_in_planner.py unchanged (latency without timers; overhead check)
  MAG-B      task0 adapters/magician_planner_grid.py --method MAG-B (official MAGICIAN core, beam 10 x 10, grid neighbours)
Output: <out>/<METHOD>/<tag>/{steps.jsonl, summary.json, planner.err}. Usage: python eff_in_replay.py --method M --scan S --level L
--start K [--tmax 150] --out DIR"""
import argparse
import json
import os
import subprocess
import sys
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task2", "scripts"))
import t23_in_sim as T23  # noqa: E402

S, C, S2E = T23.S, T23.C, T23.S2E
SRC = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "episodes", "mp3d150", "raw", "OURS", "V2M")
CK = os.path.join(R, "work_dirs", "exp6_v3", "task0_vae", "checkpoints", "final.pt")
E7S = os.path.join(R, "experiments", "cvpr_results", "exp7", "scripts")
D_CONFIG = os.path.join(R, "experiments", "cvpr_results", "exp7", "configs", "d_config_mp3d.yaml")


def ours_child(script, key, tag, od, cfg):
    g = cfg["groups"]["V2M"]
    env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO, T3_P1_MODE="b2",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", D_POLICY="d1", D_CONFIG=D_CONFIG, V3_CKPT=CK)
    cmd = [f"{S2E}/bin/python", script, "--group", "V2M", "--card", key, "--tag", tag,
           "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"], "--svc-dir", os.path.join(od, "svc")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"), text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"]
    return p, hello


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["OURS", "OURS-PROF", "MAG-B"])
    ap.add_argument("--scan", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--tmax", type=int, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    key = f"{a.scan}_L{a.level}"
    tag = f"{key}_S{a.start}"
    od = os.path.join(a.out, a.method, tag)
    assert not os.path.exists(os.path.join(od, "summary.json")), od
    os.makedirs(os.path.join(od, "svc"), exist_ok=True)
    src = os.path.join(SRC, tag)
    for t in range(a.tmax):
        f = f"svc_t{t}.npz"
        if not os.path.lexists(os.path.join(od, "svc", f)):
            os.symlink(os.path.join(src, "svc", f), os.path.join(od, "svc", f))
    rec = {}
    for line in open(os.path.join(src, "planner.jsonl")):
        q = json.loads(line)
        rec[int(q["t"])] = q.get("pick")
    cfg = json.load(open(os.path.join(C.T2, "configs", "eval_indoor_v1.json")))
    t_boot = time.perf_counter()
    if a.method == "MAG-B":
        p, hello = T23.child("MAG-B", key, tag, od, cfg)
    else:
        script = os.path.join(R, "experiments", "cvpr_results", "efficiency", "scripts", "eff_in_prof_planner.py") if a.method == "OURS-PROF" else os.path.join(E7S, "d_in_planner.py")
        p, hello = ours_child(script, key, tag, od, cfg)
    boot_s = time.perf_counter() - t_boot
    rows = []
    with open(os.path.join(od, "steps.jsonl"), "w") as fo:
        for t in range(a.tmax):
            t0 = time.perf_counter()
            p.stdin.write(json.dumps({"t": t, "svc": os.path.join(od, "svc", f"svc_t{t}.npz")}) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
            rt = time.perf_counter() - t0
            if not line:
                raise RuntimeError(f"planner ended at t={t}")
            ans = json.loads(line)
            row = {"t": t, "roundtrip_s": rt, "pick": ans.get("pick"), "recorded_pick": rec.get(t), "timing": ans.get("timing"), "diag": ans.get("diag"),
                   "n_legal": len(ans.get("scores") or []), "peak_mem_gb": ans.get("peak_mem_gb"), "refresh": ans.get("refresh") is not None}
            fo.write(json.dumps(row, default=float) + "\n")
            fo.flush()
            rows.append(row)
    p.stdin.write(json.dumps({"bye": True, "cmd": "end"}) + "\n")
    p.stdin.flush()
    try:
        p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        p.kill()
    same = [r["pick"] == r["recorded_pick"] for r in rows if r["recorded_pick"] is not None]
    summ = {"method": a.method, "tag": tag, "tmax": a.tmax, "boot_s": boot_s, "hello": hello, "host": os.uname()[1],
            "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "picks_equal_recorded": f"{sum(same)}/{len(same)}", "roundtrip_mean_s": sum(r["roundtrip_s"] for r in rows) / len(rows),
            "peak_mem_gb": max((r["peak_mem_gb"] or 0) for r in rows)}
    json.dump(summ, open(os.path.join(od, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: summ[k] for k in ("method", "tag", "boot_s", "picks_equal_recorded", "roundtrip_mean_s", "peak_mem_gb", "gpu")}, default=float), flush=True)


if __name__ == "__main__":
    main()
