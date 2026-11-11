"""EXP8 -- replay driver for decision-latency measurement (magician env; one planner child per trajectory).

Feeds the RECORDED service records of a real closed-loop trajectory (CVPR Table 1 roots, read-only; linked into a fresh root under
work_dirs/cvpr_results/efficiency/runs/) to a frozen planner child, step by step, with exactly the JSON-lines protocol of the
closed-loop simulator (m_t6_sim.PlannerClient: {t, svc} -> answer). Every method therefore sees the same observation sequence
(strictly paired input states). No simulation, rendering or coverage evaluation happens here.

Methods (planner children unchanged, environment as in their closed-loop wrappers):
  OURS-K4D2  table3 kd_out_planner.py, KD_K 4, D-v1 policy d1  (= Table 1 V3-T0-D1 = RAYSCOUT; Table 3 regression bitwise)
  OURS-K4D1 / OURS-K1D2 / OURS-K1D1  the Table 3 rows (depth1 = one-step value_cov, KD-v1)
  OURS-K4D2-PROF  same as OURS-K4D2 through eff_prof_planner.py (synchronised component timers; breakdown only)
  MAG-B      task2 t24_mag_out_planner.py --method MAG-B (official MAGICIAN core, beam 10 x 10, official settings)
Per step the driver records the wall-clock round trip (perf_counter; the planner writes its answer only after its GPU work, so
the round trip is synchronous) and the planner's own synchronised timing; the planner children also write their usual logs
(planner_logs/*.steps.jsonl for OURS). Output: <out>/<method>/<src>_<scene>_s<k>.jsonl + .json summary.
Usage: python eff_replay.py --method M --src V3-T0-D1|MAG-B --scene S --start K [--tmax 100] --out DIR"""
import argparse
import json
import os
import subprocess
import sys
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
X = os.path.join(R, "experiments")
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task2", "scripts"))
import t24_out_sim as T24  # noqa: E402

LC, C, CFG, T6 = T24.LC, T24.C, T24.CFG, T24.T6
T1 = os.path.join(R, "work_dirs", "cvpr_results", "table1", "g3", "closed_loop", "outdoor")
SRC = {"V3-T0-D1": (os.path.join(T1, "V3-T0-D1", "V2M"), "V2M"), "MAG-B": (os.path.join(T1, "MAG-B", "MAG-B"), "MAG-B")}
CK = os.path.join(R, "work_dirs", "exp6_v3", "task0_vae", "checkpoints", "final.pt")
T3 = os.path.join(X, "cvpr_results", "scripts", "table3")
DCFG = os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "configs", "d_config.yaml")
HERE = os.path.dirname(os.path.abspath(__file__))
OURS = {"OURS-K4D2": ("4", "2"), "OURS-K4D1": ("4", "1"), "OURS-K1D2": ("1", "2"), "OURS-K1D1": ("1", "1"), "OURS-K4D2-PROF": ("4", "2")}


def start_child(method, scene, start, root, svc_dir, err):
    if method == "MAG-B":
        cmd = [LC.MAGICIAN_PYTHON, "-I", os.path.join(X, "exp5_future_view", "task2", "scripts", "t24_mag_out_planner.py"), "--method", "MAG-B",
               "--scene", scene, "--start", str(start), "--svc-dir", svc_dir, "--resume-t", "0"]
        env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH")}
    else:
        kk, dd = OURS[method]
        g = CFG["groups"]["V2M"]
        entry = os.path.join(HERE, "eff_prof_planner.py") if method.endswith("-PROF") else os.path.join(T3, "kd_out_planner.py")
        cmd = [LC.SEEN2SCENE_PYTHON, entry, "--exp", "V2M", "--scene", scene, "--start", str(start), "--rep", "0", "--root", root,
               "--resume-t", "0", "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
        env = T6.seen2scene_env(os.environ)
        env.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "T3_P1_MODE": "b2", "V3_CKPT": CK, "KD_K": kk, "KD_NOISE_OFFSET": "0"})
        if dd == "2":
            env.update({"D_POLICY": "d1", "D_CONFIG": DCFG})
        else:
            env.update({"D_POLICY": "depth1", "D_CONFIG": os.path.join(T3, "kd_config.yaml")})
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, cwd=LC.REPO, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(OURS) + ["MAG-B"])
    ap.add_argument("--src", required=True, choices=list(SRC))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--tmax", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    src_root, src_grp = SRC[a.src]
    src_tag = LC.tag(src_grp, a.scene, a.start, 0)
    src_svc = os.path.join(src_root, "svc", src_tag)
    run = f"{a.src}_{a.scene}_s{a.start}"
    root = os.path.join(a.out, a.method, run)
    assert not os.path.exists(os.path.join(root, "summary.json")), root
    tg = LC.tag("V2M", a.scene, a.start, 0) if a.method != "MAG-B" else f"MAGB_{a.scene}_s{a.start}"
    svc_dir = os.path.join(root, "svc", tg)
    os.makedirs(svc_dir, exist_ok=True)
    os.makedirs(os.path.join(root, "planner_logs"), exist_ok=True)
    for t in range(a.tmax):
        f = f"svc_{a.scene}_s{a.start}_t{t}.npz"
        if not os.path.lexists(os.path.join(svc_dir, f)):
            os.symlink(os.path.join(src_svc, f), os.path.join(svc_dir, f))
    rec_picks = {}
    ck = os.path.join(src_root, "ckpt", f"{src_tag}.jsonl")
    if os.path.exists(ck):
        for line in open(ck):
            s = json.loads(line)
            if "t" in s and "pick" in s:
                rec_picks[int(s["t"])] = s["pick"]
    err = open(os.path.join(root, "planner.stderr"), "w")
    t_boot = time.perf_counter()
    p = start_child(a.method, a.scene, a.start, root, svc_dir, err)
    hello = json.loads(p.stdout.readline())
    boot_s = time.perf_counter() - t_boot
    assert hello.get("ready"), hello
    rows = []
    with open(os.path.join(root, "steps.jsonl"), "w") as fo:
        for t in range(a.tmax):
            path = os.path.join(svc_dir, f"svc_{a.scene}_s{a.start}_t{t}.npz")
            t0 = time.perf_counter()
            p.stdin.write(json.dumps({"t": t, "svc": path}) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
            rt = time.perf_counter() - t0
            if not line:
                raise RuntimeError(f"planner ended at t={t}")
            ans = json.loads(line)
            if "error" in ans:
                raise RuntimeError(f"planner error at t={t}: {ans['error']}")
            row = {"t": t, "roundtrip_s": rt, "pick": ans.get("pick"), "recorded_pick": rec_picks.get(t),
                   "decision_s": ans.get("decision_s"), "n_legal": len(ans.get("scores") or []),
                   "diag": ans.get("diag"), "timing": ans.get("timing")}
            fo.write(json.dumps(row, default=float) + "\n")
            fo.flush()
            rows.append(row)
    p.stdin.write(json.dumps({"cmd": "end"}) + "\n")
    p.stdin.flush()
    bye = json.loads(p.stdout.readline() or "{}")
    p.wait(timeout=600)
    err.close()
    same = [r["pick"] == r["recorded_pick"] for r in rows if r["recorded_pick"] is not None]
    summ = {"method": a.method, "src": a.src, "scene": a.scene, "start": a.start, "tmax": a.tmax, "boot_s": boot_s, "hello": hello, "bye": bye,
            "host": os.uname()[1], "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "picks_equal_recorded": f"{sum(same)}/{len(same)}",
            "roundtrip_mean_s": sum(r["roundtrip_s"] for r in rows) / len(rows), "returncode": p.returncode}
    json.dump(summ, open(os.path.join(root, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: summ[k] for k in ("method", "src", "scene", "start", "boot_s", "picks_equal_recorded", "roundtrip_mean_s", "gpu")}, default=float), flush=True)


if __name__ == "__main__":
    main()
