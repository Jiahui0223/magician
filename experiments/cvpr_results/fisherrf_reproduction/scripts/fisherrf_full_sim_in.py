"""FisherRF-full (frozen variant f1c4) on MP3D-150 indoors -- simulator side (magician env; GT side).

= fisherrf_sim_in.py (the D3a indoor wrapper, imported as a module, NOT edited) with only the planner child swapped:
  * task2/scripts/t23_in_sim.main runs UNCHANGED (T0.12 grid simulator through t12_sim / mp3d_sim: grid, start snapping, official
    grid neighbours, GT single-segment legality, frozen depth sensor, 10 % sampling seeds, C_obs evaluator, svc_t<t>.npz records);
  * the same textured RGB side files svc_t<t>_rgb.npz (fisherrf_sim_in.TexturedRGB / service_record, atlas 4, ambient 0.85;
    zbuf of the textured render checked against the sensor every step -> rgb_zbuf_check.json), as the D3a indoor run;
  * group "FisherRF-full"; child = fisherrf_full_planner_in.py (fisherrf env) = the frozen fisherrf_full_planner.py with the
    indoor settings dict built from the scene card; D12 --init-pose docstring, --verify-every 0, --rt-tol 1e-2 (I9), the f1c4 environment
    FRF_FRONTIER_MODE=magician FRF_REACH_FILTER=1 (asserted; inherited by the child).
GT stays on this side: mesh, textured render, segment legality, C_obs. The child receives svc paths only (open() guard).
Output <out>/FisherRF-full/<level>_S<start>/{svc/, episode.json, planner.jsonl, planner.err, planner_info.json,
planner_fisherrf_full.steps.jsonl, planner_settings/, planner_work/, rgb_zbuf_check.json}.
Usage: python fisherrf_full_sim_in.py --scan S --level L --start K --moves N --out DIR   (CUDA_VISIBLE_DEVICES: GPU of both)
"""
import json
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fisherrf_sim_in as FSI  # noqa: E402  (D3a indoor wrapper; its __main__ block does not run)

T23, T12 = FSI.T23, FSI.T12
GROUP = "FisherRF-full"
INIT_POSE, VERIFY_EVERY = "docstring", "0"                                  # D12, as the outdoor f1c4 tasks
RT_TOL = "1e-2"     # F1c-in I9: grid-vs-simulator R / T self-check; the simulator's float32 look_at (at = X + ray) gives
                    # |dT| ~ 1.1e-5 |X|, indoor |X| up to ~250 runtime -> up to ~3e-3 (outdoor default 1e-3 kept)


def child(group, key, tag, od, cfg):
    assert group == GROUP, group
    FSI.CTX["od"] = od
    start = tag.rsplit("_S", 1)[1]
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH", "PYTHONHOME")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", SLURM_JOB_QOS="kd-high",
               CONDA_PREFIX=os.path.dirname(os.path.dirname(FSI.FISHERRF_PYTHON)))
    cmd = [FSI.FISHERRF_PYTHON, "-B", "-s", os.path.join(HERE, "fisherrf_full_planner_in.py"), "--scene", key, "--start", start,
           "--svc-dir", os.path.join(od, "svc"), "--log", os.path.join(od, "planner_fisherrf_full.steps.jsonl"),
           "--workdir", os.path.join(od, "planner_work"), "--init-pose", INIT_POSE, "--verify-every", VERIFY_EVERY,
           "--rt-tol", RT_TOL]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"),
                         text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"], hello
    hello.setdefault("info", {}).update({"textured_rgb": {"geometry": FSI.CTX["tex"].geom, "atlas_size": FSI.CTX["tex"].atlas_size,
                                                          "ambient": FSI.CTX["tex"].ambient, "build_s": FSI.CTX["tex"].build_s},
                                         "variant": "f1c4", "env": {k: os.environ[k] for k in ("FRF_FRONTIER_MODE", "FRF_REACH_FILTER")},
                                         "planner_cmd": cmd, "rt_tol": float(RT_TOL), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    return p, hello


if __name__ == "__main__":
    import torch
    assert os.environ.get("FRF_FRONTIER_MODE") == "magician" and os.environ.get("FRF_REACH_FILTER") == "1", \
        "set FRF_FRONTIER_MODE=magician FRF_REACH_FILTER=1 (frozen f1c4)"
    for flag in ("--group", "--init-pose", "--verify-every", "--atlas"):
        assert flag not in sys.argv, f"{flag} is fixed by this wrapper"
    FSI.GROUP = GROUP                                                          # read by FSI._AP (--group choices)
    scan = sys.argv[sys.argv.index("--scan") + 1]
    FSI.CTX["tex"] = FSI.TexturedRGB(scan, torch.device("cuda"), atlas_size=4)
    print("textured RGB ready", json.dumps(FSI.CTX["tex"].geom), "build %.1f s" % FSI.CTX["tex"].build_s, flush=True)
    T12.service_record = FSI.service_record
    T23.child = child
    T23.argparse = types.SimpleNamespace(ArgumentParser=FSI._AP)
    sys.argv += ["--group", GROUP]
    T23.main()
    z = FSI.CTX["zcheck"]
    summ = {"n": len(z), "all_hit_mask_equal": all(x["hit_mask_equal"] for x in z),
            "zbuf_maxabs_diff_max": max(x["zbuf_maxabs_diff"] for x in z), "frames": z}
    json.dump(summ, open(os.path.join(FSI.CTX["od"], "rgb_zbuf_check.json"), "w"), indent=1)
    print("RGB / zbuf check:", json.dumps({k: v for k, v in summ.items() if k != "frames"}), flush=True)
