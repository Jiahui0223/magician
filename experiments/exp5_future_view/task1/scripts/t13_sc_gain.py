"""EXP5 Task 1.3 S-GAIN-C (magician env, 1 GPU; EVALUATION SIDE, Train-6 only): true one-step gains of every legal
action at the S-c calibration states (protocol §4.5).

Replays a held-out Train-6 trajectory with the frozen T0EpisodeSim (exp4 seeds, rec number of t0_common.traj_num); every
replayed service record must equal the L1 record bitwise; at t in Sc_times the frozen T7a Oracle evaluator
(m_gain_eval.evaluate_actions: restores camera, covered scene and RNG states) gives g per legal action (coverage points
x 100). The executed moves come from the simulator-side sim_meta.json (evaluation side only).
Output <out>/gain_<scene>_<traj>.json
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402

import t0_common as C  # noqa: E402
import t0_sim as T0  # noqa: E402

np, torch, MC, T6, GE = T0.np, T0.torch, T0.MC, T0.T6, T0.GE
SC = VF.load_cfg("scoring_v1.json")


def run_one(scene, traj, out_dir):
    p_out = os.path.join(out_dir, f"gain_{scene}_{traj}.json")
    if os.path.exists(p_out):
        return
    C.assert_geometry_allowed(scene, "train")
    tb = time.time()
    tdir = os.path.join(VF.DATA_V1, "trajectories", scene, traj)
    meta = json.load(open(os.path.join(tdir, "sim_meta.json")))
    num = C.traj_num(scene, traj)
    assert num == meta["rec_num"], (num, meta["rec_num"])
    keys = meta["executed_keys"]
    times = SC["Sc_times"]
    fr = T0.scratch_dir(f"exp5gain_{scene}_{traj}_{os.getpid()}")
    sim = T0.T0EpisodeSim(scene, meta["start_idx"], num, T0.exp4_seeds(scene, traj), fr)
    rows, bad = [], []
    try:
        for t in range(max(times) + 1):
            assert sim.key() == keys[t], (t, sim.key(), keys[t])
            ob = sim.observe(t)
            legal, _ = sim.legal()
            lk = [T6.key_of(x[0]) for x in legal]
            rec = sim.service_record(t, ob, legal)
            ref = np.load(os.path.join(tdir, "records", f"svc_{scene}_s{num}_t{t}.npz"))
            diff = [f for f in sorted(MC.SERVICE_KEYS_REQUIRED) if not np.array_equal(np.asarray(rec[f]), ref[f])]
            if diff:
                bad.append({"t": t, "fields": diff})
                raise VF.InvalidRun(f"INVALID: replay differs from the L1 record {scene}/{traj} t={t}: {diff}")
            if t in times and lk:
                g, sim.covered = GE.evaluate_actions(sim.camera, sim.mesh, sim.params, sim.dev, scene, num, t,
                                                     sim.covered, sim.gt_scene, sim.eps, sim.vis, ob["cov"], lk,
                                                     sim.camera.get_idx_from_key)
                rows.append({"t": t, "keys": lk, "g": [float(g[x]) for x in lk], "cov": ob["cov"]})
            nxt = keys[t + 1]
            hover = len(lk) == 0 and nxt == keys[t]
            sim.move(None if hover else nxt)
    finally:
        shutil.rmtree(fr, ignore_errors=True)
    out = {"scene": scene, "traj": traj, "rec_num": num, "states": rows, "replay_bitwise_equal": not bad,
           "seconds": time.time() - tb, "gpu": torch.cuda.get_device_name(0)}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(p_out + ".tmp", "w"), indent=1)
    os.replace(p_out + ".tmp", p_out)
    print(f"gain {scene}/{traj}: {len(rows)} states, {out['seconds']:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--out", default=os.path.join(VF.WORK, "fixed_state", "sc_gain"))
    a = ap.parse_args()
    VF.verify_frozen()
    sp = VF.frozen_values()["split"]["scenes"]
    items = [(s, tj) for s in sp for tj in sp[s]["heldout_trajs"]]
    pi, pn = (int(x) for x in a.part.split("/"))
    for s, tj in items[pi::pn]:
        run_one(s, tj, a.out)


if __name__ == "__main__":
    main()
