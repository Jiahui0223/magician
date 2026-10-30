"""EXP5 Task 0.4 -- world-model arms on the indoor fixed states (seen2scene env; planner side: reads only the scene card
and the whitelisted service records 0..t of the state, never GT).

Models: frozen = Frozen Seen2Scene (S0-F, t1_eval.build_outdoor_wm("frozen")), C = T1-C@2000 (sha checked); both the
frozen E-M2 world-model configuration (10 Euler steps, overlap 0.2, drop_bbox, world seeding).
Per state: region frame around the current camera (mp3d_frame.region_geometry), frozen P1 replay of records 0..t,
K = 4 worlds (m_t7_planner.generate_refresh, world key "T0|<state id>", seed_world(key, 0, t, k)); per legal
candidate of record t and per world: SceneCaster cast on the 64 x 114 scoring rays (abl_score.pose_rays): first-hit
distance t (model units) + hit; DistP1 (abl_score.new_set with the level z0) -> K4 = mean over worlds, K1 = world 0.
Outputs W/fixed_state/<model>/<state>.npz (per candidate x world t / hit, scores, int8 worlds = round(127 w / TRUNC),
P1 structure) and <state>.json (record: timings, world sha256, availability).
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_frame as F  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(F.REPO, "experiments", "exp4_multiworld_belief", "task1", "scripts"))
sys.path.insert(0, os.path.join(F.REPO, "experiments", "exp4_multiworld_belief", "ablation_study", "abl_1", "common"))
import m_common as MC  # noqa: E402
import m_p1  # noqa: E402

OUT = os.path.join(F.W, "fixed_state")
TRAJ_DIR = "traj"          # protocol M; "traj_R" for the real-captured protocol R (v1.1)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def build_wm(model):
    import abl_common as AC
    import g1_common as GC
    import t1_eval as E
    E.RUN_ROOT = os.path.relpath(os.path.join(OUT, "config_runs"), F.REPO)
    if model == "C":
        if E.sha_file(GC.CKPT_C2000) != AC.CKPT_SHA:
            raise RuntimeError("INVALID: T1-C@2000 checkpoint sha256 differs from the frozen record")
        return E.build_outdoor_wm("C", GC.CKPT_C2000)
    return E.build_outdoor_wm("frozen", None)


def replay(card, traj, t, dev):
    td = os.path.join(F.W, "preprocessing", TRAJ_DIR, traj)
    recs = [MC.load_service_record(os.path.join(td, f"svc_t{i}.npz")) for i in range(t + 1)]
    fr, dims, dom, win, box = F.region_geometry(card, recs[-1]["cam_center"])
    fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
    for r in recs:
        fu.integrate_record(r)
    return recs[-1], fr, dims, dom, win, box, fu


def run_state(wm, model, st, dev):
    import abl_score as AS
    import m_score as MS
    import m_t7_planner as T7P
    import m_world as MW
    card = F.load_card(st["level_key"])
    t0 = time.time()
    rec, fr, dims, dom, win, box, fu = replay(card, st["traj"], st["t"], dev)
    key = f"T0|{st['id']}"
    MW.register_scene(wm, key, fr)
    tg = time.time()
    ws, info = T7P.generate_refresh(wm, fu, key, 4, st["t"], dims, dev)
    gen_s = time.time() - tg
    z0 = F.z0_of_card(card)
    caster = MS.SceneCaster(fr, dims, dom, fu.input_volume(), device=dev)
    n = len(rec["legal_R"])
    tt = np.full((n, len(ws), 7296), np.inf, np.float32)
    hh = np.zeros((n, len(ws), 7296), bool)
    sc = np.zeros((n, len(ws)), np.float64)
    vols = [caster.cast_volume(w.to(dev)) for w in ws]
    for j in range(n):
        rays = AS.pose_rays(caster, rec["legal_R"][j], rec["legal_T"][j])
        for k, v in enumerate(vols):
            r = caster.cast(v, *rays[:3])
            tt[j, k] = np.where(r["hit"].cpu().numpy(), r["t"].float().cpu().numpy(), np.inf)
            hh[j, k] = r["hit"].cpu().numpy()
            _, wv = AS.new_set(caster, v, rays, z0)
            sc[j, k] = float(wv.sum())
    st_arr = fu.structure().cpu().numpy().astype(np.int8)
    q = [np.round(np.clip(w.numpy() / MC.TRUNC, -1, 1) * 127).astype(np.int8) for w in ws]
    od = os.path.join(OUT, model)
    os.makedirs(od, exist_ok=True)
    np.savez_compressed(os.path.join(od, f"{st['id']}.npz"), t=tt, hit=hh, distp1_k=sc, structure=st_arr,
                        worlds_q=np.stack(q) if q else np.zeros((0,), np.int8), legal_idx=rec["legal_idx"], dims=np.asarray(dims))
    out = {"state": st, "model": model, "availability": info["availability"], "worlds": info["worlds"],
           "generation_s": gen_s, "total_s": time.time() - t0, "n_windows": win["n"], "dims": dims, "box_m": box.tolist(),
           "k_rt": fr.k_M, "z0_rt": z0, "n_legal": n, "input_sha256": info["input_sha256"],
           "distp1_K4": sc.mean(1).tolist() if len(ws) else [0.0] * n, "distp1_K1": sc[:, 0].tolist() if len(ws) else [0.0] * n,
           "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30}
    json.dump(out, open(os.path.join(od, f"{st['id']}.json"), "w"), indent=1, default=float)
    print(st["id"], model, info["availability"], f"windows {win['n']} gen {gen_s:.0f}s total {out['total_s']:.0f}s", flush=True)
    del ws, vols, caster, fu
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["frozen", "C"], required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--states", default=os.path.join(F.W, "manifests", "indoor_states.json"))
    ap.add_argument("--out", default=None, help="output root (default: the formal fixed_state dir)")
    ap.add_argument("--protocol", choices=["M", "R", "O"], default="M")
    ap.add_argument("--strata", default="", help="comma list of strata to keep (R world models: c01-05,c05-10)")
    ap.add_argument("--only", default="", help="comma list of state ids (smoke)")
    a = ap.parse_args()
    global OUT, TRAJ_DIR
    if a.protocol == "R":
        TRAJ_DIR, OUT = "traj_R", os.path.join(F.W, "fixed_state_R")
    if a.protocol == "O":
        TRAJ_DIR, OUT = "outdoor_traj", os.path.join(F.W, "fixed_state_out")
    OUT = a.out or OUT
    dev = torch.device("cuda")
    sts = json.load(open(a.states))["states"]
    if a.only:
        sts = [s for s in sts if s["id"] in a.only.split(",")]
    if a.strata:
        sts = [s for s in sts if s["stratum"] in a.strata.split(",")]
    pi, pn = (int(x) for x in a.part.split("/"))
    sts = sts[pi::pn]
    wm, info = build_wm(a.model)
    print("world model", a.model, info.get("config", {}).get("argv", "")[:3] if isinstance(info, dict) else "", flush=True)
    os.makedirs(os.path.join(OUT, a.model), exist_ok=True)
    for st in sts:
        if os.path.exists(os.path.join(OUT, a.model, f"{st['id']}.json")):
            continue
        lk = os.path.join(OUT, a.model, f"{st['id']}.lock")
        try:                                    # atomic claim: several workers may share one state list
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f"{os.uname()[1]}:{os.getpid()}".encode())   # owner, for the stale-lock reaper (run_worker.sh)
        os.close(fd)
        if os.path.exists(lk.replace(".lock", ".failed.json")):   # gave up after repeated OOM (recorded, not dropped)
            continue
        run_state(wm, a.model, st, dev)


if __name__ == "__main__":
    main()
