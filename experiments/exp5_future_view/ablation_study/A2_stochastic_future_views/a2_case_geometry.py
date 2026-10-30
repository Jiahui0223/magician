"""EXP5 Ablation A2 -- geometry of the figure cases (seen2scene env, 1 GPU).

For every case state of results/cases.json: rebuild the planner state (s2_infer.StateReplay), re-score with the same
models and noise (the per-sample DistP1 must equal the stored v2.json values bitwise -- a re-run check), and store what
the figures need beyond v2.npz: per sample the DistP1 "new" ray mask (hit in the planning domain, outside the dilated
BAND) and the 3D points of every predicted new hit, the GT new-surface points, and the observed sample points of
records 0..t. Output W/cases/<dataset>_<sid>.npz
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ABL, "common"))
import ab_common as A  # noqa: E402
import s2_infer as S2  # noqa: E402

np, torch = S2.np, S2.torch
SCO = S2.SCO


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    cores = S2.load_cores(dev)
    cases = json.load(open(os.path.join(HERE, "results", "cases.json")))
    od = os.path.join(A.W, "cases")
    os.makedirs(od, exist_ok=True)
    checks = {}
    for dataset, cs in cases.items():
        man = {s["sid"]: s for s in A.load_manifest(dataset)["states"]}
        for nm, c in cs.items():
            sid = c["sid"]
            if sid is None:
                continue
            p = os.path.join(od, f"{dataset}_{sid}.npz")
            if os.path.exists(p):
                continue
            st = man[sid]
            rp = S2.StateReplay(dataset, S2.episode_of(dataset, st), st["sources"][0]["source"], cores, dev)
            v2 = json.load(open(os.path.join(A.state_dir(dataset, sid), "v2.json")))
            orc = np.load(os.path.join(A.state_dir(dataset, sid), "oracle.npz"))
            for t, rec, geo, h, eps in rp.run([st["t"]]):
                k_M = float(geo.fr.k_M)
                origin = np.asarray(geo.fr.origin_z, np.float64)
                save = {"k_M": k_M, "origin": origin}
                for m, core in cores.items():
                    out, _, views = S2.PC.score_state(core, geo, h[m], rec["legal_R"], rec["legal_T"], eps, rp.z0, keep_hits=True)
                    same = bool(np.array_equal(np.asarray(out["distp1_k"]), np.asarray(v2["models"][m]["distp1_k"])))
                    checks[f"{dataset}|{sid}|{m}"] = same
                    if not same:
                        raise SystemExit(f"STOP: case re-score differs {dataset} {sid} {m}")
                    news = []
                    for j, (v, hh) in enumerate(zip(views, out["hits"])):
                        nk = []
                        for k in range(hh["t"].shape[0]):
                            new, _, _ = SCO.new_mask_from_depth(hh["t"][k], v["o"], v["d"], geo.dims, geo.domain, geo.caster.band_dil)
                            nk.append(new.cpu().numpy())
                            P = (v["o"][None] + v["d"][new] * hh["t"][k][new][:, None]).cpu().numpy() * k_M + origin
                            save[f"{m}_pts_c{j}_k{k}"] = P.astype(np.float32)
                        news.append(np.stack(nk))
                    save[f"{m}_new"] = np.stack(news)
                for j, v in enumerate(views):
                    ts = torch.as_tensor(orc["tstar"][j], device=v["d"].device, dtype=v["d"].dtype)
                    tt = torch.where(ts <= v["pr"]["tmax"], ts, torch.full_like(ts, float("inf")))
                    new, _, hit = SCO.new_mask_from_depth(tt, v["o"], v["d"], geo.dims, geo.domain, geo.caster.band_dil)
                    save[f"gt_new_c{j}"] = new.cpu().numpy()
                    save[f"gt_pts_c{j}"] = ((v["o"][None] + v["d"][new] * tt[new][:, None]).cpu().numpy() * k_M + origin).astype(np.float32)
                    save[f"cam_c{j}"] = (v["o"].cpu().numpy() * k_M + origin)
                obs = []
                for tt_ in range(st["t"] + 1):
                    z = np.load(A.svc_path(dataset, st["sources"][0]["source"], S2.episode_of(dataset, st), tt_))["sample_points"]
                    obs.append(np.stack([z[:, 0], -z[:, 2], z[:, 1]], 1))           # world y-up -> frame z-up runtime
                save["obs_pts"] = np.concatenate(obs).astype(np.float32)
                save["cur_cam"] = np.asarray([rec["cam_center"][0], -rec["cam_center"][2], rec["cam_center"][1]])
                np.savez_compressed(p, **save)
                print(dataset, nm, sid, "re-score equal", {m: checks[f"{dataset}|{sid}|{m}"] for m in cores}, flush=True)
    pc = os.path.join(od, "rescore_checks.json")
    old = json.load(open(pc)) if os.path.exists(pc) else {}
    json.dump(dict(old, **checks), open(pc, "w"), indent=1)


if __name__ == "__main__":
    main()
