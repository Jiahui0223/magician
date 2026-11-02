"""EXP6 V5 -- G1 / G2 on the development fixed states (seen2scene env, 1 GPU per process): exp6_v3/shared/eval_fixed.py
run_episode unchanged (same states, candidates, noise, D100 map for the model inputs, sparse DistP1 scoring, eval-side
t* errors) with the cores B2 (frozen V2-Mix, bitwise replay check against the E2 rows), V5 (V5-Warm12k final EMA through
the V2 planner core, v5_core.install) and V4C (V4-Continue12k final EMA). Rows: work_dirs/exp6_v5/eval_fixed/<set>_<tag>/
<stratum>/<tag>.jsonl (merged with the V4 rows of exp6_v4 by sid in the analysis).
Belief dump (evaluation side, V5 only): for every scored state and every legal candidate view, 256 distinct rays (up to 128
of class UNKNOWN, the rest uniform over the other rays; seed from the state's legal poses and the view index) -> the V5 ray latent z
(fp16), the model-frame range [0, min(tcap, grid-box exit)], the ray indices / classes and the runtime camera (R, T, k_M)
-> work_dirs/exp6_v5/eval_fixed/<set>_<tag>/<stratum>/belief_<episode>.npz. Multi-surface GT of these rays is built
afterwards from the evaluation meshes (eval_belief_gt.py, magician env) and only used for metrics.
Usage: python eval_g1_g2.py --set valsel|dev3 --stratum B0|B2 [--v5 CKPT] [--v4c CKPT] [--tag s_final] [--part i/n]"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v3", "shared"))
import eval_fixed as EF  # noqa: E402

X, E2F = EF.X, EF.E2F
W = os.path.join(REPO, "work_dirs", "exp6_v5", "eval_fixed")
N_UNK, N_ALL = 128, 128
DUMP = {"core": None, "rows": [], "tag": None}


def legal_key(legal_R, legal_T):
    import numpy as np
    h = hashlib.sha256()
    for a in (legal_R, legal_T):
        for x in a:
            h.update(np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, np.float64).tobytes())
    return h.hexdigest()[:16]


def install_dump():
    np, torch, PC, G = X.np, X.torch, X.PC, X.G
    orig = PC.score_state

    def score_state(core, geo, h, legal_R, legal_T, eps, z0, *a, **k):
        res = orig(core, geo, h, legal_R, legal_T, eps, z0, *a, **k)
        if DUMP["core"] is not None and core is DUMP["core"]:
            views = res[2]
            key = legal_key(legal_R, legal_T)
            m = core.core.model
            ext = torch.as_tensor(np.asarray(geo.ext.cpu() if hasattr(geo.ext, "cpu") else geo.ext, np.float64), device=views[0]["o"].device).float()
            with torch.no_grad():
                for j, v in enumerate(views):
                    o, d, tc = v["o"].float(), v["d"].float(), v["tcap"].float()
                    ch = G.channels(v["pr"], v["tcap"], v["d"], v["o"]).float()
                    cls = v["pr"]["cls"]
                    g = torch.Generator(device="cpu"); g.manual_seed(int(hashlib.sha256(f"exp6-V5|belief|{key}|{j}".encode()).hexdigest()[:15], 16))
                    unk = torch.nonzero(cls.cpu() == G.CLS_UNKNOWN)[:, 0]
                    pu = unk[torch.randperm(len(unk), generator=g)[:N_UNK]]
                    free = torch.ones(d.shape[0], dtype=torch.bool); free[pu] = False
                    others = torch.nonzero(free)[:, 0]
                    rest = others[torch.randperm(len(others), generator=g)[:N_UNK + N_ALL - len(pu)]]
                    idx = torch.cat([pu, rest]).to(d.device)                      # always 256 distinct rays per view
                    oo = (o[None] if o.dim() == 1 else o).expand(len(idx), 3)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, aux, _ = m.ray_features(h, oo, d[idx], tc[idx], ch.t()[idx] if ch.shape[0] == 11 else ch[idx], return_aux=True)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        dn, on, en = d[idx].double().cpu().numpy(), oo.double().cpu().numpy(), ext.double().cpu().numpy()
                        t1, t2 = (0.0 - on) / dn, (en[None] - on) / dn
                        tbox = np.where(dn > 0, t2, np.where(dn < 0, t1, np.inf)).min(1)
                    DUMP["rows"].append({"key": key, "view": j, "R": np.asarray(legal_R[j].cpu() if hasattr(legal_R[j], "cpu") else legal_R[j], np.float64),
                                         "T": np.asarray(legal_T[j].cpu() if hasattr(legal_T[j], "cpu") else legal_T[j], np.float64), "k_M": float(geo.fr.k_M),
                                         "idx": idx.cpu().numpy().astype(np.int16), "n_unk_sel": int(len(pu)), "cls": cls[idx].cpu().numpy().astype(np.int8),
                                         "z": aux["z"].float().cpu().numpy().astype(np.float16), "tcap": tc[idx].cpu().numpy(), "tbox": tbox.astype(np.float32),
                                         "probe_logit": aux["probe_logit"].float().cpu().numpy().astype(np.float16)})
        return res
    PC.score_state = score_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["valsel", "dev3"], required=True)
    ap.add_argument("--stratum", choices=["B0", "B2"], default="B0")
    ap.add_argument("--v5", default=None)
    ap.add_argument("--v4c", default=None)
    ap.add_argument("--tag", default="s_final")
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
    import v5_core
    v5_core.install()
    torch, np = X.torch, X.np
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    v2 = X.S2.load_cores(dev)["V2M"]
    assert a.v5 or a.v4c
    cores = {"B2": EF.V2Core(v2)}
    if a.v5:
        c5 = X.PC.Core(a.v5, dev)
        assert not hasattr(c5.model, "rays"), "V5 core must not carry the old ray encoder"
        cores["V5"] = EF.V2Core(c5)
        DUMP["core"] = cores["V5"]
        install_dump()
    if a.v4c:
        c4 = X.PC.Core(a.v4c, dev)
        assert c4.model.ray_xattn is not None
        cores["V4C"] = EF.V2Core(c4)
    ref = {}
    for f in glob.glob(os.path.join(E2F.EP.W, "fixed", f"{a.set}_s5000", a.stratum, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ref[r["sid"]] = r["scores"]["B2"]
    pi, pn = map(int, a.part.split("/"))
    od = os.path.join(W, f"{a.set}_{a.tag}", a.stratum)
    os.makedirs(od, exist_ok=True)
    json.dump({k: (cores[k].ckpt_sha if k in cores else None) for k in ("B2", "V5", "V4C")} | {"v5": a.v5, "v4c": a.v4c},
              open(os.path.join(od, f"models_{pi}.json"), "w"), indent=1)
    for tag, sts in E2F.states(a.set, a.stratum)[pi::pn]:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        DUMP["rows"] = []
        rows = EF.run_episode(a.set, a.stratum, tag, sts, cores, dev, ref)
        if DUMP["core"] is not None:
            # map every dumped state to its sid through the legal poses of its service record
            keymap = {}
            rp = E2F.Rep(a.set, sts[0], dev)
            for s in sts:
                rec = X.MC.load_service_record(rp.svc(s["t"]))
                keymap[legal_key(rec["legal_R"], rec["legal_T"])] = s
            out = {"sid": [], "t": [], "view": [], "is_cand": []}
            arr = {k: [] for k in ("R", "T", "k_M", "idx", "n_unk_sel", "cls", "z", "tcap", "tbox", "probe_logit")}
            for r in DUMP["rows"]:
                s = keymap[r["key"]]
                out["sid"].append(s["sid"]); out["t"].append(s["t"]); out["view"].append(r["view"]); out["is_cand"].append(r["view"] in s["cand"])
                for k in arr:
                    arr[k].append(r[k])
            np.savez_compressed(os.path.join(od, f"belief_{tag}.npz"), sid=np.asarray(out["sid"]), t=np.asarray(out["t"]), view=np.asarray(out["view"]),
                                is_cand=np.asarray(out["is_cand"]), scene=np.asarray([sts[0]["scene"]] * len(out["sid"])),
                                **{k: np.stack([np.asarray(x) for x in v]) for k, v in arr.items()})
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(a.set, a.stratum, tag, len(rows), "%.0fs" % (time.time() - tb), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
