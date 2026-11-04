"""EXP6 V5 / F1 -- G1 / G2 on the Dev-3 fixed states (seen2scene env, 1 GPU per process).
Evaluator = exp5 task3 e2_pilot/scripts/e2_fixed.run_episode, unchanged (same E1.2 Dev-3 states of strata B0 / B2, legal
candidates, noise keys sha('exp5-T3E2'...), replay rules, oracle, eval-side t* truth), with its model table replaced:
every model runs on the S10 (D10) map, DistP1 with the sparse 10 % band (B0 protocol):
  B0   frozen V2-Mix EMA@17.5k (parent; the Task 2 'V2M' core)        A2k / A5k  F1-A EMA@2000 / @5000
  B2k / B5k  F1-B EMA@2000 / @5000 (V5 arch via exp6_v5/scripts/v5_core.install(), grid_sample guard on)
Regression: B0 scores must equal the E2 rows (e2/fixed/dev3_s5000/<stratum>, scores.B0) bitwise, else STOP.
Extra G1 counts per state (summed over candidates and the K = 4 worlds; e2_fixed.pred_groups extended, evaluation side
only): hit TP / FP / FN / TN on UNKNOWN rays and on all rays, Brier sum and reliability bins of p_hit = mean_k hit_k
(UNKNOWN rays), 2-d histogram GT depth x abs depth error of UNKNOWN true-hit rays (log10 bins).
Output work_dirs/exp6_v5/F1/eval_fixed/dev3/<stratum>/<tag>.jsonl (+ models.json). Resumable (existing tags skipped).
Usage: python eval_g12.py --stratum B0|B2 [--part i/n]"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(F1)))
sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts"))
import e2_fixed as E2F  # noqa: E402

X = E2F.X
W = os.path.join(R, "work_dirs", "exp6_v5", "F1")
RUNS = {"A": os.path.join(W, "runs", "A"), "B": os.path.join(W, "runs", "B")}
MODELS = {"B0": ("B0", 10), "A2k": ("A2k", 10), "A5k": ("A5k", 10), "B2k": ("B2k", 10), "B5k": ("B5k", 10)}
DB = (-2.0, 3.0, 10)        # log10 GT depth bins (model units)
EB = (-4.0, 2.0, 12)        # log10 abs error bins


def hist2(x, y):
    np = X.np
    ix = np.clip(((np.log10(np.maximum(x, 1e-12)) - DB[0]) / (DB[1] - DB[0]) * DB[2]).astype(int), 0, DB[2] - 1)
    iy = np.clip(((np.log10(np.maximum(y, 1e-12)) - EB[0]) / (EB[1] - EB[0]) * EB[2]).astype(int), 0, EB[2] - 1)
    h = np.zeros((DB[2], EB[2]), np.int64)
    np.add.at(h, (ix, iy), 1)
    return h


def install_pred_groups():
    orig = E2F.pred_groups

    def pred_groups(t_k, tgt, dz, k_M, u, ok, new, unk):
        np = X.np
        out = orig(t_k, tgt, dz, k_M, u, ok, new, unk)
        zk = t_k.double().cpu().numpy() * k_M * dz[None]
        zg = tgt * k_M * dz
        ph = np.isfinite(zk)                                   # [K, R] predicted hit
        for nm, m in (("unk", unk), ("all", np.ones_like(ok))):
            out[f"tp_{nm}"] = int((ph & ok[None] & m[None]).sum())
            out[f"fp_{nm}"] = int((ph & ~ok[None] & m[None]).sum())
            out[f"fn_{nm}"] = int((~ph & ok[None] & m[None]).sum())
            out[f"tn_{nm}"] = int((~ph & ~ok[None] & m[None]).sum())
        p = ph.mean(0)
        y = ok.astype(float)
        out["brier_sum_unk"] = float(((p - y) ** 2)[unk].sum())
        out["n_rays_unk"] = int(unk.sum())
        for i, pv in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
            mb = unk & (np.abs(p - pv) < 1e-6)
            out[f"rel_n_{i}"], out[f"rel_pos_{i}"] = int(mb.sum()), int((mb & ok).sum())
        both = ph & ok[None] & unk[None]
        if both.any():
            e = np.abs(zk - np.where(ok, zg, 0)[None]) / u
            kk, rr = np.nonzero(both)
            out["h2_unk"] = hist2(zg[rr], e[kk, rr])          # ndarray: e2_fixed.state sums candidates with + (elementwise)
        return out
    E2F.pred_groups = pred_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", choices=["B0", "B2"], required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--sub", default="dev3", help="output sub-directory (smoke runs use another one)")
    a = ap.parse_args()
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(E2F.EP.EXP5, "task2", "scripts"))
    sys.path.insert(0, os.path.join(R, "experiments", "exp6_v5", "scripts"))
    import v5_core
    v5_core.install()
    torch, np = X.torch, X.np
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    want = a.models.split(",")
    E2F.MODELS = {k: v for k, v in MODELS.items() if k in want}
    assert "B0" in E2F.MODELS, "B0 is required (regression reference)"
    cores = {"B0": X.S2.load_cores(dev)["V2M"]}
    ck = {"A2k": ("A", 2000), "A5k": ("A", 5000), "B2k": ("B", 2000), "B5k": ("B", 5000)}
    for m in E2F.MODELS:
        if m != "B0":
            arm, step = ck[m]
            cores[m] = X.PC.Core(os.path.join(RUNS[arm], f"ema_{step:06d}.pt"), dev)
            assert (arm == "B") == (not hasattr(cores[m].model, "rays")), (m, "architecture mismatch")
    install_pred_groups()
    ref = {}
    for f in glob.glob(os.path.join(E2F.EP.W, "fixed", "dev3_s5000", a.stratum, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ref[r["sid"]] = r["scores"]["B0"]
    assert ref, "no E2 reference rows"
    pi, pn = map(int, a.part.split("/"))
    od = os.path.join(W, "eval_fixed", a.sub, a.stratum)
    os.makedirs(od, exist_ok=True)
    json.dump({m: {"sha256": cores[m].ckpt_sha, "ckpt": getattr(cores[m], "ckpt", None)} for m in E2F.MODELS},
              open(os.path.join(od, f"models_{pi}.json"), "w"), indent=1, default=str)
    for tag, sts in E2F.states("dev3", a.stratum)[pi::pn]:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        rows = E2F.run_episode("dev3", a.stratum, tag, sts, cores, dev, None)
        for r in rows:
            r["repro_B0_eq_E2"] = r["scores"]["B0"] == ref.get(r["sid"])
            if not r["repro_B0_eq_E2"]:
                raise SystemExit(f"STOP: B0 replay differs from the E2 rows ({r['sid']})")
            for m in r["pred"]:
                if "h2_unk" in r["pred"][m]:
                    r["pred"][m]["h2_unk"] = np.asarray(r["pred"][m]["h2_unk"]).tolist()
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print("dev3", a.stratum, tag, len(rows), "%.0fs" % (time.time() - tb), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
