"""Unified outdoor benchmark -- CPU-only dry check (no GPU is touched: run with CUDA_VISIBLE_DEVICES="").

  python dry_check.py                 global checks + one subprocess per arm (patches isolated per process);
                                      writes work_dirs/exp_unified_outdoor/verify/dry_check.json
  python dry_check.py --arm <arm>     the per-arm check only (prints one JSON line "RESULT {...}")

Per arm: apply the uo_common patches (fails if a patch target no longer exists), import the historical trainer, resolve
the config / model / data configs exactly the way that trainer does, check every cfg key the trainer source reads,
re-derive the config diff against the historical base config (must equal unified.changed_keys), build the model on CPU
(parameter count vs the historical run), build the data index (split, shard / TSDF-cache / label files), load one state
of a small trajectory and run one CPU forward + loss (V2 / V4 / V3-T1) or the warm-start load (V4C / V5).
Global: S10 vs D100 shard equality of every P1-independent array (180 trajectories), V5 label rows vs both shard sets,
exact-resume unit test (fake trainer, CPU generators), V2-rule identity of the V3 pooled sampler vs v2_data.ViewData and
t22_data.CompactViewData (4 ranks x 300 steps), frozen-file checks, bash -n of the launch scripts.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import uo_common as U  # noqa: E402

OUTD = os.path.join(U.WU, "verify")
HIST_PARAMS = {"v2": 24822114, "v4": 24847544, "v3t1": 23730986, "v4c": 24847544, "v5": 24988647}
V4_HIST_EMA = os.path.join(U.REPO, "work_dirs", "exp6_v4", "checkpoints", "final.pt")      # format stand-in for V4@48k
SMALL = ("liberty", None)                                                                   # small 64^3 scene for the CPU forward
# keys read by one arm of train_v5 only (the other arm's config legitimately lacks them)
V5_ONLY = {"lambda_surf", "lambda_bridge0", "bridge_end_step", "lr_new", "warmup_new", "surf_rays_per_view", "surf_rand_queries",
           "bridge_steps", "model"}


def cfg_keys_read(paths):
    """Required (cfg["k"]) and optional (cfg.get("k")) top-level keys + nested ["a"]["b"] reads in the given sources."""
    req, opt, nested = set(), set(), set()
    for p in paths:
        s = open(p).read()
        req |= set(re.findall(r'\bcfg\["(\w+)"\]', s))
        opt |= set(re.findall(r'\bcfg\.get\("(\w+)"', s))
        nested |= set(re.findall(r'\bcfg\["(\w+)"\]\["(\w+)"\]', s))
        guarded = set((b, a) for a, b in re.findall(r'"(\w+)" in cfg\["(\w+)"\]', s))   # e.g. t12_train.py:181 pisa_sub
        nested -= guarded
    return req, opt, nested


def diff_cfg(base, cfg):
    out = []
    for k in sorted(set(base) | set(cfg)):
        if k == "unified":
            continue
        if base.get(k, "<absent>") != cfg.get(k, "<absent>"):
            out.append(k)
    return out


# =============================================================================================== per-arm check
def arm_check(arm):
    import numpy as np
    import torch
    res = {"arm": arm, "ok": True, "errors": [], "info": {}}

    def err(m):
        res["ok"] = False
        res["errors"].append(m)
    A = U.ARMS[arm]
    tr = A["trainer"]
    cfg_p = A["config"]
    cfg = json.load(open(cfg_p))
    U.add_paths(tr)
    # ---- patches + trainer import
    patches = {}
    if tr == "v2":
        patches.update(U.patch_v2(cfg))
    elif tr == "v3t1":
        patches.update(U.patch_v3t1(cfg))
    elif tr == "v5":
        patches.update(U.patch_v5(cfg, os.path.join(OUTD, "dry_out", arm), check_parent=False))
    er = U.ExactResume()
    patches["exact_resume_hook"] = U.install_exact_resume(tr, er)
    mod = __import__(U.TRAINER_MOD[tr])
    res["info"]["trainer"] = os.path.relpath(mod.__file__, U.REPO)
    res["info"]["patches"] = patches
    import v2_frozen as VF
    import v2_data as VD
    from seen2scene.futureview import model as M
    # ---- config resolution the trainer's way
    if tr == "v2":
        c = VF.load_cfg(cfg_p); mcfg = VF.load_cfg(c["model"]); dcfg = VF.load_cfg(c["data"])
        srcs = [mod.__file__]
    elif tr == "v4":
        import v4_paths as P
        c = P.load_cfg(cfg_p); mcfg = P.load_cfg(c["model"]); dcfg = VF.load_cfg(c["data"])
        if not mcfg.get("ray_xattn", {}).get("enabled"):
            err("V4 model config without ray_xattn")
        srcs = [mod.__file__, os.path.join(U.REPO, "experiments/exp5_future_view/task1/scripts/t12_train.py")]
        if 48000 not in c.get("ema_save_steps", []) or 60000 not in c.get("ema_save_steps", []):
            err("V4 ema_save_steps lacks 48000 / 60000 (V4@48k is the V4C / V5 parent)")
    elif tr == "v3t1":
        c = json.load(open(os.path.join(mod.CFG_DIR, cfg_p))); mcfg = json.load(open(os.path.join(mod.CFG_DIR, c["model"]))); dcfg = None
        if mcfg["xattn"]["enabled"]:
            err("model config has xattn enabled: trainer_outdoor's assert would fail (T1 is switched on at construction)")
        srcs = [mod.__file__, os.path.join(U.REPO, "seen2scene/v3/losses.py")]       # + T0.lr_at: warmup / lr / steps (below)
    else:
        c = json.load(open(os.path.join(mod.CFG, cfg_p)))
        mcfg = json.load(open(os.path.join(mod.CFG, c["model"]))) if A["v5_arm"] == "v5" else None
        dcfg = VF.load_cfg(c["data"])
        srcs = [mod.__file__]
    assert c == cfg
    # ---- keys read by the trainer
    req, opt, nested = cfg_keys_read(srcs)
    if tr == "v3t1":
        req |= {"warmup", "lr", "steps"}                                    # seen2scene/v3/trainer.py lr_at (used by trainer_outdoor)
    if tr == "v5" and A["v5_arm"] == "v4c":
        req -= V5_ONLY
        nested = {x for x in nested if x[0] not in V5_ONLY}
    miss = sorted(k for k in req if k not in cfg)
    miss_n = sorted(f"{a}.{b}" for a, b in nested if a in cfg and isinstance(cfg[a], dict) and b not in cfg[a])
    if miss or miss_n:
        err(f"config lacks keys read by the trainer: {miss + miss_n}")
    res["info"]["keys_read"] = {"required": sorted(req), "optional_present": sorted(k for k in opt if k in cfg)}
    # ---- config diff vs the historical base (must equal the documented list)
    base = json.load(open(os.path.join(U.REPO, cfg["unified"]["base_config"])))
    d = diff_cfg(base, cfg)
    doc = sorted(x["key"] for x in cfg["unified"]["changed_keys"])
    if d != doc:
        err(f"undocumented config change: diff {d} vs documented {doc}")
    if U.sha256_file(os.path.join(U.REPO, cfg["unified"]["base_config"])) != cfg["unified"]["base_config_sha256"]:
        err("historical base config changed since make_configs.py")
    res["info"]["changed_keys"] = doc
    # ---- model on CPU
    torch.manual_seed(0)
    if tr in ("v2", "v4"):
        model = M.V2Model(mcfg)
    elif tr == "v3t1":
        from seen2scene.v3 import trainer as T0
        model = T0.make_model(mcfg, cfg["seed"], torch.device("cpu"))          # patched: xattn on
        if not model.xa:
            err("V3 model built without cross-attention")
    else:
        ck4 = torch.load(V4_HIST_EMA, map_location="cpu", weights_only=False)
        if A["v5_arm"] == "v4c":
            model = M.V2Model(ck4["model_cfg"]); model.load_state_dict(ck4["ema"])
        else:
            from seen2scene.futureview import v5_model as V5
            model = V5.V5Model(mcfg)
            ws = V5.load_v4_warmstart(model, ck4["ema"])
            res["info"]["v5_warmstart"] = {k: v for k, v in ws.items() if k != "not_used_from_v4"} if isinstance(ws, dict) else None
        res["info"]["parent_format_standin"] = {"path": os.path.relpath(V4_HIST_EMA, U.REPO), "keys": sorted(ck4), "step": ck4.get("step")}
    npar = sum(p.numel() for p in model.parameters())
    res["info"]["params"] = npar
    hp = HIST_PARAMS[A.get("v5_arm") or tr] if tr == "v5" else HIST_PARAMS[tr]
    if npar != hp:
        err(f"parameter count {npar} != historical {hp}")
    # ---- data index
    train, held = VD.load_split(cfg.get("train_set", "train6_minus_Sc"))
    root = U.rp(cfg["data_root"])
    miss_sh = [x for x in train + held if not os.path.exists(os.path.join(root, x[0], f"{x[1]}.npz"))]
    res["info"]["split"] = {"train": len(train), "held": len(held), "data_root": cfg["data_root"], "missing_shards": len(miss_sh)}
    if miss_sh or len(train) != 138 or len(held) != 42:
        err(f"split / shards: train {len(train)} held {len(held)} missing {miss_sh[:3]}")
    if tr == "v5":
        par = cfg["unified"]["parent"]
        if par["ckpt"] != f"work_dirs/exp_unified_outdoor/runs/{A['parent_arm']}/ema_{U.PARENT_STEP:06d}.pt":
            err("parent path does not match the V4 arm's EMA naming")
        res["info"]["parent"] = {**par, "exists_now": os.path.exists(U.rp(par["ckpt"]))}
        if A["v5_arm"] == "v5":
            lab = U.rp(U.V5_LABELS)
            ml = [x for x in train + held if not os.path.exists(os.path.join(lab, x[0], f"{x[1]}.npz"))]
            res["info"]["labels_missing"] = len(ml)
            if ml:
                err(f"V5 labels missing: {ml[:3]}")
    if tr == "v3t1":
        from seen2scene.v3 import data as D
        hist_tr = set(map(tuple, D.items("outdoor", "train"))); hist_sc = set(map(tuple, D.items("outdoor", "sc")))
        if hist_tr != set(train) or hist_sc != set(held):
            err("V3 historical outdoor listing != V2 frozen split")
        nc = sum(os.path.exists(D.cache_path("outdoor", *x)) for x in train + held)
        res["info"]["tsdf_cache"] = {"root": cfg["tsdf_cache_root"], "present": nc, "needed": len(train) + len(held)}
        if nc < len(train) + len(held):
            res["info"]["tsdf_cache"]["status"] = "MISSING (build with build_tsdf_cache_s10.sh before launching)" if arm.endswith("s10") else "INCOMPLETE"
            if not arm.endswith("s10"):
                err("D100 TSDF cache incomplete")
    # ---- one CPU forward on a small trajectory (V2 / V4 / V3-T1)
    sm = [x for x in train if x[0] == SMALL[0]][:1]
    dev = torch.device("cpu")
    if tr in ("v2", "v4"):
        if tr == "v2":
            data = VD.ViewData(sm, log=lambda *x: None)                  # patched default root
        else:
            import t22_data as TD
            data = TD.CompactViewData(sm, root=root, log=lambda *x: None)
        rng = np.random.default_rng([cfg["seed"], 0])
        b = data.sample_batch(rng, 8, 1, dcfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], data.state_weights())
        model.rays_ckpt = False
        with torch.no_grad():
            bt = VD.build_cond(model, data, b, dev, autocast=False)
            V = bt["cond"].shape[0]
            es, nr = VD.fm_sum(model, bt, torch.randn(V, 2, M.GH, M.GW_PAD), torch.rand(V), autocast=False)
        loss = float(es / (2 * nr.clamp(min=1)))
        res["info"]["cpu_forward"] = {"traj": "/".join(sm[0]), "views": V, "labelled_rays": float(nr), "fm_loss": loss}
        if not np.isfinite(loss):
            err("non-finite CPU loss")
    if tr == "v3t1":
        from seen2scene.v3 import data as D
        from seen2scene.v3 import losses as L
        if os.path.exists(D.cache_path("outdoor", *sm[0])):
            data = D.V3Views("outdoor", sm, lambda *x: None)
            rng = np.random.default_rng([cfg["seed"], 0])
            b = data.sample_batch(rng, 8, 1, cfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], data.state_weights())
            with torch.no_grad():
                bt = D.build_cond(model, data, b, dev, autocast=False)
                V = bt["cond"].shape[0]
                es, nr = L.fm_terms(model, bt, torch.randn(V, 2, M.GH, M.GW_PAD), torch.rand(V), autocast=False)
                rec, _ = L.recon_terms(model, bt["vae"], autocast=False)
                kl, _ = L.kl_terms(bt["vae"])
            res["info"]["cpu_forward"] = {"traj": "/".join(sm[0]), "views": V, "fm": float(es / (2 * nr.clamp(min=1))), "recon": float(rec),
                                          "kl": float(kl), "mem_tokens": None if bt["mem"] is None else [list(t.shape) for t in bt["mem"]]}
            if not all(np.isfinite(v) for v in (float(es), float(rec), float(kl))):
                err("non-finite V3 CPU loss")
        else:
            res["info"]["cpu_forward"] = "skipped (TSDF cache of this density not built yet)"
    return res


# =============================================================================================== global checks
def check_s10_d100_equal():
    import numpy as np
    keys = ("t", "a", "R", "T", "jitter", "k_M", "dims", "heldout", "tstar")
    s10, d100 = U.rp(U.DATA_ROOT["s10"]), U.rp(U.DATA_ROOT["d100"])
    lab = U.rp(U.V5_LABELS)
    files = sorted(glob.glob(os.path.join(s10, "*", "*.npz")))
    bad, lab_bad, n = [], [], 0
    for p in files:
        s, tj = p.split("/")[-2], os.path.basename(p)[:-4]
        a, b = np.load(p), np.load(os.path.join(d100, s, f"{tj}.npz"))
        for k in keys:
            if not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"):
                bad.append(f"{s}/{tj}:{k}")
        lp = os.path.join(lab, s, f"{tj}.npz")
        if os.path.exists(lp):
            z = np.load(lp)
            if not (np.array_equal(z["t"], a["t"]) and np.array_equal(z["a"], a["a"])):
                lab_bad.append(f"{s}/{tj}")
        else:
            lab_bad.append(f"{s}/{tj}: missing")
        n += 1
    return {"trajectories": n, "keys": list(keys), "mismatch": bad[:20], "n_mismatch": len(bad), "label_rows_mismatch": lab_bad[:20],
            "ok": n == 180 and not bad and not lab_bad}


def check_exact_resume():
    """Fake trainer main() (same local names as the historical trainers) with a CPU torch generator in place of the CUDA
    RNG: uninterrupted 10 steps vs 7 steps + kill + resume at the last checkpoint (6) with the trainers' re-seed rule."""
    import numpy as np
    import torch
    base = os.path.join(OUTD, "dry_exact_resume")
    import shutil
    shutil.rmtree(base, ignore_errors=True)
    out = {}

    def run(out_dir, steps, step0, ckpt_every, mod, draws, kill_after=None):
        def main():                                                  # noqa: F811  (frame name must be "main")
            a = types.SimpleNamespace(out=out_dir, smoke="none")
            cfg = {"ckpt_every": ckpt_every, "seed": 0}
            rank, world, ddp = 0, 1, False
            rng = np.random.default_rng([0, rank])
            gen = torch.Generator(); gen.manual_seed(7919)
            torch.manual_seed(0)
            if step0:
                rng = np.random.default_rng([0, rank, step0])        # the trainers' DDP re-seed rule
                gen.manual_seed(7919 + 13 * step0); torch.manual_seed(7 * step0)
            for step in range(step0, steps):
                mod.lr_at(step, cfg)
                draws.append((step, int(rng.integers(1 << 30)), float(torch.rand(1)), float(torch.rand(1, generator=gen))))
                if kill_after is not None and step == kill_after:
                    return
            del a, world, ddp
        main()
    for exact in (True, False):
        d = os.path.join(base, "exact" if exact else "fallback")
        mod = types.SimpleNamespace(lr_at=None)
        er = U.ExactResume(log=lambda *x: None); mod.lr_at = er.wrap(lambda step, cfg: 0.0)
        A_ = []
        run(os.path.join(d, "A"), 10, 0, 3, mod, A_)
        er = U.ExactResume(log=lambda *x: None); mod.lr_at = er.wrap(lambda step, cfg: 0.0)
        B1 = []
        run(os.path.join(d, "B"), 10, 0, 3, mod, B1, kill_after=7)
        if not exact:
            for f in glob.glob(os.path.join(d, "B", "rng_states", "s0000006_r0.pt")):
                os.remove(f)
        er = U.ExactResume(log=lambda *x: None); mod.lr_at = er.wrap(lambda step, cfg: 0.0)
        B2 = []
        run(os.path.join(d, "B"), 10, 6, 3, mod, B2)
        same = A_[6:] == B2
        log = [json.loads(x) for x in open(os.path.join(d, "B", "resume_log_r0.jsonl"))]
        out["exact" if exact else "fallback"] = {"resumed_equal_uninterrupted": same, "resume_log": log[-1]}
    out["ok"] = out["exact"]["resumed_equal_uninterrupted"] and out["exact"]["resume_log"]["exact"] and \
        (not out["fallback"]["resumed_equal_uninterrupted"]) and (not out["fallback"]["resume_log"]["exact"])
    return out


def check_sampler_identity():
    """V2 rule identity: v2_data.ViewData (V2), t22_data.CompactViewData (V4 / V5) and the V3 pooled sampler (V3-T1, D100
    cache) draw the same per-rank batches from the same rng; the V3 micro-batches concatenate to the V2 batch."""
    import numpy as np
    U.add_paths("v3t1")
    import v2_data as VD
    import t22_data as TD
    from seen2scene.v3 import data as D
    train, _ = VD.load_split("train6_minus_Sc")
    items = [x for x in train if x[0] in ("liberty", "redeemer")][:6]
    root = U.rp(U.DATA_ROOT["d100"])
    q = lambda *x: None  # noqa: E731
    v2 = VD.ViewData(items, root=root, log=q)
    v4 = TD.CompactViewData(items, root=root, log=q)
    P = U.pooled_v3_views_class(D)
    v3 = P("outdoor", items, q, accum=2)
    same_states = v2.states == v4.states == v3.states
    w = v2.state_weights(20, 2.0)
    n_eq_v4, n_eq_v3, parts_max, n_states_part, n = 0, 0, [], [], 0
    for rank in range(4):
        r2, r4, r3 = (np.random.default_rng([0, rank]) for _ in range(3))
        for _ in range(300):
            b2 = v2.sample_batch(r2, 16, 2, 8, 0.5, 9, w)
            b4 = v4.sample_batch(r4, 16, 2, 8, 0.5, 9, w)
            m1 = v3.sample_batch(r3, 8, 1, 8, 0.5, 9, w)
            m2 = v3.sample_batch(r3, 8, 1, 8, 0.5, 9, w)
            key = lambda b: [(x[0], x[1], x[2], tuple(x[3].tolist())) for x in b]  # noqa: E731
            n_eq_v4 += key(b2) == key(b4)
            n_eq_v3 += key(b2) == key(m1 + m2)
            parts_max.append(max(sum(len(x[3]) for x in m1), sum(len(x[3]) for x in m2)))
            n_states_part.append(max(len(m1), len(m2)))
            n += 1
    return {"trajectories": ["/".join(x) for x in items], "states_identical": same_states, "draws": n, "v4_equal_v2": n_eq_v4,
            "v3_concat_equal_v2": n_eq_v3, "v3_micro_max_views_hist": {int(k): int(v) for k, v in zip(*np.unique(parts_max, return_counts=True))},
            "v3_micro_max_states_hist": {int(k): int(v) for k, v in zip(*np.unique(n_states_part, return_counts=True))},
            "ok": same_states and n_eq_v4 == n and n_eq_v3 == n}


def check_frozen():
    U.add_paths("v4")
    import v2_frozen as VF
    out = {"v2_frozen.verify_frozen": None}
    try:
        out["v2_frozen.verify_frozen"] = VF.verify_frozen()
    except Exception as e:                       # noqa: BLE001
        out["v2_frozen.verify_frozen"] = f"FAIL {e}"
    fz = json.load(open(os.path.join(U.REPO, "experiments/exp6_v4/configs/FROZEN_SHA256.json")))["records"][-1]["files"]
    ch = [k for k, v in fz.items() if U.sha256_file(os.path.join(U.REPO, k)) != v]
    out["exp6_v4_frozen_changed_code_or_config"] = [k for k in ch if not k.endswith(".md")]
    out["exp6_v4_frozen_changed_docs_warning"] = [k for k in ch if k.endswith(".md")]       # documentation only: warning
    out["ok"] = out["v2_frozen.verify_frozen"] is True and not out["exp6_v4_frozen_changed_code_or_config"]
    return out


def check_bash():
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, "*.sh"))):
        r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
        out[os.path.basename(p)] = "ok" if r.returncode == 0 else r.stderr.strip()
    out["ok"] = all(v == "ok" for v in out.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        sys.exit("run with CUDA_VISIBLE_DEVICES='' (CPU only)")
    if a.arm:
        try:
            r = arm_check(a.arm)
        except BaseException as e:               # noqa: BLE001
            import traceback
            r = {"arm": a.arm, "ok": False, "errors": [f"{type(e).__name__}: {e}", traceback.format_exc()[-2000:]], "info": {}}
        print("RESULT " + json.dumps(r, default=str), flush=True)
        return
    os.makedirs(OUTD, exist_ok=True)
    t0 = time.time()
    rep = {"time": time.strftime("%F %T"), "host": os.uname()[1], "arms": {}, "global": {}}
    for arm in U.ARMS:
        ts = time.time()
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--arm", arm], capture_output=True, text=True, env=os.environ.copy())
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
        r = json.loads(line[-1][7:]) if line else {"arm": arm, "ok": False, "errors": [p.stderr[-3000:]], "info": {}}
        r["seconds"] = round(time.time() - ts, 1)
        rep["arms"][arm] = r
        print(f"[arm] {arm:26s} {'OK ' if r['ok'] else 'FAIL'} {r['seconds']:6.1f}s  params {r['info'].get('params')}  "
              f"changed {r['info'].get('changed_keys')}  " + (f"errors {r['errors']}" if r["errors"] else ""), flush=True)
        for k in ("split", "tsdf_cache", "parent", "cpu_forward", "labels_missing"):
            if k in r["info"]:
                print(f"         {k}: {json.dumps(r['info'][k], default=str)}", flush=True)
    for name, fn in (("frozen_files", check_frozen), ("bash_syntax", check_bash), ("exact_resume_unit_test", check_exact_resume),
                     ("sampler_identity", check_sampler_identity), ("s10_vs_d100_and_labels", check_s10_d100_equal)):
        ts = time.time()
        try:
            r = fn()
        except Exception as e:                   # noqa: BLE001
            import traceback
            r = {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}
        r["seconds"] = round(time.time() - ts, 1)
        rep["global"][name] = r
        print(f"[global] {name:26s} {'OK ' if r.get('ok') else 'FAIL'} {json.dumps(r, default=str)[:900]}", flush=True)
    rep["all_ok"] = all(r["ok"] for r in rep["arms"].values()) and all(r.get("ok") for r in rep["global"].values())
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(OUTD, "dry_check.json"), "w"), indent=1, default=str)
    print(f"ALL_OK {rep['all_ok']}  ({rep['seconds']} s) -> {os.path.relpath(os.path.join(OUTD, 'dry_check.json'), U.REPO)}")
    sys.exit(0 if rep["all_ok"] else 1)


if __name__ == "__main__":
    main()
