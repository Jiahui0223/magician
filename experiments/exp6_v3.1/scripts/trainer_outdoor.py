"""EXP6 V3.1 -- scratch Outdoor-only training of the V3-T0 architecture (Simple 3D VAE + latent ray encoder + V2 2D
rectified flow, no cross-attention): seen2scene/v3/trainer.py (the frozen V3 loop) with the indoor domain removed.

Identical to the V3 trainer: model construction from seed (seen2scene.v3.model.V3Model, xattn disabled), losses
(seen2scene.v3.losses: L_FM global labelled-ray mean + lambda_rec L_recon + beta(step) KL), sampler rules
(seen2scene.v3.data.V3Views), DDP gradient sum by hand, AdamW + warmup / cosine, EMA, atomic resumable checkpoints,
development metrics (seen2scene.v3.eval_dev). Differences (exp6_v3.1/FROZEN_PROTOCOL.md):
  data     Macarons++ Train-6 minus S-c only (E2 D100 outdoor shards + 4x TSDF cache); S-c for monitoring; no MP3D
  batch    per rank batch_views / world views from states_per_batch / world states, as `accum` micro-batches
           (8 views / 1 state each); L_FM normalised by the labelled-ray count of the WHOLE global step (counted from
           the data before the forward and all-reduced), L_recon / KL by world x accum -> the gradient of the
           64-view / 8-state global batch
  ckpt     EMA at ema_save_steps and at every evaluation; ckpt_last every ckpt_every; fixed EMA @ steps as the result
  stop     no early stopping (the V3 rule needed the indoor selection set); non-finite loss -> STOP
torchrun --standalone --nproc_per_node 4 experiments/exp6_v3.1/scripts/train.py --out DIR [--smoke overfit|short|profile]
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import subprocess
import time

import numpy as np
import torch

from seen2scene import PROJECT_ROOT
from seen2scene.futureview import geometry as G
from seen2scene.futureview import model as M
from seen2scene.v3 import data as D
from seen2scene.v3 import losses as L
from seen2scene.v3 import model as V3
from seen2scene.v3 import trainer as T0
from seen2scene.v3 import vae as VAE

REPO = str(PROJECT_ROOT)
CFG_DIR = os.path.join(REPO, "experiments", "exp6_v3.1", "configs")
sha256_file, sha256_obj, lr_at, jl, save_ckpt = T0.sha256_file, T0.sha256_obj, T0.lr_at, T0.jl, T0.save_ckpt


def git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO)
        return {"sha": sha, "dirty": bool(diff), "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None}
    except Exception as e:                       # noqa: BLE001
        return {"error": str(e)}


def load_data(rank, world, smoke, say):
    out_tr, out_sc = D.items("outdoor", "train"), D.items("outdoor", "sc")
    have = lambda it: [x for x in it if os.path.exists(D.cache_path("outdoor", *x))]  # noqa: E731
    out_tr, out_sc = have(out_tr), have(out_sc)
    if smoke:
        out_tr, out_sc = out_tr[:6], (out_sc[:3] or out_tr[-1:])
    n_all = {"out_train": len(out_tr), "out_sc": len(out_sc)}
    out_tr = sorted(out_tr)[rank::world]
    if rank != 0:
        out_sc = out_sc[:1]
    say(f"rank 0 loading outdoor {len(out_tr)} train / {len(out_sc)} S-c trajectories (world {world}); indoor: none")
    return {"out_train": D.V3Views("outdoor", out_tr, say), "out_sc": D.V3Views("outdoor", out_sc, say)}, n_all


def labelled_rays(data, batch):
    """Number of supervised rays (UNKNOWN with a non-zero label) of a batch, from the shard arrays (no GPU)."""
    n = 0
    for _, _, _, rows in batch:
        n += int(((data.v["cls"][rows] == G.CLS_UNKNOWN) & (data.v["b"][rows] != 0)).sum())
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="outdoor_train.json")
    ap.add_argument("--smoke", choices=["none", "overfit", "short", "profile"], default="none")
    ap.add_argument("--override", nargs="*", default=[], help="test-only config overrides key=value (steps, ckpt_every, eval_every, eval_views)")
    a = ap.parse_args()
    import torch.distributed as dist
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=90))
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = json.load(open(os.path.join(CFG_DIR, a.config)))
    for kv in a.override:
        k, v = kv.split("=")
        if k == "eval_views":
            cfg["eval_sets"] = {kk: int(v) for kk in cfg["eval_sets"]}
        else:
            cfg[k] = type(cfg[k])(v)
    mcfg = json.load(open(os.path.join(CFG_DIR, cfg["model"])))
    assert not mcfg["xattn"]["enabled"], "V3.1 is the T0 architecture (no cross-attention)"
    os.makedirs(a.out, exist_ok=True)
    log_p, ev_p = os.path.join(a.out, "log.jsonl"), os.path.join(a.out, "eval.jsonl")
    say = (lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)) if rank == 0 else (lambda *x: None)  # noqa: E731
    data, n_all = load_data(rank, world, a.smoke != "none", say)
    say("views:", {k: v.n_views for k, v in data.items()}, "cache MB:", {k: round(v.cache_bytes / 1e6) for k, v in data.items()})
    es = cfg["eval_sets"]
    sets = {"out_sc": (data["out_sc"], data["out_sc"].fixed_states("exp6-V3|eval|out_sc", es["out_sc"], per_scene=True)),
            "out_train": (data["out_train"], data["out_train"].fixed_states("exp6-V3|eval|out_train", es["out_train"]))}
    model = T0.make_model(mcfg, cfg["seed"], dev)
    common_hash = V3.state_hash(V3.common_state(model))
    torch.manual_seed(cfg["seed"] + 1000 * rank)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng([cfg["seed"], rank])
    step0 = 0
    ck_last = os.path.join(a.out, "ckpt_last.pt")
    if os.path.exists(ck_last) and a.smoke == "none":
        ck = torch.load(ck_last, map_location=dev)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        rng = np.random.default_rng([cfg["seed"], rank, step0])          # the V3 rule: re-seeded per-rank streams at the resume step
        torch.manual_seed(cfg["seed"] + 1000 * rank + 7 * step0)
        say(f"resumed at step {step0} (model / EMA / optimizer restored; generators re-seeded with step {step0})")
    accum = int(cfg.get("accum", 1))
    eff_world = world if a.smoke == "none" else max(world, int(cfg["smoke"].get("emulate_world", 4)))
    pv, ps = cfg["batch_views"] // eff_world, max(1, cfg["states_per_batch"] // eff_world)
    mv, ms = pv // accum, max(1, ps // accum)
    run = {"arm": "v3_1", "config": a.config, "config_sha256": sha256_file(os.path.join(CFG_DIR, a.config)), "model_cfg": mcfg,
           "model_cfg_sha256": sha256_obj(mcfg), "train_cfg": cfg, "overrides": a.override, "world_size": world, "smoke": a.smoke, "git": git_info(),
           "common_init_sha256": common_hash, "params": V3.param_counts(model), "data_counts": n_all, "domains": ["outdoor"], "mp3d_training_samples": 0,
           "views": {k: v.n_views for k, v in data.items()}, "data_files": {k: v.files for k, v in data.items()},
           "data_manifest_sha256": sha256_obj({k: v.files for k, v in data.items()}), "host": os.uname()[1],
           "per_rank_batch": {"views": pv, "states": ps, "accum": accum, "micro_views": mv, "micro_states": ms},
           "gpu": torch.cuda.get_device_name(0), "slurm_job": os.environ.get("SLURM_JOB_ID"), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "torch": torch.__version__, "started": time.ctime(), "resumed_at": step0}
    if rank == 0:
        json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}_{int(time.time())}.json"), "w"), indent=1, default=str)
    say("params", run["params"], "common init", common_hash[:12])
    w_out = data["out_train"].state_weights(cfg["state_weight_t_le"], cfg["state_weight_factor"])
    steps = {"none": cfg["steps"], "overfit": cfg["smoke"]["overfit_steps"], "short": cfg["smoke"]["short_steps"],
             "profile": cfg["smoke"]["profile_steps"]}[a.smoke]
    say(f"per-rank batch: {pv} views from {ps} states as {accum} x ({mv} views / {ms} states) (effective world {eff_world})")
    dtr = data["out_train"]

    def draw():
        return [dtr.sample_batch(rng, mv, ms, cfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], w_out) for _ in range(accum)]
    fixed = [dtr.fixed_states("exp6-V3|smoke|out", mv)] * accum if a.smoke == "overfit" else None
    ema_steps = set(cfg.get("ema_save_steps", []))

    def save_ema(step):
        torch.save({"ema": ema.state_dict(), "step": step, "model_cfg": mcfg, "arm": "v3_1"}, os.path.join(a.out, f"ema_{step:06d}.pt"))

    def evaluate(step):
        from seen2scene.v3 import eval_dev as E
        ev = {"step": step}
        for nm, (dd, ss) in sets.items():
            ev[f"fm_{nm}"] = E.eval_fm(ema, dd, ss, nm, dev, cfg["eval_tau"])
        ev["samp_out_sc"] = E.eval_samples(ema, sets["out_sc"][0], sets["out_sc"][1], "out_sc", dev)
        ev["recon_out_sc"] = E.eval_recon(ema, sets["out_sc"][0], sets["out_sc"][1], dev)
        jl(ev_p, ev)
        say("eval", json.dumps(ev, default=float))
        save_ema(step)
        return ev

    if a.smoke == "none" and rank == 0 and step0 == 0:
        evaluate(0)
    if ddp and a.smoke == "none":
        dist.barrier()
    model.train()
    t_last, losses, first = time.time(), [], None
    prof = []
    for step in range(step0, steps):
        lr = cfg["lr"] if a.smoke == "overfit" else lr_at(step, cfg)
        for gp in opt.param_groups:
            gp["lr"] = lr
        beta = L.beta_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        micro = fixed if fixed is not None else draw()
        # labelled rays of the WHOLE global step, from the shard arrays, before any forward (exact global normalisation)
        nray_t = torch.tensor([float(sum(labelled_rays(dtr, b) for b in micro))], device=dev, dtype=torch.float64)
        if ddp:
            dist.all_reduce(nray_t)
        nray_g = float(nray_t.clamp(min=1.0))
        acc = {"esum": 0.0, "nray_fwd": 0.0, "rec": 0.0, "kl": 0.0}
        rec_diag = {}
        V = 0
        for b in micro:
            bt = D.build_cond(model, dtr, b, dev)
            V += bt["cond"].shape[0]
            eps = torch.randn(bt["cond"].shape[0], 2, M.GH, M.GW_PAD, device=dev)
            tau = torch.rand(bt["cond"].shape[0], device=dev)
            esum, nray = L.fm_terms(model, bt, eps, tau)
            rec, rec_diag = L.recon_terms(model, bt["vae"])
            kl, kl_ch = L.kl_terms(bt["vae"])
            loss = esum / (2.0 * nray_g) + (cfg["lambda_rec"] * rec + beta * kl) / (world * accum)
            if not torch.isfinite(loss):
                raise RuntimeError(f"STOP: non-finite loss at step {step} (fm {float(esum)}, rec {float(rec)}, kl {float(kl)})")
            loss.backward()
            acc["esum"] += float(esum); acc["nray_fwd"] += float(nray); acc["rec"] += float(rec) / accum; acc["kl"] += float(kl) / accum
        stats = torch.tensor([acc["esum"], acc["nray_fwd"], acc["rec"], acc["kl"]], device=dev, dtype=torch.float64)
        if ddp:
            params = [p_ for p_ in model.parameters()]
            for p_ in params:
                if p_.grad is None:
                    p_.grad = torch.zeros_like(p_)
            flat = torch._utils._flatten_dense_tensors([p_.grad for p_ in params])
            dist.all_reduce(flat)
            for g_, s_ in zip(params, torch._utils._unflatten_dense_tensors(flat, [p_.grad for p_ in params])):
                g_.grad.copy_(s_)
            dist.all_reduce(stats)
        esum_g, nray_fwd, rec_g, kl_g = float(stats[0]), float(stats[1]), float(stats[2]) / world, float(stats[3]) / world
        if abs(nray_fwd - nray_g) > 0.5 and nray_g > 1.0:
            raise RuntimeError(f"STOP: labelled-ray count mismatch at step {step}: data {nray_g} vs forward {nray_fwd}")
        if a.smoke == "overfit" and step == 10:
            gnorm = {k: float(sum((p_.grad.float() ** 2).sum() for p_ in getattr(model, k).parameters() if p_.grad is not None) ** 0.5)
                     for k in ("vae_enc", "vae_dec", "rays", "unet")}
            run["grad_check_step10"] = {"norms": gnorm, "pass": all(v > 0 and math.isfinite(v) for v in gnorm.values())}
            say("grad check", run["grad_check_step10"])
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm.detach(), 1.0 - cfg["ema"])
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        fm_g = esum_g / (2.0 * nray_g)
        tot_loss = fm_g + cfg["lambda_rec"] * rec_g + beta * kl_g
        losses.append(tot_loss)
        first = first if first is not None else tot_loss
        if a.smoke == "profile":
            torch.cuda.synchronize()
            prof.append(time.time() - t0)
        if (step + 1) % cfg["log_every"] == 0 or step == step0:
            dt = (time.time() - t_last) / (cfg["log_every"] if step > step0 else 1)
            t_last = time.time()
            rec_ = {"step": step + 1, "loss": float(np.mean(losses[-cfg["log_every"]:])), "fm": fm_g, "rec": rec_g, "kl": kl_g, "beta": beta,
                    "lr": lr, "grad_norm": float(gn), "views": V * world, "rays_outdoor": nray_g, "rec_diag": rec_diag,
                    "s_per_step": dt, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            if (step + 1) % (10 * cfg["log_every"]) == 0 or step == step0:
                with torch.no_grad():
                    st = bt["vae"][0]
                    rec_["posterior"] = VAE.posterior_stats(st["mu"], st["logvar"], st["valid8"])
                    rec_["kl_per_ch"] = [float(x) for x in kl_ch]
            if rank == 0:
                jl(log_p, rec_)
            if (step + 1) % (10 * cfg["log_every"]) == 0 or step == step0:
                say(json.dumps({k: v for k, v in rec_.items() if k not in ("kl_per_ch",)}, default=float))
        if a.smoke == "none" and rank == 0:
            if (step + 1) % cfg["eval_every"] == 0 or step + 1 == steps:
                evaluate(step + 1)
            elif (step + 1) in ema_steps:
                save_ema(step + 1)
            if (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps:
                save_ckpt(ck_last, model, ema, opt, step + 1, rng)
        if ddp and a.smoke == "none" and ((step + 1) % cfg["eval_every"] == 0 or (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps):
            dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return
    run["finished"] = time.ctime()
    run["final_loss_mean50"] = float(np.mean(losses[-50:])) if losses else None
    if a.smoke == "overfit":
        run["overfit"] = {"first_loss": first, "last20_loss": float(np.mean(losses[-20:])), "drop": 1.0 - float(np.mean(losses[-20:])) / first}
        say("overfit", run["overfit"])
    if a.smoke == "profile":
        p = np.asarray(prof[10:])
        run["profile"] = {"s_per_step_mean": float(p.mean()), "s_per_step_p90": float(np.quantile(p, 0.9)), "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                          "views_per_step_global": cfg["batch_views"], "est_hours_per_10k_steps": float(p.mean()) * 10000 / 3600}
        say("profile", run["profile"])
    if a.smoke == "short":
        log = [json.loads(x) for x in open(log_p)] if os.path.exists(log_p) else []
        sps = [r["s_per_step"] for r in log if r["step"] > 100]
        run["short"] = {"first_loss": first, "first100": float(np.mean(losses[:100])), "last100": float(np.mean(losses[-100:])),
                        "s_per_step_mean": float(np.mean(sps)) if sps else None, "s_per_step_p90": float(np.quantile(sps, 0.9)) if sps else None,
                        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "est_hours_40k": float(np.mean(sps)) * 40000 / 3600 if sps else None}
        run["short"]["pass"] = run["short"]["last100"] < run["short"]["first100"]
        say("short", run["short"])
    if a.smoke != "none":
        torch.save({"ema": ema.state_dict(), "step": steps, "model_cfg": mcfg, "arm": "v3_1", "smoke": a.smoke}, os.path.join(a.out, "ema_smoke.pt"))
    json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}.json"), "w"), indent=1, default=str)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
