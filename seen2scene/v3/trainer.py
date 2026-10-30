"""Scratch training of V3 (T0 / T1), one shared loop (EXP6 V3; protocol.md).

torchrun --nproc_per_node N -m seen2scene.v3.trainer --arm t0|t1 --out DIR [--config train_v3.json] [--smoke overfit|short|profile]

  init      model constructed from seed (identical common modules for both arms; the arm only switches xattn.enabled);
            EMA = copy; fresh AdamW; nothing loaded from any earlier checkpoint
  batch     per rank: batch_views / 2 / world views per domain from states_per_batch / 2 / world states each (V2 rules)
  loss      L_FM (global labelled-ray mean) + lambda_rec * L_recon + beta(step) * KL (state means, / world under DDP)
  ddp       gradients summed over ranks by hand (as the V2 trainer; no DDP wrapper, activation checkpointing safe)
  eval      every eval_every steps (and at step 0) on fixed subsets of VALSEL / S-c (+ train subsets): FM loss,
            sample metrics (fair ES, Brier, false wall, hit ratio, spread) with the EMA; the EMA is saved as ema_<step>.pt
  ckpt      ckpt_last.pt every ckpt_every steps (model, EMA, optimizer, step, numpy / torch / cuda RNG), atomic; resume
  manifest  run_<mode>.json: git SHA / dirty, config hashes, data files + sizes, common-module init hash, parameter
            counts, host / GPUs / SLURM job, world size, timings
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
from seen2scene.futureview import model as M
from seen2scene.v3 import data as D
from seen2scene.v3 import losses as L
from seen2scene.v3 import model as V3
from seen2scene.v3 import vae as VAE

REPO = str(PROJECT_ROOT)
CFG_DIR = os.path.join(REPO, "experiments", "exp6_v3", "configs")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO)
        return {"sha": sha, "dirty": bool(diff), "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None}
    except Exception as e:                       # noqa: BLE001
        return {"error": str(e)}


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def jl(p, obj):
    with open(p, "a") as f:
        f.write(json.dumps(obj, default=float) + "\n")


def save_ckpt(p, model, ema, opt, step, rng, extra=None):
    tmp = p + ".tmp"
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                "np_rng": rng.bit_generator.state, "torch_cuda_rng": torch.cuda.get_rng_state(), "torch_rng": torch.get_rng_state(),
                **(extra or {})}, tmp)
    os.replace(tmp, p)


def selection_score(evals, base_step=5000):
    """Frozen checkpoint-selection score of every evaluated step >= base_step (None where the base is missing)."""
    base = next((e for e in evals if e["step"] == base_step), None)
    if base is None:
        return {}
    es = lambda e, k: e[f"samp_{k}"]["fair_es_arclength_vox"]  # noqa: E731
    return {e["step"]: es(e, "in_val") / es(base, "in_val") + es(e, "out_sc") / es(base, "out_sc") for e in evals if e["step"] >= base_step}


def early_stop(ev_p, cfg, step):
    """Protocol v1.0: after after_fraction of the budget, stop when the last patience_evals evaluations did not improve
    the best selection score."""
    es = cfg.get("early_stop")
    if not es or step < es["after_fraction"] * cfg["steps"]:
        return False
    evals = [e for e in map(json.loads, open(ev_p)) if e["step"] <= step]
    sc = selection_score(evals)
    if len(sc) < es["patience_evals"] + 1:
        return False
    steps_sorted = sorted(sc)
    best_before = min(sc[k] for k in steps_sorted[:-es["patience_evals"]])
    return all(sc[k] >= best_before for k in steps_sorted[-es["patience_evals"]:])


def model_cfg(arm, cfg):
    mcfg = json.load(open(os.path.join(CFG_DIR, cfg["model"])))
    mcfg["xattn"]["enabled"] = arm == "t1"
    return mcfg


def make_model(mcfg, seed, dev):
    torch.manual_seed(seed)
    return V3.V3Model(mcfg).to(dev)


def load_data(rank, world, smoke, say):
    in_tr, in_val = D.items("indoor", "train"), D.items("indoor", "valsel")
    out_tr, out_sc = D.items("outdoor", "train"), D.items("outdoor", "sc")
    have = lambda dom, it: [x for x in it if os.path.exists(D.cache_path(dom, *x))]  # noqa: E731
    in_tr, in_val, out_tr, out_sc = have("indoor", in_tr), have("indoor", in_val), have("outdoor", out_tr), have("outdoor", out_sc)
    if smoke:
        in_tr, in_val, out_tr, out_sc = in_tr[:6], (in_val[:2] or in_tr[-1:]), out_tr[:6], (out_sc[:3] or out_tr[-1:])
    n_all = {"in_train": len(in_tr), "in_val": len(in_val), "out_train": len(out_tr), "out_sc": len(out_sc)}
    in_tr, out_tr = sorted(in_tr)[rank::world], sorted(out_tr)[rank::world]
    if rank != 0:
        in_val, out_sc = in_val[:1], out_sc[:1]
    say(f"rank 0 loading indoor {len(in_tr)} / valsel {len(in_val)}; outdoor {len(out_tr)} / S-c {len(out_sc)} trajectories (world {world})")
    return {"in_train": D.V3Views("indoor", in_tr, say), "in_val": D.V3Views("indoor", in_val, say),
            "out_train": D.V3Views("outdoor", out_tr, say), "out_sc": D.V3Views("outdoor", out_sc, say)}, n_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["t0", "t1"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="train_v3.json")
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
    for kv in a.override:                                  # test-only (recorded in the run manifest)
        k, v = kv.split("=")
        if k == "eval_views":
            cfg["eval_sets"] = {kk: int(v) for kk in cfg["eval_sets"]}
        else:
            cfg[k] = type(cfg[k])(v)
    mcfg = model_cfg(a.arm, cfg)
    os.makedirs(a.out, exist_ok=True)
    log_p, ev_p = os.path.join(a.out, "log.jsonl"), os.path.join(a.out, "eval.jsonl")
    say = (lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)) if rank == 0 else (lambda *x: None)  # noqa: E731
    data, n_all = load_data(rank, world, a.smoke != "none", say)
    say("views:", {k: v.n_views for k, v in data.items()}, "cache MB:", {k: round(v.cache_bytes / 1e6) for k, v in data.items()})
    es = cfg["eval_sets"]
    sets = {"in_val": (data["in_val"], data["in_val"].fixed_states("exp6-V3|eval|in_val", es["in_val"], per_scene=True)),
            "out_sc": (data["out_sc"], data["out_sc"].fixed_states("exp6-V3|eval|out_sc", es["out_sc"], per_scene=True)),
            "in_train": (data["in_train"], data["in_train"].fixed_states("exp6-V3|eval|in_train", es["in_train"])),
            "out_train": (data["out_train"], data["out_train"].fixed_states("exp6-V3|eval|out_train", es["out_train"]))}
    model = make_model(mcfg, cfg["seed"], dev)
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
        # per-rank generators are re-seeded deterministically from (seed, rank, step0) -- rank 0's saved states are kept
        # in the checkpoint for the record only (the V2 trainer's rule; exact per-rank restoration would need every
        # rank's state). The sample stream after a resume is reproducible but differs from an uninterrupted run.
        rng = np.random.default_rng([cfg["seed"], rank, step0])
        torch.manual_seed(cfg["seed"] + 1000 * rank + 7 * step0)
        say(f"resumed at step {step0} (model / EMA / optimizer restored; generators re-seeded with step {step0})")
    run = {"arm": a.arm, "config": a.config, "config_sha256": sha256_file(os.path.join(CFG_DIR, a.config)), "model_cfg": mcfg,
           "model_cfg_sha256": sha256_obj(mcfg), "train_cfg": cfg, "overrides": a.override, "world_size": world, "smoke": a.smoke, "git": git_info(),
           "common_init_sha256": common_hash, "params": V3.param_counts(model), "data_counts": n_all,
           "views": {k: v.n_views for k, v in data.items()}, "data_files": {k: v.files for k, v in data.items()},
           "data_manifest_sha256": sha256_obj({k: v.files for k, v in data.items()}), "host": os.uname()[1],
           "gpu": torch.cuda.get_device_name(0), "slurm_job": os.environ.get("SLURM_JOB_ID"), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "torch": torch.__version__, "started": time.ctime(), "resumed_at": step0}
    if rank == 0:
        json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}_{int(time.time())}.json"), "w"), indent=1, default=str)
    w_in = data["in_train"].state_weights(cfg["state_weight_t_le"], cfg["state_weight_factor"])
    w_out = data["out_train"].state_weights(cfg["state_weight_t_le"], cfg["state_weight_factor"])
    steps = {"none": cfg["steps"], "overfit": cfg["smoke"]["overfit_steps"], "short": cfg["smoke"]["short_steps"],
             "profile": cfg["smoke"]["profile_steps"]}[a.smoke]
    accum = int(cfg.get("accum", 1))
    eff_world = world if a.smoke == "none" else max(world, int(cfg["smoke"].get("emulate_world", 4)))   # smokes: per-rank sizes of the 4-GPU run
    half = cfg["batch_views"] // 2 // eff_world // accum
    hs = max(1, cfg["states_per_batch"] // 2 // eff_world // accum)
    say(f"per-rank batch: {half} + {half} views from {hs} + {hs} states, accum {accum} (effective world {eff_world})")

    def draw():
        return [(data["in_train"], data["in_train"].sample_batch(rng, half, hs, cfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], w_in)),
                (data["out_train"], data["out_train"].sample_batch(rng, half, hs, cfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], w_out))]
    fixed = [(data["in_train"], data["in_train"].fixed_states("exp6-V3|smoke|in", half)),
             (data["out_train"], data["out_train"].fixed_states("exp6-V3|smoke|out", half))] if a.smoke == "overfit" else None

    def evaluate(step):
        from seen2scene.v3 import eval_dev as E
        ev = {"step": step}
        for nm, (dd, ss) in sets.items():
            ev[f"fm_{nm}"] = E.eval_fm(ema, dd, ss, nm, dev, cfg["eval_tau"])
            if nm in ("in_val", "out_sc"):
                ev[f"samp_{nm}"] = E.eval_samples(ema, dd, ss, nm, dev)
        ev["recon_in_val"] = E.eval_recon(ema, sets["in_val"][0], sets["in_val"][1], dev)
        ev["recon_out_sc"] = E.eval_recon(ema, sets["out_sc"][0], sets["out_sc"][1], dev)
        jl(ev_p, ev)
        say("eval", json.dumps(ev, default=float))
        torch.save({"ema": ema.state_dict(), "step": step, "model_cfg": mcfg, "arm": a.arm}, os.path.join(a.out, f"ema_{step:06d}.pt"))
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
        acc = {"esum": 0.0, "nray": 0.0, "rec": 0.0, "kl": 0.0, "rays_in": 0.0, "rays_out": 0.0}
        rec_diag = {}
        for _ in range(accum):
            parts = fixed if fixed is not None else draw()
            bt = D.build_mixed(model, parts, dev)
            V = bt["cond"].shape[0]
            eps = torch.randn(V, 2, M.GH, M.GW_PAD, device=dev)
            tau = torch.rand(V, device=dev)
            esum, nray = L.fm_terms(model, bt, eps, tau)
            rec, rec_diag = L.recon_terms(model, bt["vae"])
            kl, kl_ch = L.kl_terms(bt["vae"])
            tot = torch.stack([esum.detach(), nray]).double()
            if ddp:
                dist.all_reduce(tot)
            nray_g = tot[1].clamp(min=1.0)
            loss = esum / (2.0 * nray_g.to(esum.dtype)) + (cfg["lambda_rec"] * rec + beta * kl) / (world * accum)
            if not torch.isfinite(loss):
                raise RuntimeError(f"STOP: non-finite loss at step {step} (fm {float(esum)}, rec {float(rec)}, kl {float(kl)})")
            loss.backward()
            n_in = bt["n_views"][0]
            with torch.no_grad():
                lm = bt["lmask"].float()
                acc["rays_in"] += float(lm[:n_in].sum()); acc["rays_out"] += float(lm[n_in:].sum())
                acc["esum"] += float(tot[0]); acc["nray"] += float(tot[1]); acc["rec"] += float(rec) / accum; acc["kl"] += float(kl) / accum
        if ddp:
            params = [p_ for p_ in model.parameters()]
            for p_ in params:
                if p_.grad is None:
                    p_.grad = torch.zeros_like(p_)
            flat = torch._utils._flatten_dense_tensors([p_.grad for p_ in params])
            dist.all_reduce(flat)
            for g_, s_ in zip(params, torch._utils._unflatten_dense_tensors(flat, [p_.grad for p_ in params])):
                g_.grad.copy_(s_)
            stats = torch.tensor([acc["rec"], acc["kl"]], device=dev, dtype=torch.float64)
            dist.all_reduce(stats)
            acc["rec"], acc["kl"] = float(stats[0] / world), float(stats[1] / world)
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
        fm_g = acc["esum"] / (2.0 * max(acc["nray"], 1.0))
        tot_loss = fm_g + cfg["lambda_rec"] * acc["rec"] + beta * acc["kl"]
        losses.append(tot_loss)
        first = first if first is not None else tot_loss
        if a.smoke == "profile":
            torch.cuda.synchronize()
            prof.append(time.time() - t0)
        if (step + 1) % cfg["log_every"] == 0 or step == step0:
            dt = (time.time() - t_last) / (cfg["log_every"] if step > step0 else 1)
            t_last = time.time()
            rec_ = {"step": step + 1, "loss": float(np.mean(losses[-cfg["log_every"]:])), "fm": fm_g, "rec": acc["rec"], "kl": acc["kl"], "beta": beta,
                    "lr": lr, "grad_norm": float(gn), "views": V * accum, "rays_indoor": acc["rays_in"], "rays_outdoor": acc["rays_out"],
                    "rec_diag": rec_diag, "s_per_step": dt, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            if (step + 1) % (10 * cfg["log_every"]) == 0 or step == step0:
                with torch.no_grad():
                    st = bt["vae"][0]
                    rec_["posterior"] = VAE.posterior_stats(st["mu"], st["logvar"], st["valid8"])
                    rec_["kl_per_ch"] = [float(x) for x in kl_ch]
            if rank == 0:
                jl(log_p, rec_)
            if (step + 1) % (10 * cfg["log_every"]) == 0 or step == step0:
                say(json.dumps({k: v for k, v in rec_.items() if k not in ("kl_per_ch",)}, default=float))
        stop = torch.zeros(1, device=dev)
        if a.smoke == "none" and rank == 0:
            if (step + 1) % cfg["eval_every"] == 0 or step + 1 == steps:
                evaluate(step + 1)
                if early_stop(ev_p, cfg, step + 1):
                    say(f"early stop at step {step + 1} (frozen rule: no selection-score improvement for "
                        f"{cfg['early_stop']['patience_evals']} evaluations after {cfg['early_stop']['after_fraction']:.2f} of the budget)")
                    stop += 1
            if (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps:
                save_ckpt(ck_last, model, ema, opt, step + 1, rng)
        if ddp and a.smoke == "none" and ((step + 1) % cfg["eval_every"] == 0 or (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps):
            dist.all_reduce(stop)
            dist.barrier()
        if float(stop) > 0:
            run["early_stopped_at"] = step + 1
            break
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
        run["short"] = {"first_loss": first, "last50_loss": float(np.mean(losses[-50:]))}
    if a.smoke != "none":                                   # smoke EMA for pipeline tests (never a training result)
        torch.save({"ema": ema.state_dict(), "step": steps, "model_cfg": mcfg, "arm": a.arm, "smoke": a.smoke}, os.path.join(a.out, "ema_smoke.pt"))
    json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}.json"), "w"), indent=1, default=str)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
