"""EXP6 V4 -- scratch Outdoor-only training of V4 = V2 3D U-Net + original camera-ray sampling + Ray-Aligned Cross-
Attention (seen2scene.futureview.model.RayXAttn, config flag ray_xattn) + the original 2D rectified flow.

= task1/scripts/t12_train.py (the V2 Original recipe: model class, loss v2_data.fm_sum, sampler, DDP gradient sum,
AdamW + warmup / cosine, EMA, eval_fm / eval_samples imported UNCHANGED) with
  config   exp6_v4/configs/outdoor_train.json (+ model_v4.json, task1 data_v1.json)
  data     Macarons++ Train-6 minus S-c (task1 frozen_values.json split) read from the E2 D100 view shards
           (work_dirs/.../e2/out_views_d100, D100 P1 coarse grids; t22_data.CompactViewData = v2_data.ViewData with
           compressed grids); S-c (held-out trajectories of the same 6 scenes) only for the development metrics;
           ZERO MP3D samples; no Dev-3 / Test-3 scene is read
  ckpt     EMA at cfg ema_save_steps (ema_<step>.pt), resumable ckpt_last.pt every ckpt_every steps (model, EMA,
           optimizer, step, RNG); checkpoint policy fixed EMA @ steps (no selection on any evaluation scene)
  manifest run_<mode>.json: git SHA / dirty, config sha256, data file sha256, parameter counts, host / GPUs / job
  smoke    --smoke overfit (1 GPU, 200 steps on one fixed batch: loss drop, sign agreement, gradient check incl.
           ray_xattn at step 10) | --smoke short (4 GPUs, 1,000 steps of the real recipe: throughput / VRAM / loss)
torchrun --standalone --nproc_per_node 4 experiments/exp6_v4/scripts/train_v4.py --out DIR [--smoke overfit|short]
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v4_paths as P  # noqa: E402

sys.path.insert(0, os.path.join(P.EXP5, "task2", "scripts"))
sys.path.insert(0, os.path.join(P.EXP5, "task1", "scripts"))
import v2_frozen as VF  # noqa: E402
import v2_data as VD  # noqa: E402
import t12_train as TR  # noqa: E402  (eval_fm, eval_samples, lr_at, save_ckpt, jl)
import t22_data as TD  # noqa: E402  (CompactViewData)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from seen2scene.futureview import model as M  # noqa: E402


def git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=P.REPO).decode().strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=P.REPO)
        return {"sha": sha, "dirty": bool(diff), "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None}
    except Exception as e:                       # noqa: BLE001
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="outdoor_train.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", choices=["none", "overfit", "short"], default="none")
    ap.add_argument("--override", nargs="*", default=[], help="test-only overrides key=value (steps, ckpt_every, eval_every)")
    a = ap.parse_args()
    import torch.distributed as dist
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=90))
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    VF.verify_frozen()
    cfg = P.load_cfg(a.config)
    for kv in a.override:
        k, v = kv.split("=")
        cfg[k] = type(cfg[k])(v)
    mcfg = P.load_cfg(cfg["model"])
    assert mcfg.get("ray_xattn", {}).get("enabled"), "V4 config must enable ray_xattn"
    dcfg = VF.load_cfg(cfg["data"])
    root = os.path.join(P.REPO, cfg["data_root"])
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log_p, ev_p = os.path.join(a.out, "log.jsonl"), os.path.join(a.out, "eval.jsonl")
    say = (lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)) if rank == 0 else (lambda *x: None)  # noqa: E731
    train_items, held_items = VD.load_split(cfg["train_set"])
    built = lambda x: os.path.exists(os.path.join(root, x[0], f"{x[1]}.npz"))  # noqa: E731
    missing = [x for x in train_items + held_items if not built(x)]
    if missing:
        raise VF.InvalidRun(f"STOP: {len(missing)} shards missing under {root}: {missing[:5]}")
    if a.smoke != "none":
        train_items, held_items = train_items[:8], held_items[:3]
    say(f"loading {len(train_items)} train / {len(held_items)} S-c trajectories from {root}")
    data = TD.CompactViewData(train_items, root=root, log=say)
    held = TD.CompactViewData(held_items, root=root, log=say)
    say(f"train views {data.n_views}, states {len(data.states)}; S-c views {held.n_views}")
    for s_ in set(data.scene_of) | set(held.scene_of):
        assert s_ in dcfg["scenes_train6"], s_
    sets = {"train_sub": (data, data.fixed_states(f"exp5-V2|eval|train_sub|{cfg['train_set']}", cfg["eval_sets"]["train_sub"])),
            "sc_sub": (held, held.fixed_states("exp5-V2|eval|sc_sub", cfg["eval_sets"]["sc_sub"], per_scene=True))}
    torch.manual_seed(cfg["seed"])
    model = M.V2Model(mcfg).to(dev)
    if ddp:
        model.rays_ckpt = False
        for p_ in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(p_.data, 0)
        torch.manual_seed(cfg["seed"] + 1000 * rank)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    groups = ("encoder", "rays", "ray_xattn", "unet")
    n_par = {k: sum(p_.numel() for p_ in getattr(model, k).parameters()) for k in groups}
    n_par["total"] = sum(n_par.values())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng([cfg["seed"], rank])
    step0 = 0
    ck_last = os.path.join(a.out, "ckpt_last.pt")
    if os.path.exists(ck_last) and a.smoke == "none":
        ck = torch.load(ck_last, map_location=dev)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        if ddp:                                   # per-rank streams re-seeded deterministically at the resume step (V2 rule)
            rng = np.random.default_rng([cfg["seed"], rank, step0])
            torch.manual_seed(cfg["seed"] + 1000 * rank + 7 * step0)
        else:
            rng.bit_generator.state = ck["np_rng"]
            torch.cuda.set_rng_state(ck["torch_cuda_rng"].cpu()); torch.set_rng_state(ck["torch_rng"].cpu())
        say(f"resumed at step {step0}")
    run = {"config": a.config, "config_sha256": P.sha256_file(os.path.join(P.CFG, a.config)), "train_cfg": cfg, "overrides": a.override,
           "model_config_sha256": P.sha256_file(os.path.join(P.CFG, cfg["model"])), "model_cfg": mcfg,
           "data_config_sha256": VF.sha256_file(os.path.join(VF.T1, "configs", cfg["data"])), "data_root": root, "params": n_par,
           "train_trajs": len(train_items), "train_views": data.n_views, "train_states": len(data.states),
           "held_trajs": len(held_items), "held_views": held.n_views, "domains": ["outdoor"], "mp3d_training_samples": 0,
           "data_files_sha256": {**data.files, **held.files}, "smoke": a.smoke, "host": os.uname()[1], "git": git_info(),
           "gpu": torch.cuda.get_device_name(0), "started": time.ctime(), "resumed_at": step0, "world_size": world,
           "per_rank_batch": {"views": cfg["batch_views"] // world, "states": cfg["states_per_batch"] // world},
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__}
    run["data_manifest_sha256"] = hashlib.sha256(json.dumps(run["data_files_sha256"], sort_keys=True).encode()).hexdigest()
    if rank == 0:
        json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}_{int(time.time())}.json"), "w"), indent=1, default=str)
    say("params", n_par)
    weights = data.state_weights(dcfg["state_weight_t_le"], dcfg["state_weight_factor"])
    steps = {"none": cfg["steps"], "overfit": cfg["smoke"]["overfit_steps"], "short": cfg["smoke"]["short_steps"]}[a.smoke]
    fixed = data.fixed_states("exp5-V2|smoke|overfit", cfg["batch_views"]) if a.smoke == "overfit" else None
    ema_steps = set(cfg.get("ema_save_steps", []))

    def save_ema(step):
        torch.save({"ema": ema.state_dict(), "step": step, "model_cfg": mcfg, "arm": "v4"}, os.path.join(a.out, f"ema_{step:06d}.pt"))

    if a.smoke == "none" and rank == 0 and step0 == 0 and 0 in ema_steps:
        save_ema(0)
    model.train()
    t_last, losses, first_loss = time.time(), [], None
    for step in range(step0, steps):
        lr = cfg["lr"] if a.smoke == "overfit" else TR.lr_at(step, cfg)
        for gp in opt.param_groups:
            gp["lr"] = lr
        batch = fixed if fixed is not None else data.sample_batch(rng, cfg["batch_views"] // world, cfg["states_per_batch"] // world,
                                                                  dcfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], weights)
        bt = VD.build_cond(model, data, batch, dev)
        V = bt["cond"].shape[0]
        eps = torch.randn(V, 2, M.GH, M.GW_PAD, device=dev)
        tau = torch.rand(V, device=dev)
        esum, nray = VD.fm_sum(model, bt, eps, tau)
        if ddp:
            tot = torch.stack([esum.detach(), nray]).double()
            dist.all_reduce(tot)
            loss_g, nray = float(tot[0] / (2 * tot[1].clamp(min=1))), tot[1]
        else:
            loss_g = None
        loss = esum / (2.0 * nray.clamp(min=1.0).to(esum.dtype))
        if not torch.isfinite(loss):
            raise VF.InvalidRun(f"STOP: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if ddp:
            params = list(model.parameters())
            for p_ in params:
                if p_.grad is None:
                    p_.grad = torch.zeros_like(p_)
            flat = torch._utils._flatten_dense_tensors([p_.grad for p_ in params])
            dist.all_reduce(flat)
            for g_, s_ in zip(params, torch._utils._unflatten_dense_tensors(flat, [p_.grad for p_ in params])):
                g_.grad.copy_(s_)
            loss = torch.tensor(loss_g)
        if a.smoke == "overfit" and step == 10:
            gnorm = {k: float(sum((p_.grad.float() ** 2).sum() for p_ in getattr(model, k).parameters() if p_.grad is not None) ** 0.5) for k in groups}
            gx = {k: float(p_.grad.float().norm()) for k, p_ in model.ray_xattn.named_parameters()}
            run["grad_check_step10"] = {"norms": gnorm, "ray_xattn_tensors": gx,
                                       "pass": all(v > 0 and math.isfinite(v) for v in gnorm.values()) and all(v > 0 and math.isfinite(v) for v in gx.values())}
            say("grad check", run["grad_check_step10"])
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm.detach(), 1.0 - cfg["ema"])
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        losses.append(float(loss))
        if first_loss is None:
            first_loss = float(loss)
        if (step + 1) % 50 == 0 or step == step0:
            dt = (time.time() - t_last) / (50 if step > step0 else 1)
            t_last = time.time()
            rec = {"step": step + 1, "loss": float(np.mean(losses[-50:])), "lr": lr, "grad_norm": float(gn), "views": V * world,
                   "rays": float(nray), "s_per_step": dt, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            if (step + 1) % 500 == 0 or step == step0:
                with torch.no_grad():
                    rec["xattn_wo_rms"] = float(model.ray_xattn.wo.weight.float().pow(2).mean().sqrt())
            if rank == 0:
                TR.jl(log_p, rec)
            if (step + 1) % 500 == 0 or step == step0:
                say(json.dumps(rec))
        if a.smoke == "none" and rank == 0:
            if (step + 1) % cfg["eval_every"] == 0 or step + 1 == steps:
                ev = {"step": step + 1}
                for nm, (dd, ss) in sets.items():
                    ev[f"fm_{nm}"] = TR.eval_fm(ema, dd, ss, nm, dev, cfg["eval_tau"])
                if (step + 1) % cfg["sample_eval_every"] == 0 or step + 1 == steps:
                    for nm, (dd, ss) in sets.items():
                        ev[f"samp_{nm}"] = TR.eval_samples(ema, dd, ss, nm, dev)
                TR.jl(ev_p, ev)
                say("eval", json.dumps(ev))
            if (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps:
                TR.save_ckpt(ck_last, model, ema, opt, step + 1, rng)
            if (step + 1) in ema_steps or step + 1 == steps:
                save_ema(step + 1)
        if ddp and a.smoke == "none" and ((step + 1) % cfg["eval_every"] == 0 or (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps):
            dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return
    run["finished"] = time.ctime()
    run["final_loss_mean50"] = float(np.mean(losses[-50:])) if losses else None
    if a.smoke == "overfit":
        bt = VD.build_cond(model, data, fixed, dev)
        x = M.sample(model, bt["cond"], bt["m_unk"], torch.randn(bt["cond"].shape[0], 2, M.GH, M.GW_PAD, device=dev))
        lm = bt["lmask"]
        agree = float(((x[:, 0] > 0) == (bt["x1"][:, 0] > 0))[lm].float().mean())
        drop = 1.0 - float(np.mean(losses[-20:])) / first_loss
        run["overfit"] = {"first_loss": first_loss, "last20_loss": float(np.mean(losses[-20:])), "drop": drop, "sign_agree": agree,
                          "pass": drop > cfg["smoke"]["overfit_loss_drop"] and agree > cfg["smoke"]["overfit_sign_agree"]}
        say("overfit", run["overfit"])
    if a.smoke == "short":
        log = [json.loads(x) for x in open(log_p)] if os.path.exists(log_p) else []
        sps = [r["s_per_step"] for r in log if r["step"] > 100]
        run["short"] = {"first100": float(np.mean(losses[:100])), "last100": float(np.mean(losses[-100:])), "s_per_step_mean": float(np.mean(sps)) if sps else None,
                        "s_per_step_p90": float(np.quantile(sps, 0.9)) if sps else None, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                        "views_per_step_global": cfg["batch_views"], "est_hours_40k": float(np.mean(sps)) * 40000 / 3600 if sps else None}
        run["short"]["pass"] = run["short"]["last100"] < run["short"]["first100"]
        say("short", run["short"])
    if a.smoke == "none":
        final = os.path.join(a.out, f"ema_{steps:06d}.pt")
        run["final_checkpoint"] = os.path.relpath(final, P.REPO)
        run["final_checkpoint_sha256"] = P.sha256_file(final)
    run["code_sha256"] = VF.code_fingerprint()
    json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}.json"), "w"), indent=1, default=str)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
