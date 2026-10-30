"""EXP5 Task 2 / T2.1b -- V2-Mix: indoor + outdoor mixed fine-tuning of V2 Original (seen2scene env; torchrun on up to 4
GPUs; task2/configs/train_mix_v1.json). Architecture, loss, sampler rules and evaluation functions are those of Task 1
(task1/scripts/t12_train.py, imported unchanged); only the data (two domains) and the initialisation differ.

  init      model and EMA = V2 Original EMA@60k (W20-1); fresh AdamW
  batch     batch_views / 2 MP3D views (t21 shards, n_win = 5 frames) + batch_views / 2 Macarons++ views (T1.1 shards,
            Train-6 minus S-c); loss = rectified-flow velocity MSE over all labelled M_unknown rays of the batch
            (exactly the Task 1 loss)
  eval      every eval_every steps (and at step 0 = V2 Original) on fixed subsets: indoor selection (MP3D VALSEL houses,
            never the evaluation houses) and outdoor S-c; FM loss + sample metrics (fair ES in unknown-arclength voxels,
            Brier, false-wall rate, hit ratio)
  selection score = ES_in / ES_in(step 0) + ES_out / ES_out(step 0); minimum; ties -> later step (frozen before
            training); every evaluated EMA is saved in W21
  --smoke overfit / short : one fixed batch (loss drop, sign agreement, gradient check) / short run (loss decreases)
Logs: <out>/log.jsonl, eval.jsonl, run_<mode>.json; checkpoints <out>/ema_<step>.pt, ckpt_last.pt.
"""
import argparse
import copy
import datetime
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402
import t22_data as TD  # noqa: E402

VF, VD = TD.VF, TD.VD
import t12_train as TR  # noqa: E402  (eval_fm, eval_samples, lr_at, save_ckpt, jl)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from seen2scene.futureview import model as M  # noqa: E402


def load_cfg(name):
    return json.load(open(os.path.join(C.T2, "configs", name)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train_mix_v1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", choices=["none", "overfit", "short"], default="none")
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
    cfg = load_cfg(a.config)
    mcfg = VF.load_cfg(cfg["model"])
    dcfg = VF.load_cfg(cfg["data_outdoor"])
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log_p, ev_p = os.path.join(a.out, "log.jsonl"), os.path.join(a.out, "eval.jsonl")
    say = (lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)) if rank == 0 else (lambda *x: None)  # noqa: E731
    out_tr, out_held = VD.load_split("train6_minus_Sc")
    in_tr, in_val = TD.mp3d_items("train"), TD.mp3d_items("valsel")
    if a.smoke != "none":
        out_tr = [x for x in out_tr][:6]
        out_held = out_held[:3]
        in_tr, in_val = in_tr[:6], (in_val[:2] or in_tr[-1:])     # smoke only: before the selection shards exist
    # per-rank shard of the TRAINING trajectories (round-robin over the sorted lists); every rank draws its half-batches
    # from its own shard, gradients are summed over ranks; the selection / evaluation sets live on rank 0 only
    out_tr, in_tr = sorted(out_tr)[rank::world], sorted(in_tr)[rank::world]
    if rank != 0:
        out_held, in_val = out_held[:1], in_val[:1]
    say(f"rank 0 loading outdoor {len(out_tr)} / S-c {len(out_held)}; indoor {len(in_tr)} / valsel {len(in_val)} trajectories (world {world})")
    od_tr, od_held = TD.CompactViewData(out_tr, log=say), TD.CompactViewData(out_held, log=say)
    id_tr, id_val = TD.MP3DViewData(in_tr, log=say), TD.MP3DViewData(in_val, log=say)
    say(f"views: outdoor {od_tr.n_views} / S-c {od_held.n_views}; indoor {id_tr.n_views} / valsel {id_val.n_views}")
    es = cfg["eval_sets"]
    sets = {"in_val": (id_val, id_val.fixed_states("exp5-T2|eval|in_val", es["in_val"], per_scene=True)),
            "out_sc": (od_held, od_held.fixed_states("exp5-V2|eval|sc_sub", es["out_sc"], per_scene=True)),
            "in_train": (id_tr, id_tr.fixed_states("exp5-T2|eval|in_train", es["in_train"])),
            "out_train": (od_tr, od_tr.fixed_states("exp5-T2|eval|out_train", es["out_train"]))}
    ck0 = torch.load(os.path.join(VF.REPO, cfg["init_checkpoint"]), map_location=dev)
    assert VF.sha256_file(os.path.join(VF.REPO, cfg["init_checkpoint"])) == cfg["init_checkpoint_sha256"]
    model = M.V2Model(mcfg).to(dev)
    model.load_state_dict(ck0["ema"])
    if ddp:
        model.rays_ckpt = False
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
        rng = np.random.default_rng([cfg["seed"], rank, step0])
        torch.manual_seed(cfg["seed"] + 1000 * rank + 7 * step0)
        say(f"resumed at step {step0}")
    run = {"config": a.config, "config_sha256": VF.sha256_file(os.path.join(C.T2, "configs", a.config)), "world_size": world,
           "views": {"out_train": od_tr.n_views, "out_sc": od_held.n_views, "in_train": id_tr.n_views, "in_val": id_val.n_views},
           "data_files_sha256": {**od_tr.files, **od_held.files, **id_tr.files, **id_val.files}, "smoke": a.smoke,
           "host": os.uname()[1], "gpu": torch.cuda.get_device_name(0), "started": time.ctime(), "resumed_at": step0}
    w_out = od_tr.state_weights(dcfg["state_weight_t_le"], dcfg["state_weight_factor"])
    w_in = id_tr.state_weights(dcfg["state_weight_t_le"], dcfg["state_weight_factor"])
    steps = {"none": cfg["steps"], "overfit": cfg["smoke"]["overfit_steps"], "short": cfg["smoke"]["short_steps"]}[a.smoke]
    half = cfg["batch_views"] // 2 // world
    hs = cfg["states_per_batch"] // 2 // world

    def draw():
        return [(id_tr, id_tr.sample_batch(rng, half, hs, dcfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], w_in)),
                (od_tr, od_tr.sample_batch(rng, half, hs, dcfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], w_out))]
    fixed = [(id_tr, id_tr.fixed_states("exp5-T2|smoke|in", half)), (od_tr, od_tr.fixed_states("exp5-T2|smoke|out", half))] \
        if a.smoke == "overfit" else None

    def evaluate(step):
        ev = {"step": step}
        for nm, (dd, ss) in sets.items():
            ev[f"fm_{nm}"] = TR.eval_fm(ema, dd, ss, nm, dev, cfg["eval_tau"])
            if nm in ("in_val", "out_sc"):
                ev[f"samp_{nm}"] = TR.eval_samples(ema, dd, ss, nm, dev)
        TR.jl(ev_p, ev)
        say("eval", json.dumps(ev))
        torch.save({"ema": ema.state_dict(), "step": step, "model_cfg": mcfg}, os.path.join(a.out, f"ema_{step:06d}.pt"))
        return ev

    if a.smoke == "none" and rank == 0 and step0 == 0:
        evaluate(0)
    if ddp and a.smoke == "none":
        dist.barrier()
    model.train()
    t_last, losses, first_loss = time.time(), [], None
    for step in range(step0, steps):
        lr = cfg["lr"] if a.smoke == "overfit" else TR.lr_at(step, cfg)
        for gp in opt.param_groups:
            gp["lr"] = lr
        parts = fixed if fixed is not None else draw()
        bt = TD.build_mixed(model, parts, dev)
        V = bt["cond"].shape[0]
        eps = torch.randn(V, 2, M.GH, M.GW_PAD, device=dev)
        tau = torch.rand(V, device=dev)
        esum, nray = VD.fm_sum(model, bt, eps, tau)
        n_in = sum(len(r) for *_, r in parts[0][1])
        with torch.no_grad():
            lm = bt["lmask"].float()
            dom_rays = (float(lm[:n_in].sum()), float(lm[n_in:].sum()))
        if ddp:
            tot = torch.stack([esum.detach(), nray]).double()
            dist.all_reduce(tot)
            loss_g, nray = float(tot[0] / (2 * tot[1].clamp(min=1))), tot[1]
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
            gnorm = {k: float(sum((p_.grad.float() ** 2).sum() for p_ in getattr(model, k).parameters() if p_.grad is not None) ** 0.5)
                     for k in ("encoder", "rays", "unet")}
            run["grad_check_step10"] = {"norms": gnorm, "pass": all(v > 0 and math.isfinite(v) for v in gnorm.values())}
            say("grad check", run["grad_check_step10"])
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm.detach(), 1.0 - cfg["ema"])
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        losses.append(float(loss))
        first_loss = first_loss if first_loss is not None else float(loss)
        if (step + 1) % 50 == 0 or step == step0:
            dt = (time.time() - t_last) / (50 if step > step0 else 1)
            t_last = time.time()
            rec = {"step": step + 1, "loss": float(np.mean(losses[-50:])), "lr": lr, "grad_norm": float(gn), "views": V, "views_indoor": n_in,
                   "rays_indoor": dom_rays[0], "rays_outdoor": dom_rays[1], "s_per_step": dt, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            if rank == 0:
                TR.jl(log_p, rec)
            if (step + 1) % 500 == 0 or step == step0:
                say(json.dumps(rec))
        if a.smoke == "none" and rank == 0:
            if (step + 1) % cfg["eval_every"] == 0 or step + 1 == steps:
                evaluate(step + 1)
            if (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps:
                TR.save_ckpt(ck_last, model, ema, opt, step + 1, rng)
        if ddp and a.smoke == "none" and ((step + 1) % cfg["eval_every"] == 0 or (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps):
            dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return
    run["finished"] = time.ctime()
    run["final_loss_mean50"] = float(np.mean(losses[-50:])) if losses else None
    if a.smoke == "overfit":
        bt = TD.build_mixed(model, fixed, dev)
        x = M.sample(model, bt["cond"], bt["m_unk"], torch.randn(bt["cond"].shape[0], 2, M.GH, M.GW_PAD, device=dev))
        lm = bt["lmask"]
        agree = float(((x[:, 0] > 0) == (bt["x1"][:, 0] > 0))[lm].float().mean())
        drop = 1.0 - float(np.mean(losses[-20:])) / first_loss
        run["overfit"] = {"first_loss": first_loss, "last20_loss": float(np.mean(losses[-20:])), "drop": drop, "sign_agree": agree}
        say("overfit", run["overfit"])
    if a.smoke == "short":
        run["short"] = {"first100": float(np.mean(losses[:100])), "last100": float(np.mean(losses[-100:]))}
        run["short"]["pass"] = run["short"]["last100"] < run["short"]["first100"]
        say("short", run["short"])
    run["code_sha256"] = VF.code_fingerprint()
    json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}.json"), "w"), indent=1, default=float)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
