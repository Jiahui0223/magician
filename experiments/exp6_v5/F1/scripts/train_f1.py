"""EXP6 V5 / F1 -- matched 5k-step training of the two arms from the V2-Mix B0 EMA@17.5k (seen2scene env; torchrun, 4 ranks).
Adapted from experiments/exp6_v5/scripts/train_v5.py (same data loader, sampler, FM loss, EMA, DDP gradient sum, resume
rule); differences: parent = B0 (not V4), training data = the S10 shards of Train-6 minus S-c (not D100), no L_surf /
L_bridge / teacher / multi-surface labels.
  A  F1-A Continue: the parent V2Model (Encoder3D + Ray32/Conv1D RayEncoder + UNet2D), all parameters, lr 2e-5.
  B  F1-B Geometry Query: V5Model (seen2scene/futureview/v5_model.py) with encoder / unet copied exactly from the parent
     (strict name + shape; torch.equal checked), new V5RayDecoder (seed 0); stage 1 (steps < 2000) encoder / unet frozen,
     stage 2 all trained (inherited lr 2e-5 from step 2000, decoder lr 1e-4).
Both arms: same per-rank sample stream (numpy [seed, rank]) and noise stream (torch seed + 1000 rank, reset after model
construction) -> identical batches / eps / tau; a per-step batch digest is logged (digest.jsonl) to verify it.
Outputs <out>/: ema_<step>.pt (EMA at ema_save_steps; keys ema / step / model_cfg / provenance), ckpt_last.pt (resume),
log.jsonl, eval.jsonl, digest.jsonl, run_*.json.
Usage: torchrun --standalone --nproc_per_node 4 train_f1.py --arm A|B --out DIR [--smoke overfit|short|profile|resume]
       [--set key=value ...]"""
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
F1 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(F1)))
EXP5 = os.path.join(R, "experiments", "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
sys.path.insert(0, os.path.join(EXP5, "task1", "scripts"))
import v2_frozen as VF  # noqa: E402
import v2_data as VD  # noqa: E402
import t12_train as TR  # noqa: E402
import t22_data as TD  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.futureview import v5_model as V5  # noqa: E402

CFG = os.path.join(F1, "configs")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=R).decode().strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=R)
        return {"sha": sha, "dirty": bool(diff), "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None}
    except Exception as e:                       # noqa: BLE001
        return {"error": str(e)}


def lr_cos(step, start, warm, total, peak):
    if step < start:
        return 0.0
    s = step - start
    if s < warm:
        return peak * (s + 1) / warm
    p = (s - warm) / max(1, total - start - warm)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def load_cfg(arm, sets):
    common = yaml.safe_load(open(os.path.join(CFG, "f1_common.yaml")))
    armf = "f1_a_baseline.yaml" if arm == "A" else "f1_b_geometry_query.yaml"
    cfg = dict(common, **yaml.safe_load(open(os.path.join(CFG, armf))))
    for kv in sets:
        k, v = kv.split("=", 1)
        cur = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            cur = cur[kk]
        cur[ks[-1]] = yaml.safe_load(v)
    files = {f: sha256_file(os.path.join(CFG, f)) for f in ("f1_common.yaml", armf)}
    chash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    return cfg, files, chash


def build_model(arm, cfg, parent):
    """-> model, model_cfg, warm-start report. Arm B: V5Model with the parent encoder / unet (exact copy)."""
    pcfg = parent["model_cfg"]
    torch.manual_seed(cfg["seed"])
    if arm == "A":
        model = M.V2Model(pcfg)
        res = model.load_state_dict(parent["ema"], strict=True)
        return model, pcfg, {"strict": True, "missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys), "copied": len(parent["ema"])}
    base = json.load(open(os.path.join(R, cfg["model_cfg_base"])))
    mcfg = copy.deepcopy(pcfg)
    mcfg["version"] = "model_f1_gq_v1"
    mcfg["protocol"] = "exp6_v5/F1: parent V2-Mix encoder + UNet2D (exact copy) + V5RayDecoder (v5_model.py, unchanged code)"
    mcfg["arch"] = "v5"
    mcfg["ray"] = {"out_ch": pcfg["ray"]["out_ch"], "doc": "64 ray channels produced by the V5 CondAdapter; RayEncoder not used"}
    mcfg["v5"] = dict(base["v5"], **{k: v for k, v in cfg["v5_overrides"].items() if v is not None})
    assert mcfg["v5"].get("out_scale_init") is not None or cfg.get("_smoke_ok_without_scale"), "out_scale_init must be frozen in f1_b_geometry_query.yaml"
    for k in ("encoder", "unet", "proj_ch", "flow"):
        assert mcfg[k] == base[k] or k == "flow" and mcfg[k]["x_ch"] == base[k]["x_ch"], ("parent / V5 config mismatch", k)
    model = V5.V5Model(mcfg)
    ws = V5.load_v4_warmstart(model, parent["ema"])
    sd = model.state_dict()
    neq = [k for k, v in parent["ema"].items() if (k.startswith("encoder.") or k.startswith("unet.")) and not torch.equal(sd[k], v)]
    assert not neq, ("inherited tensors differ after the copy", neq[:5])
    ws["torch_equal_inherited"] = True
    ws["not_copied_parent_keys"] = sorted(k for k in parent["ema"] if not (k.startswith("encoder.") or k.startswith("unet.")))
    ws["not_copied_parent_numel"] = int(sum(parent["ema"][k].numel() for k in ws["not_copied_parent_keys"]))
    return model, mcfg, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", choices=["none", "overfit", "short", "profile", "resume"], default="none")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    import torch.distributed as dist
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=90))
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    cfg, cfg_files, chash = load_cfg(a.arm, a.set)
    if a.smoke == "none":
        assert world == cfg["world_size"], f"formal runs need world size {cfg['world_size']}"
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    log_p, ev_p, dg_p = (os.path.join(a.out, f) for f in ("log.jsonl", "eval.jsonl", "digest.jsonl"))
    say = (lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)) if rank == 0 else (lambda *x: None)  # noqa: E731
    dcfg = VF.load_cfg(cfg["data"]["config"])
    train_items, held_items = VD.load_split(cfg["data"]["train_set"])
    if a.smoke != "none":
        train_items, held_items = train_items[:8], held_items[:3]
    say(f"arm {a.arm}: loading {len(train_items)} train / {len(held_items)} S-c trajectories (S10 shards)")
    data = TD.CompactViewData(train_items, log=say)
    held = TD.CompactViewData(held_items, log=say)
    say(f"train views {data.n_views}, states {len(data.states)}; S-c views {held.n_views}")
    sets = {"train_sub": (data, data.fixed_states("exp5-V2|eval|train_sub|train6_minus_Sc", cfg["eval_sets"]["train_sub"])),
            "sc_sub": (held, held.fixed_states("exp5-V2|eval|sc_sub", cfg["eval_sets"]["sc_sub"], per_scene=True))}
    pp = os.path.join(R, cfg["parent"]["ckpt"])
    psha = sha256_file(pp)
    assert psha == cfg["parent"]["sha256"], ("parent checkpoint sha256 mismatch", psha)
    parent = torch.load(pp, map_location="cpu", weights_only=False)
    p_mtime = os.path.getmtime(pp)
    model, mcfg, ws = build_model(a.arm, cfg, parent)
    model = model.to(dev)
    if ddp:
        for p_ in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(p_.data, 0)
    torch.manual_seed(cfg["seed"] + 1000 * rank)            # noise stream identical in both arms from here on
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    if a.arm == "B":
        model.dec_ckpt = True
        new_p = [p_ for n_, p_ in model.named_parameters() if n_.startswith("dec.")]
        inh_p = [p_ for n_, p_ in model.named_parameters() if not n_.startswith("dec.")]
    else:
        model.rays_ckpt = False
        new_p, inh_p = [], list(model.parameters())
    groups = ([{"params": new_p, "lr": 0.0, "name": "new"}] if new_p else []) + [{"params": inh_p, "lr": 0.0, "name": "inherited"}]
    opt = torch.optim.AdamW(groups, betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng([cfg["seed"], rank])
    step0 = 0
    ck_last = os.path.join(a.out, "ckpt_last.pt")
    if os.path.exists(ck_last) and a.smoke in ("none", "resume"):
        ck = torch.load(ck_last, map_location=dev, weights_only=False)
        assert ck.get("config_hash") == chash, "resume with a different config"
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        if ddp:                                   # per-rank streams re-seeded at the resume step (the V2 / V4 / V5 DDP rule)
            rng = np.random.default_rng([cfg["seed"], rank, step0])
            torch.manual_seed(cfg["seed"] + 1000 * rank + 7 * step0)
        else:
            rng.bit_generator.state = ck["np_rng"]
            torch.cuda.set_rng_state(ck["torch_cuda_rng"].cpu()); torch.set_rng_state(ck["torch_rng"].cpu())
        say(f"resumed at step {step0}")
    npar = {"total": sum(p_.numel() for p_ in model.parameters()), "new": sum(p_.numel() for p_ in new_p), "inherited": sum(p_.numel() for p_ in inh_p)}
    prov = {"parent_ckpt": cfg["parent"]["ckpt"], "parent_sha256": psha, "config_hash": chash, "config_files_sha256": cfg_files,
            "model_cfg_sha256": hashlib.sha256(json.dumps(mcfg, sort_keys=True).encode()).hexdigest(), "arm": a.arm, "model_id": cfg["model_id"]}
    run = {**prov, "train_cfg": cfg, "model_cfg": mcfg, "warm_start": ws, "params": npar, "sets": a.set, "smoke": a.smoke, "git": git_info(),
           "host": os.uname()[1], "gpu": torch.cuda.get_device_name(0), "world_size": world, "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "train_trajs": len(train_items), "train_views": data.n_views, "data_files_sha256": {**data.files, **held.files},
           "started": time.ctime(), "resumed_at": step0, "domains": ["outdoor"], "mp3d_training_samples": 0}
    if rank == 0:
        json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}_{int(time.time())}.json"), "w"), indent=1, default=str)
    say("params", npar, "| warm-start", {k: v for k, v in ws.items() if k not in ("not_copied_parent_keys",)})
    weights = data.state_weights(dcfg["state_weight_t_le"], dcfg["state_weight_factor"])
    sm = cfg["smoke"]
    steps = {"none": cfg["steps"], "resume": cfg["steps"], "overfit": sm["overfit_steps"], "short": sm["short_steps"], "profile": 40}[a.smoke]
    stage1 = (sm["short_stage1"] if a.smoke == "short" else cfg["stage1"]["steps"]) if a.arm == "B" else 0
    eff_world = world if a.smoke == "none" else max(world, int(sm.get("emulate_world", 4)))
    bv, bs = cfg["batch_views"] // eff_world, max(1, cfg["states_per_batch"] // eff_world)
    fixed = data.fixed_states("exp6-F1|smoke|overfit", bv) if a.smoke == "overfit" else None
    ema_steps = set(cfg["ema_save_steps"])

    def set_phase(step):
        if a.arm == "B":
            for p_ in inh_p:
                p_.requires_grad_(step >= stage1)

    def lrs(step):
        out = {}
        for g in opt.param_groups:
            if g["name"] == "new":
                g["lr"] = lr_cos(step, 0, cfg["warmup_new"], steps, cfg["lr_new"])
            elif a.arm == "B":
                g["lr"] = lr_cos(step, stage1, cfg["stage2"]["inherited_warmup"], steps, cfg["stage2"]["inherited_lr"])
            else:
                g["lr"] = lr_cos(step, 0, cfg["warmup"], steps, cfg["lr"])
            out[g["name"]] = g["lr"]
        return out

    def batch_forward(batch, want_stats=False):
        conds, keep = [], {"m_unk": [], "lmask": [], "x1": [], "cat": [], "lu_vox": []}
        stats = {}
        for ti, t, dlt, rows in batch:
            st = VD.state_tensors(data, ti, t, dlt, rows, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.encode(st["cin"])
                if a.arm == "B":
                    c = model.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"])
                    if want_stats and not stats:
                        Hc = c[:, mcfg["proj_ch"]:mcfg["proj_ch"] + 64, :, :M.GW].float()
                        stats = {"H_mean": float(Hc.mean()), "H_std": float(Hc.std()), "H_rms": float(Hc.pow(2).mean().sqrt()), "H_finite": bool(torch.isfinite(Hc).all())}
                    conds.append(c)
                else:
                    conds.append(model.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"]))
            for k in keep:
                keep[k].append(st[k])
        bt = {k: torch.cat(v) for k, v in keep.items()}
        bt["cond"] = torch.cat(conds)
        return bt, stats

    def save_ema(step):
        torch.save({"ema": ema.state_dict(), "step": step, "model_cfg": mcfg, "arm": a.arm, "provenance": prov,
                    "lineage": f"V2O EMA@60k -> V2-Mix EMA@17.5k (B0) -> F1-{a.arm} +{step} (outdoor S10, L_FM)"},
                   os.path.join(a.out, f"ema_{step:06d}.pt"))

    def evaluate(step):
        ev = {"step": step}
        for nm, (dd_, ss) in sets.items():
            ev[f"fm_{nm}"] = TR.eval_fm(ema, dd_, ss, nm, dev, cfg["eval_tau"])
        ev["samp_sc_sub"] = TR.eval_samples(ema, sets["sc_sub"][0], sets["sc_sub"][1], "sc_sub", dev)
        TR.jl(ev_p, ev)
        say("eval", json.dumps(ev, default=float))

    if a.smoke == "none" and rank == 0 and step0 == 0:
        save_ema(0)
        evaluate(0)
    if ddp and a.smoke == "none":
        dist.barrier()
    model.train()
    t_last, losses, first_loss, prof = time.time(), [], None, []
    for step in range(step0, steps):
        set_phase(step)
        lr = lrs(step)
        t0 = time.time()
        batch = fixed if fixed is not None else data.sample_batch(rng, bv, bs, dcfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], weights)
        want = (step + 1) % 50 == 0 or step == step0
        bt, hstats = batch_forward(batch, want)
        V = bt["cond"].shape[0]
        eps = torch.randn(V, 2, M.GH, M.GW_PAD, device=dev)
        tau = torch.rand(V, device=dev)
        if rank == 0 and a.smoke in ("none", "resume", "short"):
            dg = hashlib.sha256(json.dumps([[int(ti), int(t), int(dlt), [int(x) for x in np.asarray(rows).ravel()]] for ti, t, dlt, rows in batch]).encode()
                                + tau.cpu().numpy().tobytes() + eps[:, :, 0, :8].cpu().numpy().tobytes()).hexdigest()[:16]
            TR.jl(dg_p, {"step": step, "digest": dg})
        esum, nray = VD.fm_sum(model, bt, eps, tau)
        tot = torch.stack([esum.detach(), nray]).double()
        if ddp:
            dist.all_reduce(tot)
        nray_g = tot[1].clamp(min=1.0)
        loss = esum / (2.0 * nray_g.to(esum.dtype))
        if not torch.isfinite(loss):
            raise RuntimeError(f"STOP: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        params = [p_ for p_ in model.parameters() if p_.requires_grad]
        if ddp:
            for p_ in params:
                if p_.grad is None:
                    p_.grad = torch.zeros_like(p_)
            flat = torch._utils._flatten_dense_tensors([p_.grad for p_ in params])
            dist.all_reduce(flat)
            for g_, s_ in zip(params, torch._utils._unflatten_dense_tensors(flat, [p_.grad for p_ in params])):
                g_.grad.copy_(s_)
        gparts = None
        if want:
            gparts = {}
            for nme, pre in (("dec", "dec."), ("encoder", "encoder."), ("unet", "unet."), ("rays", "rays.")):
                gs = [p_.grad.float().norm() ** 2 for n_, p_ in model.named_parameters() if n_.startswith(pre) and p_.grad is not None]
                if gs:
                    gparts[nme] = float(torch.stack(gs).sum().sqrt())
        gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm.detach(), 1.0 - cfg["ema"])
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        total = float(loss.detach()) * (world if ddp else 1)
        losses.append(total)
        first_loss = first_loss if first_loss is not None else total
        if a.smoke == "profile":
            torch.cuda.synchronize(); prof.append(time.time() - t0)
        if want:
            dt = (time.time() - t_last) / (50 if step > step0 else 1)
            t_last = time.time()
            rec = {"step": step + 1, "phase": ("stage1" if step < stage1 else "stage2") if a.arm == "B" else "continue", "loss": float(np.mean(losses[-50:])),
                   "fm": float(tot[0] / (2 * nray_g)), "lr": lr, "grad_norm": float(gn), "grad_parts": gparts, "H": hstats, "views": V * world,
                   "rays": float(nray_g), "s_per_step": dt, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                   "n_trainable": int(sum(p_.numel() for p_ in params))}
            if rank == 0:
                TR.jl(log_p, rec)
            if (step + 1) % 250 == 0 or step == step0:
                say(json.dumps(rec, default=float))
        if a.smoke in ("none", "resume") and rank == 0:
            if a.smoke == "none" and ((step + 1) % cfg["eval_every"] == 0 or step + 1 == steps):
                evaluate(step + 1)
            if (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps:
                TR.save_ckpt(ck_last, model, ema, opt, step + 1, rng, {"config_hash": chash, "parent_sha256": psha, "arm": a.arm})
            if a.smoke == "none" and ((step + 1) in ema_steps or step + 1 == steps):
                save_ema(step + 1)
        if ddp and a.smoke in ("none", "resume") and ((step + 1) % cfg["eval_every"] == 0 or (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps):
            dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return
    assert sha256_file(pp) == psha and os.path.getmtime(pp) == p_mtime, "parent checkpoint changed during training"
    run["finished"] = time.ctime()
    run["final_loss_mean50"] = float(np.mean(losses[-50:])) if losses else None
    if a.smoke == "overfit":
        run["overfit"] = {"first_loss": first_loss, "last20_loss": float(np.mean(losses[-20:])), "drop": 1.0 - float(np.mean(losses[-20:])) / first_loss}
        say("overfit", run["overfit"])
    if a.smoke == "profile":
        p_ = np.asarray(prof[5:])
        run["profile"] = {"s_per_step_mean": float(p_.mean()), "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
        say("profile", run["profile"])
    if a.smoke == "short":
        run["short"] = {"first50": float(np.mean(losses[:50])), "stage1_last50": float(np.mean(losses[max(0, stage1 - 50):stage1])) if stage1 else None,
                        "last50": float(np.mean(losses[-50:]))}
        say("short", run["short"])
    if a.smoke == "none":
        final = os.path.join(a.out, f"ema_{steps:06d}.pt")
        run["final_checkpoint"] = os.path.relpath(final, R)
        run["ema_sha256"] = {f: sha256_file(os.path.join(a.out, f)) for f in sorted(os.listdir(a.out)) if f.startswith("ema_")}
    json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}.json"), "w"), indent=1, default=str)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
