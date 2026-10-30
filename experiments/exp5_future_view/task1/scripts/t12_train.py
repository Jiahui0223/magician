"""EXP5 Task 1.2 training of the V2 model (seen2scene env, 1 GPU; protocol §3, configs/train_*.json, model_v1.json).

  --smoke overfit : one fixed 64-view batch, 200 steps (loss drop, sign agreement of sampled b; gradient check at step 10)
  --smoke short   : 2,000 steps of the real recipe (loss decreases, no NaN)
  (default)       : the full run; EMA eval every eval_every (deterministic FM loss), sample metrics every
                    sample_eval_every (arclength coordinates), checkpoints every ckpt_every, final = EMA@steps.
Resumes from the last checkpoint of --out. Logs: <out>/log.jsonl, <out>/eval.jsonl, <out>/run.json.
Multi-GPU (torchrun): each of W ranks draws batch_views / W views (states_per_batch / W states); the loss of a rank is
its error sum divided by the GLOBAL number of labelled rays and the gradients are summed over ranks (one flat
all-reduce), i.e. exactly the gradient of the single-GPU 64-view batch; parameters start identical (broadcast).
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
import v2_frozen as VF  # noqa: E402
import v2_data as VD  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import model as M  # noqa: E402


def jl(p, obj):
    with open(p, "a") as f:
        f.write(json.dumps(obj, default=float) + "\n")


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def gen(dev, key):
    g = torch.Generator(device=dev)
    g.manual_seed(VD.seed_of(key) % (2 ** 63))
    return g


def chunks(states, max_views=64):
    cur, n = [], 0
    for s in states:
        if n + len(s[3]) > max_views and cur:
            yield cur
            cur, n = [], 0
        cur.append(s)
        n += len(s[3])
    if cur:
        yield cur


@torch.no_grad()
def eval_fm(model, data, states, name, dev, taus):
    model.eval()
    num = torch.zeros((), dtype=torch.float64, device=dev)
    den = torch.zeros((), dtype=torch.float64, device=dev)
    for ch in chunks(states):
        bt = VD.build_cond(model, data, ch, dev)
        V = bt["cond"].shape[0]
        g = gen(dev, f"exp5-V2|evalfm|{name}|{ch[0][0]}|{ch[0][1]}")
        eps = torch.randn(V, 2, M.GH, M.GW_PAD, generator=g, device=dev)
        for tau in taus:
            e, c = VD.per_view_loss(model, bt, eps, torch.full((V,), tau, device=dev))
            num += e.sum().double()
            den += c.sum().double()
    model.train()
    return float(num / den.clamp(min=1))


@torch.no_grad()
def eval_samples(model, data, states, name, dev, K=4, steps=10):
    """Sample metrics on M_unknown rays with a valid label, in unknown-arclength coordinates (voxels)."""
    model.eval()
    acc = {"brier": [0.0, 0.0], "fw": [0.0, 0.0], "ph": [0.0, 0.0], "lh": [0.0, 0.0], "es": [0.0, 0.0],
           "spread": [0.0, 0.0], "ambig": [0.0, 0.0], "s_sq": [0.0, 0.0], "s_sum": [0.0, 0.0]}
    for ch in chunks(states, 32):
        bt = VD.build_cond(model, data, ch, dev)
        V = bt["cond"].shape[0]
        g = gen(dev, f"exp5-V2|evalsample|{name}|{ch[0][0]}|{ch[0][1]}")
        eps = torch.randn(K, V, 2, M.GH, M.GW_PAD, generator=g, device=dev).reshape(K * V, 2, M.GH, M.GW_PAD)
        cond = bt["cond"].repeat(K, 1, 1, 1)
        mu = bt["m_unk"].repeat(K, 1, 1)
        x = M.sample(model, cond, mu, eps, steps).reshape(K, V, 2, M.GH, M.GW_PAD)
        x = M.unpad_rays(x)                                              # [K, V, 2, R]
        b, y = x[:, :, 0], x[:, :, 1]
        lm = M.unpad_rays(bt["lmask"])                                   # [V, R]
        cat = bt["cat"].reshape(V, -1)
        lab = (cat == G.CAT["HIT"]) & lm
        hit = (b > 0) & (y < 1.0)
        p = hit.float().mean(0)
        s = ((y + 1) / 2).clamp(0, 1)
        lu = bt["lu_vox"].reshape(V, -1)
        ell = torch.where(hit, s, torch.ones_like(s)) * lu
        s_lab = ((M.unpad_rays(bt["x1"])[:, 1] + 1) / 2).clamp(0, 1)
        ell_t = torch.where(lab, s_lab, torch.ones_like(s_lab)) * lu
        nohit = (cat == G.CAT["NOHIT"]) & lm
        for v in range(V):
            m = lm[v]
            if not m.any():
                continue
            rms = lambda a: torch.sqrt((a[..., m] ** 2).mean(-1))      # noqa: E731
            e1 = rms(ell[:, v] - ell_t[v][None]).mean()
            pair = sum(rms(ell[i, v] - ell[j, v]) for i in range(K) for j in range(K) if i != j)
            acc["es"][0] += float(e1 - pair / (2 * K * (K - 1)))
            acc["es"][1] += 1
        acc["brier"][0] += float(((p - lab.float()) ** 2)[lm].sum()); acc["brier"][1] += float(lm.sum())
        acc["fw"][0] += float(p[nohit].sum()); acc["fw"][1] += float(nohit.sum())
        acc["ph"][0] += float(p[lm].sum()); acc["ph"][1] += float(lm.sum())
        acc["lh"][0] += float(lab[lm].float().sum()); acc["lh"][1] += float(lm.sum())
        pd = torch.stack([(ell[i] - ell[j]).abs() for i in range(K) for j in range(i + 1, K)]).mean(0)
        acc["spread"][0] += float(pd[lm].sum()); acc["spread"][1] += float(lm.sum())
        acc["ambig"][0] += float((b.abs() < 0.5)[:, lm].float().sum()); acc["ambig"][1] += float(lm.sum()) * K
        hs = s[hit & lm[None]]
        acc["s_sum"][0] += float(hs.sum()); acc["s_sum"][1] += float(hs.numel())
        acc["s_sq"][0] += float((hs ** 2).sum()); acc["s_sq"][1] += float(hs.numel())
    model.train()
    r = {k: (v[0] / v[1] if v[1] else None) for k, v in acc.items() if k not in ("s_sq", "s_sum")}
    n = acc["s_sum"][1]
    r["s_std_hits"] = math.sqrt(max(0.0, acc["s_sq"][0] / n - (acc["s_sum"][0] / n) ** 2)) if n else None
    r["hit_ratio"] = r["ph"] / r["lh"] if r.get("lh") else None
    return {"brier": r["brier"], "false_wall": r["fw"], "pred_hit_rate": r["ph"], "label_hit_rate": r["lh"],
            "hit_ratio": r["hit_ratio"], "fair_es_arclength_vox": r["es"], "spread_vox": r["spread"],
            "ambiguous_bit_rate": r["ambig"], "s_std_hits": r["s_std_hits"]}


def save_ckpt(p, model, ema, opt, step, rng, extra=None):
    tmp = p + ".tmp"
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                "np_rng": rng.bit_generator.state, "torch_cuda_rng": torch.cuda.get_rng_state(),
                "torch_rng": torch.get_rng_state(), **(extra or {})}, tmp)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", choices=["none", "overfit", "short"], default="none")
    ap.add_argument("--smoke-steps", type=int, default=None, help="diagnostic override of the smoke length")
    a = ap.parse_args()
    import torch.distributed as dist
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    VF.verify_frozen()
    cfg = VF.load_cfg(a.config)
    mcfg = VF.load_cfg(cfg["model"])
    dcfg = VF.load_cfg(cfg["data"])
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log_p, ev_p = os.path.join(a.out, "log.jsonl"), os.path.join(a.out, "eval.jsonl")
    say = (lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)) if rank == 0 else (lambda *x: None)  # noqa: E731
    train_items, held_items = VD.load_split(cfg["train_set"])
    if a.smoke != "none":                         # smoke: a few trajectories whose T1.1 shards exist
        built = lambda x: os.path.exists(os.path.join(VF.WORK, "data", "views", x[0], f"{x[1]}.npz"))  # noqa: E731
        train_items = [x for x in train_items if built(x)][:8]
        held_items = [x for x in held_items if built(x)][:3]
    say(f"loading {len(train_items)} train / {len(held_items)} held-out trajectories")
    data = VD.ViewData(train_items, log=say)
    held = VD.ViewData(held_items, log=say)
    say(f"train views {data.n_views}, states {len(data.states)}; held-out views {held.n_views}")
    sets = {"train_sub": (data, data.fixed_states(f"exp5-V2|eval|train_sub|{cfg['train_set']}", cfg["eval_sets"]["train_sub"])),
            "sc_sub": (held, held.fixed_states("exp5-V2|eval|sc_sub", cfg["eval_sets"]["sc_sub"], per_scene=True))}
    if "pisa_sub" in cfg["eval_sets"]:
        pis = VD.ViewData([x for x in held_items if x[0] == "pisa"], log=say)
        sets["pisa_sub"] = (pis, pis.fixed_states("exp5-V2|eval|pisa_sub", cfg["eval_sets"]["pisa_sub"]))
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
    n_par = {k: sum(p_.numel() for p_ in getattr(model, k).parameters()) for k in ("encoder", "rays", "unet")}
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng([cfg["seed"], rank])
    step0 = 0
    ck_last = os.path.join(a.out, "ckpt_last.pt")
    if os.path.exists(ck_last) and a.smoke == "none":
        ck = torch.load(ck_last, map_location=dev)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        if ddp:                                   # per-rank streams re-seeded deterministically at the resume step
            rng = np.random.default_rng([cfg["seed"], rank, step0])
            torch.manual_seed(cfg["seed"] + 1000 * rank + 7 * step0)
        else:
            rng.bit_generator.state = ck["np_rng"]
            torch.cuda.set_rng_state(ck["torch_cuda_rng"].cpu()); torch.set_rng_state(ck["torch_rng"].cpu())
        say(f"resumed at step {step0}")
    run = {"config": a.config, "config_sha256": VF.sha256_file(os.path.join(VF.T1, "configs", a.config)),
           "model_config_sha256": VF.sha256_file(os.path.join(VF.T1, "configs", cfg["model"])), "params": n_par,
           "train_trajs": len(train_items), "train_views": data.n_views, "train_states": len(data.states),
           "data_files_sha256": {**data.files, **held.files}, "smoke": a.smoke, "host": os.uname()[1],
           "gpu": torch.cuda.get_device_name(0), "started": time.ctime(), "resumed_at": step0, "world_size": world}
    say("params", n_par)
    weights = data.state_weights(dcfg["state_weight_t_le"], dcfg["state_weight_factor"])
    steps = {"none": cfg["steps"], "overfit": cfg["smoke"]["overfit_steps"], "short": cfg["smoke"]["short_steps"]}[a.smoke]
    if a.smoke != "none" and a.smoke_steps:
        steps = a.smoke_steps
    fixed = data.fixed_states("exp5-V2|smoke|overfit", cfg["batch_views"]) if a.smoke == "overfit" else None
    model.train()
    t_last, losses, first_loss = time.time(), [], None
    for step in range(step0, steps):
        lr = cfg["lr"] if a.smoke == "overfit" else lr_at(step, cfg)
        for gp in opt.param_groups:
            gp["lr"] = lr
        batch = fixed if fixed is not None else data.sample_batch(rng, cfg["batch_views"] // world, cfg["states_per_batch"] // world,
                                                                  dcfg["max_cands_per_state"], cfg["stale_p"],
                                                                  cfg["stale_max"], weights)
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
        if first_loss is None:
            first_loss = float(loss)
        if (step + 1) % 50 == 0 or step == step0:
            dt = (time.time() - t_last) / (50 if step > step0 else 1)
            t_last = time.time()
            rec = {"step": step + 1, "loss": float(np.mean(losses[-50:])), "lr": lr, "grad_norm": float(gn), "views": V,
                   "rays": float(nray), "s_per_step": dt, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            if rank == 0:
                jl(log_p, rec)
            if (step + 1) % 500 == 0 or step == step0:
                say(json.dumps(rec))
        if a.smoke == "none" and rank == 0:
            if (step + 1) % cfg["eval_every"] == 0 or step + 1 == steps:
                ev = {"step": step + 1}
                for nm, (dd, ss) in sets.items():
                    ev[f"fm_{nm}"] = eval_fm(ema, dd, ss, nm, dev, cfg["eval_tau"])
                if (step + 1) % cfg["sample_eval_every"] == 0 or step + 1 == steps:
                    for nm, (dd, ss) in sets.items():
                        ev[f"samp_{nm}"] = eval_samples(ema, dd, ss, nm, dev)
                jl(ev_p, ev)
                say("eval", json.dumps(ev))
            if (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps:
                save_ckpt(ck_last, model, ema, opt, step + 1, rng)
                pk = os.path.join(a.out, f"ema_{step + 1:06d}.pt")
                torch.save({"ema": ema.state_dict(), "step": step + 1, "model_cfg": mcfg}, pk)
        if ddp and a.smoke == "none" and ((step + 1) % cfg["eval_every"] == 0 or (step + 1) % cfg["ckpt_every"] == 0
                                          or step + 1 == steps):
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
        run["overfit"] = {"first_loss": first_loss, "last20_loss": float(np.mean(losses[-20:])), "drop": drop,
                          "sign_agree": agree, "pass": drop > cfg["smoke"]["overfit_loss_drop"] and agree > cfg["smoke"]["overfit_sign_agree"]}
        say("overfit", run["overfit"])
    if a.smoke == "short":
        run["short"] = {"first100": float(np.mean(losses[:100])), "last100": float(np.mean(losses[-100:]))}
        run["short"]["pass"] = run["short"]["last100"] < run["short"]["first100"]
        say("short", run["short"])
    if a.smoke == "none":
        final = os.path.join(a.out, f"ema_{steps:06d}.pt")
        run["final_checkpoint"] = os.path.relpath(final, VF.REPO)
        run["final_checkpoint_sha256"] = VF.sha256_file(final)
    run["code_sha256"] = VF.code_fingerprint()
    json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}.json"), "w"), indent=1, default=float)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
