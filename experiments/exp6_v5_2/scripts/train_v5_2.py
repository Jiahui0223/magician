"""EXP6 V5.2 -- training of the camera-conditioned ray geometry belief (seen2scene env; torchrun, 4 ranks).

Model  seen2scene/futureview/v52_model.py V52Model: encoder + V5RayDecoder copied strictly from the V5 EMA@12000
       (work_dirs/exp6_v5/checkpoints/v5_final.pt, sha 5ef30ac4...), new HazardHead; no flow U-Net (not loaded, never
       called); the V5 flow-conditioning parts of the decoder (BeliefSummary, CondAdapter, out_scale) are frozen / unused.
Loss   L = L_surface + beta L_firsthit, nothing else (no flow matching, coverage, ranking or KL loss)
         L_surface   the V5 multi-surface BCE (exp6_v5/scripts/train_v5.py surface_queries / surf_loss: stored mesh
                     surfaces jittered, near-surface hard negatives, log-uniform and behind-first queries, balanced over
                     positives / negatives in front of / behind the first surface; outside [t_lo, t_hi] unsupervised)
         L_firsthit  -log p_first[i*] for the interval i* holding the true first hit t* (shard tstar, the T1.1 label
                     camera = the training rays), -log p_nohit when t* is beyond the sensor range tcap (or none); rays
                     with t* before the near clip are masked. GT only in the loss, never an input.
Rays   per view 256 rays of the label cache (surface + first-hit) and 256 uniform rays (first-hit only)
Stages Stage H (steps < hazard_steps): encoder / decoder / surface head frozen, only HazardHead trained;
       Stage J: encoder + decoder (+ surface head) unfrozen at a small lr, hazard continues.
Data   Train-6 minus S-c (138 trajectories, E2 D100 shards), S-c only for monitoring; MP3D 0.
torchrun --standalone --nproc_per_node 4 train_v5_2.py --out DIR [--smoke overfit|short|gradscale|resume] [--override k=v ...]
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
R = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
EXP5 = os.path.join(R, "experiments", "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
sys.path.insert(0, os.path.join(EXP5, "task1", "scripts"))
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v5", "scripts"))
import v2_frozen as VF  # noqa: E402
import v2_data as VD  # noqa: E402
import t12_train as TR  # noqa: E402
import t22_data as TD  # noqa: E402
import train_v5 as TV  # noqa: E402  (Labels, surface_queries, surf_loss, surf_metrics -- the V5 surface supervision)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.futureview import v52_model as V52  # noqa: E402

CFG = os.path.join(R, "experiments", "exp6_v5_2", "configs")
OUTD = os.path.join(R, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2", "out_views_d100")
V5CK = os.path.join(R, "work_dirs", "exp6_v5", "checkpoints", "v5_final.pt")
V5SHA = "5ef30ac4fd0b4a175d1ddd72d78c504c82b31b40b56da68b1d9454533918d543"


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
        untracked = subprocess.check_output(["git", "status", "--porcelain"], cwd=R).decode().splitlines()
        return {"sha": sha, "dirty": bool(diff), "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None, "n_untracked_or_modified": len(untracked)}
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


class ViewDataT(VD.ViewData):
    """v2_data.ViewData + the T1.1 first-hit distances tstar [views, 7296] (float16, model units, inf = none) and
    zlib-compressed coarse grids (t22_data.CompactViewData)."""

    def __init__(self, items, root, log=print):
        super().__init__(items, root=root, keys=VD.VIEW_KEYS + ("tstar",), log=log)
        self.v["tstar"] = self.v["tstar"].astype(np.float16)
        self.coarse = [TD.ZCoarse(c.numpy()) for c in self.coarse]


def firsthit_targets(edges, tstar):
    """edges [N, B + 1], tstar [N] -> idx [N] (interval of t*, B = no hit), valid [N]."""
    B = edges.shape[1] - 1
    t = tstar.float()
    valid = ~(t < edges[:, 0])
    hit = torch.isfinite(t) & (t < edges[:, -1])
    idx = torch.searchsorted(edges.contiguous(), t.clamp(max=1e9)[:, None].contiguous())[:, 0] - 1
    idx = torch.where(hit, idx.clamp(0, B - 1), torch.full_like(idx, B))
    return idx, valid


def firsthit_nll(out, idx, valid):
    """-log p(first hit = idx) per ray (idx = B -> no hit); masked mean and per-ray values."""
    B = out["log_pf"].shape[1]
    lp_hit = torch.gather(out["log_pf"], 1, idx.clamp(max=B - 1)[:, None])[:, 0]
    lp_no = -out["a"].sum(-1)
    lp = torch.where(idx == B, lp_no, lp_hit)
    nll = -lp
    return (nll * valid).sum() / valid.sum().clamp(min=1), nll


@torch.no_grad()
def firsthit_stats(out, idx, valid, tstar):
    B = out["p_first"].shape[1]
    allp = torch.cat([out["p_first"], out["p_nohit"][:, None]], 1)
    top = allp.argmax(1)
    hit = idx < B
    ph = 1.0 - out["p_nohit"]
    mode_t = torch.gather(out["t_mid"], 1, out["p_first"].argmax(1, keepdim=True))[:, 0]
    dm = hit & valid & (ph > 0.5)
    v = valid
    return {"fh_top_acc": float((top == idx)[v].float().mean()) if v.any() else float("nan"),
            "fh_hit_rate": float(hit[v].float().mean()) if v.any() else float("nan"),
            "fh_pred_nohit": float(out["p_nohit"][v].mean()) if v.any() else float("nan"),
            "fh_mode_err_vox": float(((mode_t - tstar.float()).abs() / M.VS)[dm].mean()) if dm.any() else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="train_v52.json")
    ap.add_argument("--smoke", choices=["none", "overfit", "short", "gradscale", "resume"], default="none")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    import torch.distributed as dist
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=90))
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    cfg = json.load(open(os.path.join(CFG, a.config)))
    for kv in a.override:
        k, v = kv.split("=")
        cfg[k] = float(v) if cfg.get(k) is None else type(cfg[k])(v)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    log_p, ev_p = os.path.join(a.out, "log.jsonl"), os.path.join(a.out, "eval.jsonl")
    say = (lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)) if rank == 0 else (lambda *x: None)  # noqa: E731
    dcfg = VF.load_cfg("data_v1.json")
    train_items, held_items = VD.load_split("train6_minus_Sc")
    if a.smoke != "none":
        train_items = [tuple(x) for x in cfg["smoke"]["train_items"]]
        held_items = [tuple(x) for x in cfg["smoke"]["held_items"]]
    say(f"loading {len(train_items)} train / {len(held_items)} S-c trajectories")
    data = ViewDataT(train_items, OUTD, log=say)
    held = ViewDataT(held_items, OUTD, log=say)
    labels, hlabels = TV.Labels(train_items, data), TV.Labels(held_items, held)
    say(f"train views {data.n_views}, states {len(data.states)}; S-c views {held.n_views}; label store {labels.k_store}")
    # ---------------- model
    assert sha256_file(V5CK) == V5SHA, "V5 parent sha256 mismatch"
    ck5 = torch.load(V5CK, map_location="cpu", weights_only=False)
    mcfg = json.load(open(os.path.join(CFG, cfg["model"])))
    torch.manual_seed(cfg["seed"])
    model = V52.V52Model(mcfg)
    ws = V52.load_v5_warmstart(model, ck5["ema"])
    model = model.to(dev)
    if ddp:
        for p_ in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(p_.data, 0)
    torch.manual_seed(cfg["seed"] + 1000 * rank)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    haz_p = list(model.hazard.parameters())
    bb_p = [p_ for n_, p_ in model.named_parameters() if not n_.startswith("hazard.") and p_.requires_grad]
    opt = torch.optim.AdamW([{"params": haz_p, "lr": 0.0, "name": "hazard"}, {"params": bb_p, "lr": 0.0, "name": "backbone"}],
                            betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng([cfg["seed"], rank])
    gen = torch.Generator(device=dev); gen.manual_seed(cfg["seed"] * 7919 + rank)
    step0 = 0
    ck_last = os.path.join(a.out, "ckpt_last.pt")
    if os.path.exists(ck_last) and a.smoke in ("none", "resume"):
        ck = torch.load(ck_last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        if ddp:
            rng = np.random.default_rng([cfg["seed"], rank, step0])
            torch.manual_seed(cfg["seed"] + 1000 * rank + 7 * step0)
            gen.manual_seed(cfg["seed"] * 7919 + rank + 13 * step0)
        else:
            rng.bit_generator.state = ck["np_rng"]
            torch.cuda.set_rng_state(ck["torch_cuda_rng"].cpu()); torch.set_rng_state(ck["torch_rng"].cpu())
            gen.set_state(ck["gen"].cpu() if hasattr(ck["gen"], "cpu") else ck["gen"])
        say(f"resumed at step {step0}")
    npar = {"total": sum(p_.numel() for p_ in model.parameters()), "hazard": sum(p_.numel() for p_ in haz_p),
            "backbone_trainable": sum(p_.numel() for p_ in bb_p), "frozen_unused": sum(p_.numel() for p_ in model.parameters() if not p_.requires_grad)}
    run = {"config": a.config, "config_sha256": sha256_file(os.path.join(CFG, a.config)), "train_cfg": cfg, "model_cfg": mcfg,
           "model_cfg_sha256": hashlib.sha256(json.dumps(mcfg, sort_keys=True).encode()).hexdigest(), "init": {"v5_ckpt": V5CK, "v5_sha256": V5SHA, **ws},
           "params": npar, "overrides": a.override, "smoke": a.smoke, "git": git_info(), "host": os.uname()[1], "gpu": torch.cuda.get_device_name(0),
           "world_size": world, "slurm_job": os.environ.get("SLURM_JOB_ID"), "train_trajs": len(train_items), "train_views": data.n_views,
           "data_files_sha256": {**data.files, **held.files}, "label_meta": labels.meta, "started": time.ctime(), "resumed_at": step0,
           "domains": ["outdoor"], "mp3d_training_samples": 0, "losses": ["L_surface", "beta * L_firsthit"]}
    if rank == 0:
        json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}_{int(time.time())}.json"), "w"), indent=1, default=str)
    say("params", npar, "| warm-start", ws)
    weights = data.state_weights(dcfg["state_weight_t_le"], dcfg["state_weight_factor"])
    steps = {"none": cfg["steps"], "resume": cfg["steps"], "overfit": cfg["smoke"]["overfit_steps"], "short": cfg["smoke"]["short_steps"],
             "gradscale": cfg["smoke"]["gradscale_steps"]}[a.smoke]
    hz_steps = cfg["hazard_steps"]
    eff_world = world if a.smoke == "none" else max(world, int(cfg["smoke"].get("emulate_world", 4)))
    bv, bs = cfg["batch_views"] // eff_world, max(1, cfg["states_per_batch"] // eff_world)
    fixed = data.fixed_states("exp6-V5.2|smoke|overfit", bv) if a.smoke == "overfit" else None
    Delta = cfg["surf_delta_vox"] * M.VS
    beta = cfg["beta"]
    gscale = []

    def set_phase(step):
        fz = step < hz_steps
        for p_ in bb_p:
            p_.requires_grad_(not fz)

    def lrs(step):
        out = {}
        for g in opt.param_groups:
            if g["name"] == "hazard":
                g["lr"] = lr_cos(step, 0, cfg["warmup_hazard"], steps, cfg["lr_hazard"])
            else:
                g["lr"] = lr_cos(step, hz_steps, cfg["warmup_backbone"], steps, cfg["lr_backbone"])
            out[g["name"]] = g["lr"]
        return out

    def state_forward(model_, dset, lab, ti, t, dlt, rows, n_lab, n_uni, g):
        """One state: label rays + uniform rays of each view -> (surface loss terms, first-hit terms)."""
        st = VD.state_tensors(dset, ti, t, dlt, rows, dev)
        V_, R_ = st["d"].shape[:2]
        frozen = not any(p_.requires_grad for p_ in model_.encoder.parameters())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if frozen:
                with torch.no_grad():
                    h = model_.encode(st["cin"])
            else:
                h = model_.encode(st["cin"])
            mem = model_.memory(h)
            ri = torch.as_tensor(lab.v["ray_idx"][rows], device=dev).long()                         # [V, 1024]
            pick = torch.argsort(torch.rand(ri.shape, generator=g, device=dev), 1)[:, :n_lab]
            ri = torch.gather(ri, 1, pick)
            ok_r = (ri >= 0).reshape(-1)
            ri = ri.clamp(min=0)
            ru = torch.randint(0, R_, (V_, n_uni), generator=g, device=dev)
            rr = torch.cat([ri, ru], 1)                                                               # [V, n_lab + n_uni]
            vv = torch.arange(V_, device=dev)[:, None].expand_as(rr)
            oo = st["o"][vv.reshape(-1)]
            dd = st["d"][vv.reshape(-1), rr.reshape(-1)]
            tc = st["tcap"][vv.reshape(-1), rr.reshape(-1)]
            chh = st["ch"].transpose(1, 2)[vv.reshape(-1), rr.reshape(-1)].float()
            out = model_.belief(mem, oo.float(), dd.float(), tc.float(), chh)
            # surface supervision on the label rays (first n_lab of every view)
            is_lab = torch.zeros_like(rr, dtype=torch.bool); is_lab[:, :n_lab] = True
            is_lab = is_lab.reshape(-1)
            surf = torch.as_tensor(lab.v["surf"][rows], device=dev).float()
            surf = torch.gather(surf, 1, pick[..., None].expand(-1, -1, surf.shape[-1])).reshape(-1, surf.shape[-1])
            tlo = torch.gather(torch.as_tensor(lab.v["t_lo"][rows], device=dev).float(), 1, pick).reshape(-1)
            thi = torch.gather(torch.as_tensor(lab.v["t_hi"][rows], device=dev).float(), 1, pick).reshape(-1)
            thi = torch.where(ok_r, thi, torch.zeros_like(thi))
            tq, y, vq, order, first = TV.surface_queries(surf, tlo, thi, g, cfg["surf_rand_queries"], Delta)
            lg = model_.dec.surface_logits(out["z"][is_lab], tq, tc[is_lab].float()).float()
        ts = torch.as_tensor(dset.v["tstar"][rows], device=dev)[vv.reshape(-1), rr.reshape(-1)]
        ts = torch.where(torch.isfinite(ts.float()), ts.float(), torch.full_like(ts.float(), float("inf")))
        idx, valid = firsthit_targets(out["edges"], ts)
        return {"lg": lg, "y": y, "vq": vq, "order": order, "tq": tq, "first": first, "out": out, "idx": idx, "valid": valid, "ts": ts}

    @torch.no_grad()
    def eval_sc(model_, step):
        states = held.fixed_states("exp6-V5.2|eval|sc", cfg["eval_views"], per_scene=True)
        g = torch.Generator(device=dev); g.manual_seed(12345)
        acc = {"loss_surf": [], "recall_first": [], "recall_second_plus": [], "neg_acc": [], "neg_acc_behind_first": [], "n_second_plus": [],
               "fh_nll": [], "fh_top_acc": [], "fh_hit_rate": [], "fh_pred_nohit": [], "fh_mode_err_vox": []}
        for ti, t, dlt, rows in states:
            r = state_forward(model_, held, hlabels, ti, t, dlt, rows, 256, 256, g)
            ls, _ = TV.surf_loss(r["lg"], r["y"], r["vq"], r["tq"], r["first"].expand_as(r["tq"]))
            sm = TV.surf_metrics(r["lg"], r["y"], r["vq"], r["order"], r["tq"], r["first"].expand_as(r["tq"]))
            nll, _ = firsthit_nll(r["out"], r["idx"], r["valid"])
            fs = firsthit_stats(r["out"], r["idx"], r["valid"], r["ts"])
            acc["loss_surf"].append(float(ls)); acc["fh_nll"].append(float(nll))
            for k, v in {**sm, **fs}.items():
                acc[k].append(v)
        ev = {"step": step, **{k: (float(np.nansum(v)) if k == "n_second_plus" else float(np.nanmean(v))) for k, v in acc.items()}}
        TR.jl(ev_p, ev)
        say("eval", json.dumps(ev))

    def save_ema(step):
        torch.save({"ema": ema.state_dict(), "step": step, "model_cfg": mcfg, "arm": "v52",
                    "lineage": f"V5 EMA@12000 (v5_final.pt sha {V5SHA[:12]}) -> V5.2 +{step} (hazard {min(step, hz_steps)} + joint {max(0, step - hz_steps)})"},
                   os.path.join(a.out, f"ema_{step:06d}.pt"))

    if a.smoke == "none" and rank == 0 and step0 == 0:
        save_ema(0)
        eval_sc(ema, 0)
    if ddp and a.smoke == "none":
        dist.barrier()
    model.train()
    t_last, losses, first_loss = time.time(), [], None
    for step in range(step0, steps):
        set_phase(step)
        lr = lrs(step)
        batch = fixed if fixed is not None else data.sample_batch(rng, bv, bs, dcfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], weights)
        want = (step + 1) % 50 == 0 or step == step0
        res = [state_forward(model, data, labels, ti, t, dlt, rows, cfg["rays_label"], cfg["rays_uniform"], gen) for ti, t, dlt, rows in batch]
        lg = torch.cat([r["lg"] for r in res]); y = torch.cat([r["y"] for r in res]); vq = torch.cat([r["vq"] for r in res])
        tq = torch.cat([r["tq"] for r in res]); first = torch.cat([r["first"] for r in res]).expand_as(tq); order = torch.cat([r["order"] for r in res])
        ls, _ = TV.surf_loss(lg, y, vq, tq, first)
        outs = {k: torch.cat([r["out"][k] for r in res]) for k in ("a", "log_pf", "p_first", "p_nohit", "t_mid")}
        idx = torch.cat([r["idx"] for r in res]); valid = torch.cat([r["valid"] for r in res]); ts = torch.cat([r["ts"] for r in res])
        lfh, _ = firsthit_nll(outs, idx, valid)
        loss = (ls + beta * lfh) / world
        terms = {"surf": float(ls), "firsthit": float(lfh)}
        if want:
            terms.update(TV.surf_metrics(lg, y, vq, order, tq, first))
            terms.update(firsthit_stats(outs, idx, valid, ts))
        if not torch.isfinite(loss):
            raise RuntimeError(f"STOP: non-finite loss at step {step}: {terms}")
        if a.smoke == "gradscale" and step >= steps - 20:
            shared = [p_ for p_ in bb_p if p_.requires_grad]
            gs = {}
            for nme, term in (("surf", ls), ("firsthit", lfh)):
                gg = torch.autograd.grad(term, shared, retain_graph=True, allow_unused=True)
                gs[nme] = float(torch.sqrt(sum((g_.float() ** 2).sum() for g_ in gg if g_ is not None)))
            gscale.append(gs)
            say("gradscale", step, gs)
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
            for nme, pre in (("hazard", "hazard."), ("dec", "dec."), ("encoder", "encoder.")):
                gsq = [p_.grad.float().norm() ** 2 for n_, p_ in model.named_parameters() if n_.startswith(pre) and p_.grad is not None]
                if gsq:
                    gparts[nme] = float(torch.stack(gsq).sum().sqrt())
        gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm.detach(), 1.0 - cfg["ema"])
        total = float(loss.detach()) * world
        losses.append(total)
        first_loss = first_loss if first_loss is not None else total
        if want:
            dt = (time.time() - t_last) / (50 if step > step0 else 1)
            t_last = time.time()
            rec = {"step": step + 1, "phase": "hazard" if step < hz_steps else "joint", "loss": float(np.mean(losses[-50:])), **terms, "beta": beta,
                   "lr": lr, "grad_norm": float(gn), "grad_parts": gparts, "views": sum(len(r["idx"]) for r in res) // (cfg["rays_label"] + cfg["rays_uniform"]) * world,
                   "s_per_step": dt, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            if rank == 0:
                TR.jl(log_p, rec)
            if (step + 1) % 500 == 0 or step == step0:
                say(json.dumps(rec, default=float))
        if a.smoke in ("none", "resume") and rank == 0:
            if a.smoke == "none" and ((step + 1) % cfg["eval_every"] == 0 or step + 1 == steps):
                eval_sc(ema, step + 1)
            if (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps:
                TR.save_ckpt(ck_last, model, ema, opt, step + 1, rng, {"gen": gen.get_state()})
            if a.smoke == "none" and ((step + 1) in set(cfg["ema_save_steps"]) or step + 1 == steps):
                save_ema(step + 1)
        if ddp and a.smoke in ("none", "resume") and ((step + 1) % cfg["eval_every"] == 0 or (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps):
            dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return
    run["finished"] = time.ctime()
    run["final_loss_mean50"] = float(np.mean(losses[-50:])) if losses else None
    if a.smoke == "overfit":
        run["overfit"] = {"first_loss": first_loss, "last20_loss": float(np.mean(losses[-20:])), "drop": 1.0 - float(np.mean(losses[-20:])) / first_loss, "last_terms": terms}
        say("overfit", run["overfit"])
    if a.smoke == "short":
        run["short"] = {"first50": float(np.mean(losses[:50])), "last50": float(np.mean(losses[-50:]))}
        say("short", run["short"])
    if a.smoke == "gradscale":
        med = {k: float(np.median([x[k] for x in gscale])) for k in gscale[0]}
        run["gradscale"] = {"median_grad_norm_shared": med, "per_step": gscale}
        say("gradscale median", med)
    if a.smoke == "none":
        final = os.path.join(a.out, f"ema_{steps:06d}.pt")
        run["final_checkpoint"] = os.path.relpath(final, R)
        run["final_checkpoint_sha256"] = sha256_file(final)
    json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}.json"), "w"), indent=1, default=str)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
