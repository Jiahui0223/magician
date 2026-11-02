"""EXP6 V5 -- Outdoor-only warm-start training from V4@40k (seen2scene env; torchrun, 4 ranks).

Arms (same data, same per-rank sample streams [seed, rank], same batch 64 views / 8 states per step, same EMA 0.999):
  v5   V5Model (seen2scene/futureview/v5_model.py): encoder + flow U-Net copied strictly from the V4 EMA@40k, new
       V5RayDecoder. Phase Bridge (steps < bridge_steps): encoder / flow frozen; new decoder trained with
         L = L_FM + lambda_surf L_surf + lambda_bridge(s) L_bridge
       Phase Joint: encoder / flow unfrozen (lr_inherited), lambda_bridge decays to 0 by bridge_end_step.
       L_FM     the V2 / V4 rectified-flow velocity MSE (v2_data.fm_sum), global labelled-ray normalisation
       L_surf   BCE of the surface belief b_r(t) on the Train-6 multi-surface labels (build_multisurface_labels),
                balanced over positives / negatives in front of the first surface / negatives behind it, at queries
                per labelled ray: every stored surface (jittered within Delta/4), hard negatives Delta..2 Delta beside
                every surface, log-uniform queries and uniform queries behind the first surface; y(t) = 1 if a surface
                lies within Delta/2; queries outside [t_lo, t_hi] are not supervised
       L_bridge mean ((H_new - H_old) / rms(H_old))^2 (scalar teacher RMS: raw scale matched): H_old = the 64 ray channels of the frozen V4 teacher
                (its own RayEncoder + RayXAttn on the same P1 / camera), training only
  v4c  V4-Continue12k: the V4 model itself from EMA@40k, all parameters trained with lr_inherited, L = L_FM
Checkpoints: ckpt_last (model, EMA, optimizer, step, rng) every ckpt_every; EMA at ema_save_steps; final = EMA@steps
(fixed before training). Monitoring on S-c (held-out trajectories of the Train-6 scenes) only.
torchrun --standalone --nproc_per_node 4 train_v5.py --arm v5|v4c --out DIR [--smoke overfit|short|profile]
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
import v2_frozen as VF  # noqa: E402
import v2_data as VD  # noqa: E402
import t12_train as TR  # noqa: E402
import t22_data as TD  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.futureview import v5_model as V5  # noqa: E402

CFG = os.path.join(R, "experiments", "exp6_v5", "configs")
OUTD = os.path.join(R, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2", "out_views_d100")
LAB = os.path.join(R, "work_dirs", "exp6_v5", "labels", "v1")
V4CK = os.path.join(R, "work_dirs", "exp6_v4", "checkpoints", "final.pt")
V4SHA = "1b31afffbe4da2d94344e0a509b711b30561a34bc4aa521773f1595aafac00d4"


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


class Labels:
    """Multi-surface labels aligned with a ViewData's global rows (same trajectory order)."""

    def __init__(self, items, data):
        cols = {k: [] for k in ("ray_idx", "surf", "n_surf", "t_lo", "t_hi")}
        self.meta = []
        n0 = 0
        for (s, tj) in items:
            z = np.load(os.path.join(LAB, s, f"{tj}.npz"))
            m = json.loads(str(z["meta"]))
            assert m["schema"] == "v5-multisurface-labels-v1"
            rows = data.v["t"][n0:n0 + m["rows"]]
            assert np.array_equal(z["t"], rows), ("label rows do not match the shard rows", s, tj)
            for k in cols:
                cols[k].append(z[k])
            self.meta.append({"scene": s, "traj": tj, "mesh_sha256": m["mesh_sha256"], "rows_sha256": m["rows_sha256"]})
            n0 += m["rows"]
        assert n0 == data.n_views, (n0, data.n_views)
        self.v = {k: np.concatenate(v) for k, v in cols.items()}
        self.k_store = self.v["surf"].shape[-1]


def surface_queries(surf, t_lo, t_hi, gen, Q_rand, delta, Q_back=8):
    """surf [n, Ks] (inf = none), t_lo / t_hi [n] -> queries tq [n, Q] and per query y (1 = a stored surface within
    delta / 2), valid (inside [t_lo, t_hi] of a labelled ray), order (1 = first surface, 2 = second or later surface,
    0 = other) and first [n, 1] (first surface depth, inf if none). Query groups:
      Ks   at every stored surface, jittered within delta / 4                     (positives)
      Ks   at every stored surface + / - (delta .. 2 delta), random side          (hard negatives near surfaces)
      Q_rand  log-uniform in [t_lo, t_hi]
      Q_back  uniform in [t_1 + delta, t_hi] behind the first surface            (free space / later surfaces)
    Every label is the same rule y = min_k |t - t_k| <= delta / 2, so a query that falls on another surface is positive."""
    n, Ks = surf.shape
    dev = surf.device
    fin = torch.isfinite(surf)
    jit = (torch.rand(n, Ks, generator=gen, device=dev) * 2 - 1) * delta / 4
    tpos = torch.where(fin, surf + jit, torch.zeros_like(surf))
    side = torch.where(torch.rand(n, Ks, generator=gen, device=dev) < 0.5, -1.0, 1.0)
    off = delta * (1.0 + torch.rand(n, Ks, generator=gen, device=dev))
    tnear = torch.where(fin, surf + side * off, torch.zeros_like(surf))
    u = torch.rand(n, Q_rand, generator=gen, device=dev)
    lo, hi = torch.log(t_lo.clamp(min=1e-3)), torch.log(t_hi.clamp(min=1e-3))
    trand = torch.exp(lo[:, None] + (hi - lo).clamp(min=0)[:, None] * u)
    first = torch.where(fin[:, :1], surf[:, :1], torch.full_like(surf[:, :1], float("inf")))
    b0 = (first[:, 0] + delta)
    has_back = torch.isfinite(first[:, 0]) & (b0 < t_hi)
    ub = torch.rand(n, Q_back, generator=gen, device=dev)
    tback = torch.where(has_back[:, None], b0[:, None] + (t_hi - b0).clamp(min=0)[:, None] * ub, torch.zeros_like(ub))
    tq = torch.cat([tpos, tnear, trand, tback], 1).clamp(min=1e-4)     # (invalid queries may fall before the camera)
    dist = (tq[..., None] - surf[:, None]).abs()
    dist = torch.where(fin[:, None].expand_as(dist), dist, torch.full_like(dist, float("inf")))
    y = (dist.min(-1).values <= delta / 2).float()
    valid = (tq >= t_lo[:, None]) & (tq <= t_hi[:, None]) & (t_hi > t_lo)[:, None]
    valid[:, :Ks] &= fin
    valid[:, Ks:2 * Ks] &= fin
    valid[:, 2 * Ks + Q_rand:] &= has_back[:, None]
    order = torch.zeros_like(tq, dtype=torch.long)
    order[:, 0] = 1
    order[:, 1:Ks] = 2
    return tq, y, valid, order, first


def surf_loss(logit, y, valid, tq=None, first=None):
    """Balanced BCE over valid queries: mean of (positives, negatives in front of the first surface, negatives behind
    it); without tq / first: 0.5 positives + 0.5 negatives."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(torch.where(valid, logit, torch.zeros_like(logit)), y, reduction="none")
    pos, neg = valid & (y > 0.5), valid & (y < 0.5)
    mean = lambda m: torch.where(m, bce, torch.zeros_like(bce)).sum() / m.sum().clamp(min=1)  # noqa: E731
    if tq is None:
        return 0.5 * (mean(pos) + mean(neg)), {"n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
    behind = neg & (tq > first)
    front = neg & ~behind
    parts = [m for m in (pos, front, behind) if bool(m.any())]
    return sum(mean(m) for m in parts) / max(1, len(parts)), {"n_pos": int(pos.sum()), "n_front": int(front.sum()), "n_behind": int(behind.sum())}


@torch.no_grad()
def surf_metrics(logit, y, valid, order, tq, first):
    p = torch.sigmoid(logit.float()) > 0.5
    pos, neg = valid & (y > 0.5), valid & (y < 0.5)
    behind = neg & (tq > first)
    out = {"recall_first": float((p & pos & (order == 1)).sum() / (pos & (order == 1)).sum().clamp(min=1)),
           "recall_second_plus": float((p & pos & (order == 2)).sum() / (pos & (order == 2)).sum().clamp(min=1)),
           "neg_acc": float((~p & neg).sum() / neg.sum().clamp(min=1)),
           "neg_acc_behind_first": float((~p & behind).sum() / behind.sum().clamp(min=1)),
           "n_second_plus": int((pos & (order == 2)).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["v5", "v4c"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", choices=["none", "overfit", "short", "profile", "gradscale", "resume"], default="none")
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
    cfg_name = a.config or ("train_v5_outdoor.json" if a.arm == "v5" else "train_v4_continue.json")
    cfg = json.load(open(os.path.join(CFG, cfg_name)))
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
    if a.smoke != "none":                       # smoke: the first trajectories whose labels exist (labels are built in parallel)
        pin_t = [tuple(x) for x in cfg["smoke"]["train_items"]]
        pin_h = [tuple(x) for x in cfg["smoke"]["held_items"]]
        assert set(pin_t) <= set(train_items) and set(pin_h) <= set(held_items), "smoke items outside the split"
        train_items, held_items = pin_t, pin_h
    say(f"loading {len(train_items)} train / {len(held_items)} S-c trajectories")
    data = TD.CompactViewData(train_items, root=OUTD, log=say)
    held = TD.CompactViewData(held_items, root=OUTD, log=say)
    labels = Labels(train_items, data) if a.arm == "v5" else None
    hlabels = Labels(held_items, held) if a.arm == "v5" else None
    say(f"train views {data.n_views}, states {len(data.states)}; S-c views {held.n_views}" + (f"; label store {labels.k_store} surfaces" if labels else ""))
    sets = {"train_sub": (data, data.fixed_states("exp5-V2|eval|train_sub|train6_minus_Sc", cfg["eval_sets"]["train_sub"])),
            "sc_sub": (held, held.fixed_states("exp5-V2|eval|sc_sub", cfg["eval_sets"]["sc_sub"], per_scene=True))}
    # ---------------- models
    assert sha256_file(V4CK) == V4SHA, "V4 checkpoint sha256 mismatch"
    ck4 = torch.load(V4CK, map_location="cpu", weights_only=False)
    torch.manual_seed(cfg["seed"])
    teacher = None
    if a.arm == "v5":
        mcfg = json.load(open(os.path.join(CFG, cfg["model"])))
        model = V5.V5Model(mcfg)
        ws = V5.load_v4_warmstart(model, ck4["ema"])
        teacher = M.V2Model(ck4["model_cfg"])
        teacher.load_state_dict(ck4["ema"])
        teacher = teacher.to(dev).eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        teacher.rays_ckpt = False
    else:
        mcfg = ck4["model_cfg"]
        model = M.V2Model(mcfg)
        model.load_state_dict(ck4["ema"])
        ws = {"copied": len(ck4["ema"]), "init": "V4 EMA@40k (all parameters)"}
    model = model.to(dev)
    if ddp:
        for p_ in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(p_.data, 0)
    torch.manual_seed(cfg["seed"] + 1000 * rank)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    if a.arm == "v5":
        model.dec_ckpt = True
        new_p = [p_ for n_, p_ in model.named_parameters() if n_.startswith("dec.")]
        inh_p = [p_ for n_, p_ in model.named_parameters() if not n_.startswith("dec.")]
    else:
        model.rays_ckpt = False
        new_p, inh_p = [], list(model.parameters())
    groups = ([{"params": new_p, "lr": 0.0, "name": "new"}] if new_p else []) + [{"params": inh_p, "lr": 0.0, "name": "inherited"}]
    opt = torch.optim.AdamW(groups, betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng([cfg["seed"], rank])
    gen = torch.Generator(device=dev); gen.manual_seed(cfg["seed"] * 7919 + rank)
    step0 = 0
    ck_last = os.path.join(a.out, "ckpt_last.pt")
    if os.path.exists(ck_last) and a.smoke in ("none", "resume"):
        ck = torch.load(ck_last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        if ddp:                                   # per-rank streams re-seeded at the resume step (the V2 / V4 DDP rule)
            rng = np.random.default_rng([cfg["seed"], rank, step0])
            torch.manual_seed(cfg["seed"] + 1000 * rank + 7 * step0)
            gen.manual_seed(cfg["seed"] * 7919 + rank + 13 * step0)
        else:                                     # single process: exact RNG restore
            rng.bit_generator.state = ck["np_rng"]
            torch.cuda.set_rng_state(ck["torch_cuda_rng"].cpu()); torch.set_rng_state(ck["torch_rng"].cpu())
            gen.set_state(ck["gen"].to(gen.get_state().device) if hasattr(ck["gen"], "to") else ck["gen"])
        say(f"resumed at step {step0}")
    npar = {"total": sum(p_.numel() for p_ in model.parameters()), "new": sum(p_.numel() for p_ in new_p), "inherited": sum(p_.numel() for p_ in inh_p)}
    run = {"arm": a.arm, "config": cfg_name, "config_sha256": sha256_file(os.path.join(CFG, cfg_name)), "train_cfg": cfg, "model_cfg": mcfg,
           "model_cfg_sha256": hashlib.sha256(json.dumps(mcfg, sort_keys=True).encode()).hexdigest(), "init": {"v4_ckpt": V4CK, "v4_sha256": V4SHA, **ws},
           "params": npar, "overrides": a.override, "smoke": a.smoke, "git": git_info(), "host": os.uname()[1], "gpu": torch.cuda.get_device_name(0),
           "world_size": world, "slurm_job": os.environ.get("SLURM_JOB_ID"), "train_trajs": len(train_items), "train_views": data.n_views,
           "data_files_sha256": {**data.files, **held.files}, "label_meta": (labels.meta if labels else None), "started": time.ctime(), "resumed_at": step0,
           "domains": ["outdoor"], "mp3d_training_samples": 0}
    if rank == 0:
        json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}_{int(time.time())}.json"), "w"), indent=1, default=str)
    say("params", npar, "| warm-start", {k: v for k, v in ws.items() if k != "not_used_from_v4"})
    weights = data.state_weights(dcfg["state_weight_t_le"], dcfg["state_weight_factor"])
    steps = {"none": cfg["steps"], "resume": cfg["steps"], "overfit": cfg["smoke"]["overfit_steps"], "short": cfg["smoke"]["short_steps"], "profile": cfg["smoke"]["profile_steps"],
             "gradscale": cfg["smoke"].get("gradscale_warm", 300) + 30}[a.smoke]
    if a.arm == "v5" and a.smoke == "none":
        assert cfg["lambda_surf"] is not None and cfg["lambda_bridge0"] is not None, "lambdas must be frozen in the config before formal training"
    gscale = []
    bridge = cfg.get("bridge_steps", 0)
    eff_world = world if a.smoke == "none" else max(world, int(cfg["smoke"].get("emulate_world", 4)))
    bv, bs = cfg["batch_views"] // eff_world, max(1, cfg["states_per_batch"] // eff_world)
    fixed = data.fixed_states("exp6-V5|smoke|overfit", bv) if a.smoke == "overfit" else None
    ema_steps = set(cfg.get("ema_save_steps", []))
    Delta = cfg.get("surf_delta_vox", 8) * M.VS

    def set_phase(step):
        if a.arm != "v5":
            return
        fz = step < bridge
        for p_ in inh_p:
            p_.requires_grad_(not fz)

    def lrs(step):
        out = {}
        for g in opt.param_groups:
            if g["name"] == "new":
                g["lr"] = cfg["lr_new"] if a.smoke == "overfit" else lr_cos(step, 0, cfg["warmup_new"], steps, cfg["lr_new"])
            else:
                st0 = bridge if a.arm == "v5" else 0
                g["lr"] = cfg["lr_inherited"] if (a.smoke == "overfit" and step >= st0) else lr_cos(step, st0, cfg["warmup_inherited"], steps, cfg["lr_inherited"])
            out[g["name"]] = g["lr"]
        return out

    def lam_bridge(step):
        if a.arm != "v5":
            return 0.0
        end = cfg["bridge_end_step"]
        return cfg["lambda_bridge0"] * max(0.0, 1.0 - step / end)

    def batch_forward(batch, step, want_metrics=False):
        """-> bt (V2 batch dict with cond), surface (logit, y, valid, order, tq, first) or None, bridge terms."""
        conds, keep = [], {"m_unk": [], "lmask": [], "x1": [], "cat": [], "lu_vox": []}
        s_logit, s_y, s_valid, s_order, s_tq, s_first = [], [], [], [], [], []
        br_num = br_den = 0.0
        br_terms = []
        stats = {}
        for ti, t, dlt, rows in batch:
            st = VD.state_tensors(data, ti, t, dlt, rows, dev)
            V_, R_ = st["d"].shape[:2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.encode(st["cin"])
                if a.arm == "v5":
                    oo = st["o"][:, None].expand(V_, R_, 3).reshape(-1, 3)
                    dd, tc = st["d"].reshape(-1, 3), st["tcap"].reshape(-1)
                    chh = st["ch"].transpose(1, 2).reshape(-1, st["ch"].shape[1]).float()
                    H = model.ray_features(h, oo, dd, tc, chh)
                    c = torch.cat([st["ch"].to(H.dtype), H.reshape(V_, R_, -1).transpose(1, 2)], 1).reshape(V_, -1, M.GH, M.GW)
                    c = torch.nn.functional.pad(c, (0, M.GW_PAD - M.GW))
                    valid = torch.zeros(V_, 1, M.GH, M.GW_PAD, device=dev, dtype=c.dtype); valid[..., :M.GW] = 1
                    conds.append(torch.cat([c, valid], 1))
                    # surface belief on the labelled rays (the decoder is per-ray: z of a ray subset = its z in the full pass)
                    ri = torch.as_tensor(labels.v["ray_idx"][rows], device=dev).long()                       # [V, 1024]
                    nsub = cfg["surf_rays_per_view"]
                    pick = torch.argsort(torch.rand(ri.shape, generator=gen, device=dev), 1)[:, :nsub]
                    ri = torch.gather(ri, 1, pick)
                    ok_r = ri >= 0
                    ri = ri.clamp(min=0)
                    flat = (torch.arange(V_, device=dev)[:, None] * R_ + ri).reshape(-1)
                    surf = torch.as_tensor(labels.v["surf"][rows], device=dev).float()
                    surf = torch.gather(surf, 1, pick[..., None].expand(-1, -1, surf.shape[-1])).reshape(-1, surf.shape[-1])
                    tlo = torch.gather(torch.as_tensor(labels.v["t_lo"][rows], device=dev).float(), 1, pick).reshape(-1)
                    thi = torch.gather(torch.as_tensor(labels.v["t_hi"][rows], device=dev).float(), 1, pick).reshape(-1)
                    thi = torch.where(ok_r.reshape(-1), thi, torch.zeros_like(thi))
                    mem = V5.Memory(h[0], h[1], h[3])
                    _, aux = model.dec(mem, oo[flat].float(), dd[flat].float(), tc[flat].float(), chh[flat], return_aux=True)
                    tq, y, vq, order, first = surface_queries(surf, tlo, thi, gen, cfg["surf_rand_queries"], Delta)
                    lg = model.dec.surface_logits(aux["z"], tq, tc[flat].float())
                    s_logit.append(lg.float()); s_y.append(y); s_valid.append(vq); s_order.append(order); s_tq.append(tq); s_first.append(first.expand_as(tq))
                    if lam_bridge(step) > 0 or want_metrics:
                        with torch.no_grad():
                            F0t, F1t = teacher.encoder(st["cin"])
                            Hold = teacher._ray_context_v4(F0t, F1t, st["cin"][:, 2:3], oo, dd, tc).float()
                        Hn = H.float()
                        s_old = Hold.pow(2).mean().sqrt().clamp(min=1e-4)       # scalar RMS of the teacher (raw scale kept)
                        br_terms.append(((Hn - Hold) / s_old).pow(2).mean())
                        stats = {"H_new_mean": float(Hn.mean()), "H_new_std": float(Hn.std()), "H_old_mean": float(Hold.mean()), "H_old_std": float(Hold.std()),
                                 "H_cos": float(torch.nn.functional.cosine_similarity(Hn, Hold, dim=1).mean())}
                else:
                    conds.append(model.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"]))
            for k in keep:
                keep[k].append(st[k])
        bt = {k: torch.cat(v) for k, v in keep.items()}
        bt["cond"] = torch.cat(conds)
        sres = None
        if s_logit:
            sres = tuple(torch.cat(x) for x in (s_logit, s_y, s_valid, s_order, s_tq, s_first))
        bres = (torch.stack(br_terms).mean() if br_terms else None, stats)
        return bt, sres, bres

    @torch.no_grad()
    def eval_surface(model_, dset, lab, key, n_views=256, n_rays=256):
        if lab is None:
            return None
        states = dset.fixed_states(key, n_views, per_scene=True)
        acc = {"recall_first": [], "recall_second_plus": [], "neg_acc": [], "neg_acc_behind_first": [], "loss": [], "n_second_plus": []}
        g = torch.Generator(device=dev); g.manual_seed(12345)
        for ti, t, dlt, rows in states:
            st = VD.state_tensors(dset, ti, t, dlt, rows, dev)
            V_, R_ = st["d"].shape[:2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model_.encode(st["cin"])
                oo = st["o"][:, None].expand(V_, R_, 3).reshape(-1, 3); dd, tc = st["d"].reshape(-1, 3), st["tcap"].reshape(-1)
                chh = st["ch"].transpose(1, 2).reshape(-1, st["ch"].shape[1]).float()
                ri = torch.as_tensor(lab.v["ray_idx"][rows][:, :n_rays], device=dev).long()
                ok_r = ri >= 0
                flat = (torch.arange(V_, device=dev)[:, None] * R_ + ri.clamp(min=0)).reshape(-1)
                surf = torch.as_tensor(lab.v["surf"][rows][:, :n_rays], device=dev).float().reshape(-1, lab.k_store)
                tlo = torch.as_tensor(lab.v["t_lo"][rows][:, :n_rays], device=dev).float().reshape(-1)
                thi = torch.where(ok_r.reshape(-1), torch.as_tensor(lab.v["t_hi"][rows][:, :n_rays], device=dev).float().reshape(-1), torch.zeros(ok_r.numel(), device=dev))
                mem = V5.Memory(h[0], h[1], h[3])
                _, aux = model_.dec(mem, oo[flat].float(), dd[flat].float(), tc[flat].float(), chh[flat], return_aux=True)
                tq, y, vq, order, first = surface_queries(surf, tlo, thi, g, cfg["surf_rand_queries"], Delta)
                lg = model_.dec.surface_logits(aux["z"], tq, tc[flat].float()).float()
            l_, _ = surf_loss(lg, y, vq, tq, first.expand_as(tq))
            m_ = surf_metrics(lg, y, vq, order, tq, first.expand_as(tq))
            acc["loss"].append(float(l_))
            for k in m_:
                acc[k].append(m_[k])
        return {k: (float(np.sum(v)) if k == "n_second_plus" else float(np.mean(v))) for k, v in acc.items()}

    def save_ema(step):
        torch.save({"ema": ema.state_dict(), "step": step, "model_cfg": mcfg, "arm": a.arm, "lineage": f"V4 Outdoor EMA@40k -> {a.arm} +{step}"},
                   os.path.join(a.out, f"ema_{step:06d}.pt"))

    def evaluate(step):
        ev = {"step": step}
        for nm, (dd_, ss) in sets.items():
            ev[f"fm_{nm}"] = TR.eval_fm(ema, dd_, ss, nm, dev, cfg["eval_tau"])
        ev["samp_sc_sub"] = TR.eval_samples(ema, sets["sc_sub"][0], sets["sc_sub"][1], "sc_sub", dev)
        if a.arm == "v5":
            ev["surf_sc"] = eval_surface(ema, held, hlabels, "exp6-V5|eval|surf_sc")
        TR.jl(ev_p, ev)
        say("eval", json.dumps(ev, default=float))

    if a.smoke == "none" and rank == 0 and step0 == 0:
        save_ema(0)
        evaluate(0)
    if ddp and a.smoke == "none":
        dist.barrier()
    model.train()
    if a.arm == "v4c":
        model.rays_ckpt = False
    t_last, losses, first_loss, prof = time.time(), [], None, []
    for step in range(step0, steps):
        set_phase(step)
        lr = lrs(step)
        lb = lam_bridge(step)
        t0 = time.time()
        batch = fixed if fixed is not None else data.sample_batch(rng, bv, bs, dcfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], weights)
        want = (step + 1) % 50 == 0 or step == step0
        bt, sres, (lbr, hstats) = batch_forward(batch, step, want)
        V = bt["cond"].shape[0]
        eps = torch.randn(V, 2, M.GH, M.GW_PAD, device=dev)
        tau = torch.rand(V, device=dev)
        esum, nray = VD.fm_sum(model, bt, eps, tau)
        tot = torch.stack([esum.detach(), nray]).double()
        if ddp:
            dist.all_reduce(tot)
        nray_g = tot[1].clamp(min=1.0)
        loss = esum / (2.0 * nray_g.to(esum.dtype))
        terms = {"fm": float(tot[0] / (2 * nray_g))}
        if sres is not None:
            ls, cnt = surf_loss(sres[0], sres[1], sres[2], sres[4], sres[5])
            loss = loss + cfg["lambda_surf"] * ls / world
            terms["surf"] = float(ls)
            if want:
                terms.update(surf_metrics(*sres[:4], sres[4], sres[5]))
        if lbr is not None and lb > 0:
            loss = loss + lb * lbr / world
            terms["bridge"] = float(lbr)
        if not torch.isfinite(loss):
            raise RuntimeError(f"STOP: non-finite loss at step {step}: {terms}")
        if a.smoke == "gradscale" and a.arm == "v5" and step >= steps - 30:
            fm_l = esum / (2.0 * nray_g.to(esum.dtype))
            parts = {"fm": fm_l, "surf": ls if sres is not None else None, "bridge": lbr}
            gs = {}
            for nme, term in parts.items():
                if term is None:
                    continue
                gg = torch.autograd.grad(term, new_p, retain_graph=True, allow_unused=True)
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
        gnorm_parts = None
        if want:
            gnorm_parts = {}
            for nme, pre in (("dec", "dec."), ("encoder", "encoder."), ("unet", "unet."), ("rays", "rays."), ("ray_xattn", "ray_xattn.")):
                gs = [p_.grad.float().norm() ** 2 for n_, p_ in model.named_parameters() if n_.startswith(pre) and p_.grad is not None]
                if gs:
                    gnorm_parts[nme] = float(torch.stack(gs).sum().sqrt())
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
            rec = {"step": step + 1, "phase": ("bridge" if step < bridge else "joint") if a.arm == "v5" else "continue", "loss": float(np.mean(losses[-50:])),
                   **terms, "lambda_bridge": lb, "lr": lr, "grad_norm": float(gn), "grad_parts": gnorm_parts, "H": hstats, "views": V * world, "rays": float(nray_g),
                   "s_per_step": dt, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            if rank == 0:
                TR.jl(log_p, rec)
            if (step + 1) % 500 == 0 or step == step0:
                say(json.dumps(rec, default=float))
        if a.smoke in ("none", "resume") and rank == 0:
            if a.smoke == "none" and ((step + 1) % cfg["eval_every"] == 0 or step + 1 == steps):
                evaluate(step + 1)
            if (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps:
                TR.save_ckpt(ck_last, model, ema, opt, step + 1, rng, {"gen": gen.get_state()})
            if a.smoke == "none" and ((step + 1) in ema_steps or step + 1 == steps):
                save_ema(step + 1)
        if ddp and a.smoke in ("none", "resume") and ((step + 1) % cfg["eval_every"] == 0 or (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps):
            dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return
    run["finished"] = time.ctime()
    run["final_loss_mean50"] = float(np.mean(losses[-50:])) if losses else None
    if a.smoke == "overfit":
        run["overfit"] = {"first_loss": first_loss, "last20_loss": float(np.mean(losses[-20:])), "drop": 1.0 - float(np.mean(losses[-20:])) / first_loss}
        if a.arm == "v5":
            run["overfit"]["surf_last"] = terms
        say("overfit", run["overfit"])
    if a.smoke == "gradscale":
        g_ = {k: float(np.median([x[k] for x in gscale if k in x])) for k in gscale[0]}
        run["gradscale"] = {"median_grad_norm_new_modules": g_, "per_step": gscale}
        say("gradscale median", g_)
    if a.smoke == "profile":
        p_ = np.asarray(prof[5:])
        run["profile"] = {"s_per_step_mean": float(p_.mean()), "s_per_step_p90": float(np.quantile(p_, 0.9)), "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
        say("profile", run["profile"])
    if a.smoke == "short":
        run["short"] = {"first50": float(np.mean(losses[:50])), "last50": float(np.mean(losses[-50:]))}
        say("short", run["short"])
    if a.smoke == "none":
        final = os.path.join(a.out, f"ema_{steps:06d}.pt")
        run["final_checkpoint"] = os.path.relpath(final, R)
        run["final_checkpoint_sha256"] = sha256_file(final)
    json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}.json"), "w"), indent=1, default=str)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
