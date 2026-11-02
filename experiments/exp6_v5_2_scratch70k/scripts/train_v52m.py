"""EXP6 V5.2-M Scratch70k -- trainer of the Idea-2 Geometry Belief Scorer (seen2scene env; torchrun, 4 or 8 ranks).

Model  seen2scene/futureview/v52m_model.py V52MModel, ALL parameters randomly initialised (no V4 / V5 / V5.2 weight is
       read). Loss L = L_surface + beta * L_firsthit only.
         L_surface   per depth interval i of a labelled ray: y_i = 1 iff a stored Train-6 mesh surface lies in
                     [t_i^- - Delta/2, t_i^+ + Delta/2] (Delta = 8 VS); supervised only for intervals inside the label's
                     trusted range [t_lo, t_hi]; BCE balanced over positives / negatives in front of the first surface /
                     negatives behind it (the V5 semantics on the interval grid)
         L_firsthit  -log p_first[i*] (interval of the T1.1 first hit t* < tcap) or -log p_nohit (t* beyond tcap / none);
                     masked: t* before the near clip, and LABEL CONFLICTS (t* inside an interval the D100 forces FREE;
                     counted as the FREE-violation rate, never trained on)
Rays   per view: 256 rays of the label cache (surface + first hit) and 256 first-hit rays (128 of class UNKNOWN when
       available + uniform rest); the whole 64 x 114 ray grid of every view goes through the ray query and the 2D
       spatial U-Net (context), only the sampled rays through the depth decoder.
Batch  global 64 views / 8 states per optimizer step, independent of the world size (per rank = global / world, the
       OOM fallback splits it into micro-batches with gradient accumulation; seen samples are logged).
Ckpt   last.pt every ckpt_every steps (atomic tmp -> rename; model, EMA, optimizer, step, world size, config, git) +
       per-rank RNG files; EMA tags step_XXXXXX.pt; best_train_holdout.pt by the pre-registered Train-6 S-c score
       (UNKNOWN first-hit NLL + 2 x UNKNOWN high-confidence false-no-hit rate). Resume is exact for an unchanged world
       size (per-rank streams restored) and re-seeded deterministically from the step for a changed one.
Exits  0 done | 85 checkpointed on a stop request / time limit | 86 OOM fallback exhausted | 87 non-finite loss (dump)
torchrun ... train_v52m.py --out RUN_DIR [--smoke overfit|short|gradscale|resume] [--override k=v ...]
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
import train_v5 as TV  # noqa: E402  (Labels)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.futureview import v52_model as V52  # noqa: E402
from seen2scene.futureview import v52m_model as VM  # noqa: E402

CFG = os.path.join(R, "experiments", "exp6_v5_2_scratch70k", "configs")
OUTD = os.path.join(R, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2", "out_views_d100")
EXIT_CKPT, EXIT_OOM, EXIT_NAN = 85, 86, 87


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=R).decode().strip()
        diff = subprocess.check_output(["git", "diff", "HEAD", "--", "seen2scene", "experiments/exp6_v5_2_scratch70k"], cwd=R)
        return {"sha": sha, "code_dirty": bool(diff), "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None}
    except Exception as e:                       # noqa: BLE001
        return {"error": str(e)}


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * (cfg["lr_min_frac"] + (1 - cfg["lr_min_frac"]) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p))))


class ViewDataM(VD.ViewData):
    """v2_data.ViewData + tstar [views, 7296] float16 + fine grid dims per trajectory; coarse grids zlib-compressed."""

    def __init__(self, items, root, log=print):
        super().__init__(items, root=root, keys=VD.VIEW_KEYS + ("tstar",), log=log)
        self.v["tstar"] = self.v["tstar"].astype(np.float16)
        self.coarse = [TD.ZCoarse(c.numpy()) for c in self.coarse]
        self.dims = [np.asarray(np.load(os.path.join(root, s, f"{tj}.npz"))["dims"], np.int64) for s, tj in items]


def atomic_save(obj, path):
    tmp = path + f".tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def surface_targets(edges, surf, t_lo, t_hi, delta):
    """edges [N, B+1], surf [N, K] (inf = none), t_lo / t_hi [N] -> y, valid, behind_first, order (1 first-surface
    interval, 2 second+ surface interval, 0 none) [N, B]."""
    a, b = edges[:, :-1], edges[:, 1:]
    fin = torch.isfinite(surf)
    s = torch.where(fin, surf, torch.full_like(surf, 1e9))
    lo, hi = a[..., None] - delta / 2, b[..., None] + delta / 2
    inb = (s[:, None] >= lo) & (s[:, None] <= hi) & fin[:, None]
    y = inb.any(-1).float()
    valid = (a >= t_lo[:, None]) & (b <= t_hi[:, None]) & (t_hi > t_lo)[:, None]
    first = torch.where(fin[:, 0], surf[:, 0], torch.full_like(surf[:, 0], float("inf")))
    behind = a > first[:, None] + delta / 2
    order = torch.zeros_like(y, dtype=torch.long)
    order[inb[..., 0]] = 1
    order[(inb[..., 1:].any(-1)) & (order == 0)] = 2
    return y, valid, behind, order


def surface_loss(logit, y, valid, behind):
    bce = F.binary_cross_entropy_with_logits(torch.where(valid, logit, torch.zeros_like(logit)), y, reduction="none")
    pos, neg = valid & (y > 0.5), valid & (y < 0.5)
    groups = [pos, neg & ~behind, neg & behind]
    terms = [torch.where(m, bce, torch.zeros_like(bce)).sum() / m.sum().clamp(min=1) for m in groups if bool(m.any())]
    if not terms:
        return logit.sum() * 0.0
    return sum(terms) / len(terms)


def firsthit_targets(out, tstar):
    """-> idx [N] (interval of t*, B = no hit), valid [N], conflict [N] (t* in an interval forced FREE)."""
    e = out["edges"]
    B = e.shape[1] - 1
    t = tstar.float()
    valid = ~(t < e[:, 0])
    hit = torch.isfinite(t) & (t < e[:, -1])
    idx = torch.searchsorted(e.contiguous(), t.clamp(max=1e9)[:, None].contiguous())[:, 0] - 1
    idx = torch.where(hit, idx.clamp(0, B - 1), torch.full_like(idx, B))
    ff = torch.gather(torch.cat([out["free_frac"], torch.zeros_like(out["free_frac"][:, :1])], 1), 1, idx[:, None])[:, 0]
    conflict = hit & (ff >= 0.999)
    return idx, valid & ~conflict, conflict


def firsthit_nll(out, idx, valid):
    B = out["log_pf"].shape[1]
    lp = torch.where(idx == B, -out["a"].sum(-1), torch.gather(out["log_pf"], 1, idx.clamp(max=B - 1)[:, None])[:, 0])
    return (-lp * valid).sum() / valid.sum().clamp(min=1), -lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="train_v52m.json")
    ap.add_argument("--smoke", choices=["none", "overfit", "short", "gradscale", "resume"], default="none")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--deadline", type=float, default=0.0, help="unix time: save and exit 85 before it")
    a = ap.parse_args()
    import torch.distributed as dist
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    cfg = json.load(open(os.path.join(CFG, a.config)))
    for kv in a.override:
        k, v = kv.split("=")
        tgt = cfg
        if "." in k:
            k0, k = k.split(".")
            tgt = cfg[k0]
        tgt[k] = float(v) if tgt.get(k) is None else type(tgt[k])(v)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    log_p, ev_p = os.path.join(a.out, "log.jsonl"), os.path.join(a.out, "holdout.jsonl")
    say = (lambda *x: print(time.strftime("%F %T"), *x, flush=True)) if rank == 0 else (lambda *x: None)  # noqa: E731
    dcfg = VF.load_cfg("data_v1.json")
    train_items, held_items = VD.load_split("train6_minus_Sc")
    if a.smoke != "none":
        train_items = [tuple(x) for x in cfg["smoke"]["train_items"]]
        held_items = [tuple(x) for x in cfg["smoke"]["held_items"]]
    say(f"loading {len(train_items)} train / {len(held_items)} S-c trajectories (world {world})")
    data = ViewDataM(train_items, OUTD, log=say)
    held = ViewDataM(held_items, OUTD, log=say)
    labels, hlabels = TV.Labels(train_items, data), TV.Labels(held_items, held)
    mcfg = json.load(open(os.path.join(CFG, cfg["model"])))
    torch.manual_seed(cfg["seed"])                                             # identical init on every rank (scratch)
    model = VM.V52MModel(mcfg).to(dev)
    model.spatial.use_ckpt = cfg["ckpt_spatial"]
    model.depth.use_ckpt = cfg["ckpt_depth"]
    if ddp:
        for p_ in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(p_.data, 0)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0, betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    ckdir = os.path.join(a.out, "checkpoints")
    os.makedirs(ckdir, exist_ok=True)
    last_p = os.path.join(ckdir, "last.pt")
    rng_p = lambda r: os.path.join(ckdir, f"last_rng_w{world}_r{r}.pt")  # noqa: E731
    step0, accum, oom_retries, best = 0, 1, 0, None
    rng = np.random.default_rng([cfg["seed"], rank])
    gen = torch.Generator(device=dev); gen.manual_seed(cfg["seed"] * 7919 + rank)
    torch.manual_seed(cfg["seed"] + 1000 * rank)
    if os.path.exists(last_p) and a.smoke in ("none", "resume"):
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step0, accum, oom_retries, best = ck["step"], ck.get("accum", 1), ck.get("oom_retries", 0), ck.get("best")
        same_world = ck.get("world_size") == world and os.path.exists(rng_p(rank))
        if same_world:
            rs = torch.load(rng_p(rank), map_location="cpu", weights_only=False)
            assert rs["step"] == step0, ("rng file / last.pt step mismatch", rs["step"], step0)
            rng.bit_generator.state = rs["np"]; torch.set_rng_state(rs["torch"]); torch.cuda.set_rng_state(rs["cuda"]); gen.set_state(rs["gen"])
        else:
            rng = np.random.default_rng([cfg["seed"], rank, step0, world])
            torch.manual_seed(cfg["seed"] + 1000 * rank + 7 * step0)
            gen.manual_seed(cfg["seed"] * 7919 + rank + 13 * step0 + 101 * world)
        say(f"resumed at step {step0} (world {ck.get('world_size')} -> {world}, {'exact per-rank RNG' if same_world else 're-seeded streams'}, accum {accum})")
    npar = VM.param_table(model)
    run = {"config": a.config, "config_sha256": sha256_file(os.path.join(CFG, a.config)), "train_cfg": cfg, "model_cfg": mcfg,
           "model_cfg_sha256": hashlib.sha256(json.dumps(mcfg, sort_keys=True).encode()).hexdigest(), "init": "scratch (seed %d), no checkpoint read" % cfg["seed"],
           "params": npar, "overrides": a.override, "smoke": a.smoke, "git": git_info(), "host": os.uname()[1], "gpu": torch.cuda.get_device_name(0),
           "world_size": world, "slurm_job": os.environ.get("SLURM_JOB_ID"), "train_trajs": len(train_items), "train_views": data.n_views,
           "data_files_sha256": {**data.files, **held.files}, "label_meta": labels.meta, "started": time.ctime(), "resumed_at": step0,
           "domains": ["outdoor"], "mp3d_training_samples": 0, "losses": ["L_surface", "beta * L_firsthit"]}
    if rank == 0:
        json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}_{int(time.time())}.json"), "w"), indent=1, default=str)
    say("params", npar)
    weights = data.state_weights(dcfg["state_weight_t_le"], dcfg["state_weight_factor"])
    steps = {"none": cfg["steps"], "resume": cfg["steps"], "overfit": cfg["smoke"]["overfit_steps"], "short": cfg["smoke"]["short_steps"],
             "gradscale": cfg["smoke"]["gradscale_steps"]}[a.smoke]
    eff_world = world if a.smoke in ("none", "resume") else max(world, int(cfg["smoke"].get("emulate_world", 8)))
    bv = cfg["batch_views"] // eff_world
    bs = max(1, cfg["states_per_batch"] // eff_world)
    fixed = data.fixed_states("exp6-V52M|smoke|overfit", bv) if a.smoke == "overfit" else None
    Delta = cfg["surf_delta_vox"] * M.VS
    beta = cfg["beta"]
    tag_steps = set(cfg["tag_steps"])
    gscale = []

    def state_forward(model_, dset, lab, ti, t, dlt, rows, n_lab, n_fh, g, train=True):
        st = VD.state_tensors(dset, ti, t, dlt, rows, dev)
        V_, R_ = st["d"].shape[:2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model_.encode(st["cin"])
            mem = model_.memory(h)
            oo = st["o"][:, None].expand(V_, R_, 3).reshape(-1, 3)
            chT = st["ch"].transpose(1, 2).reshape(-1, st["ch"].shape[1]).float()
            z = model_.ray_latents(mem, oo, st["d"].reshape(-1, 3), st["tcap"].reshape(-1), chT, use_ckpt=train).reshape(V_, R_, -1)
            c = model_.context(z, st["ch"])                                                  # [V, R, d_ctx]
            ri = torch.as_tensor(lab.v["ray_idx"][rows], device=dev).long()
            pick = torch.argsort(torch.rand(ri.shape, generator=g, device=dev), 1)[:, :n_lab]
            ri = torch.gather(ri, 1, pick)
            ok_r = (ri >= 0).reshape(-1)
            ri = ri.clamp(min=0)
            cls = torch.as_tensor(dset.v["cls"][rows], device=dev)
            unk = (cls == 0).float()
            n_u = n_fh // 2
            pu = torch.multinomial(unk + 1e-6, n_u, replacement=True, generator=g)           # UNKNOWN-class rays (uniform among them)
            pa = torch.randint(0, R_, (V_, n_fh - n_u), generator=g, device=dev)              # uniform rays
            rr = torch.cat([ri, pu, pa], 1)
            vv = torch.arange(V_, device=dev)[:, None].expand_as(rr)
            fv, fr = vv.reshape(-1), rr.reshape(-1)
            stats = {"tk": torch.as_tensor(dset.v["tk"][rows], device=dev)[fv, fr], "dlo": torch.as_tensor(dset.v["dlo"][rows], device=dev)[fv, fr],
                     "dhi": torch.as_tensor(dset.v["dhi"][rows], device=dev)[fv, fr], "cls": cls[fv, fr]}
            o_s, d_s, tc_s = st["o"][fv], st["d"][fv, fr], st["tcap"][fv, fr]
            stats["tmax"] = VM.tmax_from_dims(o_s.float(), d_s.float(), tc_s.float(), dset.dims[ti])
            out = model_.decode(mem, o_s.float(), d_s.float(), tc_s.float(), z[fv, fr], c[fv, fr], stats)
        is_lab = torch.zeros_like(rr, dtype=torch.bool); is_lab[:, :n_lab] = True
        is_lab = is_lab.reshape(-1)
        surf = torch.as_tensor(lab.v["surf"][rows], device=dev).float()
        surf = torch.gather(surf, 1, pick[..., None].expand(-1, -1, surf.shape[-1])).reshape(-1, surf.shape[-1])
        tlo = torch.gather(torch.as_tensor(lab.v["t_lo"][rows], device=dev).float(), 1, pick).reshape(-1)
        thi = torch.gather(torch.as_tensor(lab.v["t_hi"][rows], device=dev).float(), 1, pick).reshape(-1)
        thi = torch.where(ok_r, thi, torch.zeros_like(thi))
        y, sv, behind, order = surface_targets(out["edges"][is_lab], surf, tlo, thi, Delta)
        ts = torch.as_tensor(dset.v["tstar"][rows], device=dev)[fv, fr].float()
        ts = torch.where(torch.isfinite(ts), ts, torch.full_like(ts, float("inf")))
        idx, fvalid, conflict = firsthit_targets(out, ts)
        kind = torch.zeros_like(rr); kind[:, n_lab:n_lab + n_u] = 1; kind[:, n_lab + n_u:] = 2
        return {"out": out, "is_lab": is_lab, "y": y, "sv": sv, "behind": behind, "order": order, "idx": idx, "fvalid": fvalid, "conflict": conflict,
                "ts": ts, "cls": stats["cls"], "kind": kind.reshape(-1), "surf": surf, "tk_abs": stats["tk"].float() * tc_s.float()}

    def losses(res_list):
        lg = torch.cat([r["out"]["logit_b"][r["is_lab"]] for r in res_list])
        y = torch.cat([r["y"] for r in res_list]); sv = torch.cat([r["sv"] for r in res_list]); bh = torch.cat([r["behind"] for r in res_list])
        ls = surface_loss(lg, y, sv, bh)
        o = {k: torch.cat([r["out"][k] for r in res_list]) for k in ("a", "log_pf")}
        idx = torch.cat([r["idx"] for r in res_list]); fv = torch.cat([r["fvalid"] for r in res_list])
        lfh, _ = firsthit_nll(o, idx, fv)
        return ls, lfh

    @torch.no_grad()
    def stats_of(res_list):
        """Train-side / holdout statistics (natural = uniform rays only for calibration-type numbers)."""
        acc = {}
        P = torch.cat([torch.cat([r["out"]["p_first"], r["out"]["p_nohit"][:, None]], 1) for r in res_list])
        idx = torch.cat([r["idx"] for r in res_list]); fv = torch.cat([r["fvalid"] for r in res_list]); cf = torch.cat([r["conflict"] for r in res_list])
        kind = torch.cat([r["kind"] for r in res_list]); cls = torch.cat([r["cls"] for r in res_list])
        o = {k: torch.cat([r["out"][k] for r in res_list]) for k in ("a", "log_pf", "t_mid", "p_first", "p_nohit")}
        ts = torch.cat([r["ts"] for r in res_list])
        _, nll = firsthit_nll(o, idx, fv)
        B = P.shape[1] - 1
        hit = idx < B
        top = P.argmax(1) == idx
        hcfn = (o["p_nohit"] > 0.95) & hit
        mode_t = torch.gather(o["t_mid"], 1, o["p_first"].argmax(1, keepdim=True))[:, 0]
        for nm, m in (("nat", (kind == 2) & fv), ("nat_unknown", (kind == 2) & fv & (cls == 0)), ("unknown", fv & (cls == 0)), ("known", fv & (cls != 0))):
            if not bool(m.any()):
                continue
            acc[f"nll_{nm}"] = float(nll[m].mean())
            acc[f"top_{nm}"] = float(top[m].float().mean())
            acc[f"hit_rate_{nm}"] = float(hit[m].float().mean())
            acc[f"hc_false_nohit_{nm}"] = float(hcfn[m].float().sum() / (hit & m).float().sum().clamp(min=1))
            dm = m & hit & (o["p_nohit"] < 0.5)
            acc[f"mode_err_vox_{nm}"] = float(((mode_t - ts).abs() / M.VS)[dm].mean()) if bool(dm.any()) else float("nan")
        mn = (kind == 2) & fv
        if bool(mn.any()):
            pn, yn = o["p_nohit"][mn], (~hit[mn]).float()
            bins = torch.clamp((pn * 10).long(), max=9)
            acc["nohit_ece_nat"] = float(sum((bins == k).float().sum() * (pn[bins == k].mean() - yn[bins == k].mean()).abs() for k in range(10) if bool((bins == k).any())) / len(pn))
            acc["pred_nohit_nat"], acc["true_nohit_nat"] = float(pn.mean()), float(yn.mean())
        # known-physics violation of the model: on KNOWN-class rays (trusted D100 TSDF crossing at t_known) the mass placed beyond
        # t_known + 8 VS or on no-hit (pass-through of an observed surface). Mass inside D100 FREE intervals is 0 by construction.
        tkk = torch.cat([r["tk_abs"] for r in res_list])
        ed = torch.cat([r["out"]["edges"] for r in res_list])
        mk = (cls == 1) & torch.isfinite(tkk) & (tkk < ed[:, -1])
        if bool(mk.any()):
            beyond = ed[:, :-1] > (tkk + 8 * M.VS)[:, None]
            acc["known_passthrough"] = float(((o["p_first"] * beyond).sum(1) + o["p_nohit"])[mk].mean())
        # sampling strata (first-hit rays; "nat" = the uniform rays = natural proportions)
        for nm, m in (("all", fv), ("nat", (kind == 2) & fv)):
            n_ = m.float().sum().clamp(min=1)
            acc[f"share_{nm}_known_free"] = float(((cls == 2) & m).float().sum() / n_)
            acc[f"share_{nm}_known_band"] = float(((cls == 1) & m).float().sum() / n_)
            acc[f"share_{nm}_unknown_hit"] = float(((cls == 0) & hit & m).float().sum() / n_)
            acc[f"share_{nm}_unknown_nohit"] = float(((cls == 0) & ~hit & m).float().sum() / n_)
            acc[f"share_{nm}_near_hit"] = float((hit & m & (ts < 64 * M.VS)).float().sum() / n_)
            acc[f"share_{nm}_far_hit"] = float((hit & m & (ts >= 256 * M.VS)).float().sum() / n_)
        acc["share_label_second_plus"] = float((torch.isfinite(torch.cat([r["surf"] for r in res_list])).sum(1) >= 2).float().mean())
        acc["free_violation_rate"] = float(cf.float().sum() / (torch.cat([r["idx"] for r in res_list]) < B).float().sum().clamp(min=1))
        lg = torch.cat([r["out"]["logit_b"][r["is_lab"]] for r in res_list])
        y = torch.cat([r["y"] for r in res_list]); sv = torch.cat([r["sv"] for r in res_list]); od = torch.cat([r["order"] for r in res_list])
        bh = torch.cat([r["behind"] for r in res_list])
        pr = torch.sigmoid(lg) > 0.5
        for nm, m in (("surf_recall_first", sv & (od == 1)), ("surf_recall_second_plus", sv & (od == 2))):
            acc[nm] = float(pr[m].float().mean()) if bool(m.any()) else float("nan")
        neg = sv & (y < 0.5)
        acc["surf_neg_acc_behind_first"] = float((~pr[neg & bh]).float().mean()) if bool((neg & bh).any()) else float("nan")
        # far occlusion: second+ surface >= 128 VS behind the first stored surface
        fars = []
        for r in res_list:
            s = r["surf"]
            fin = torch.isfinite(s)
            gap = torch.where(fin[:, 1:], s[:, 1:] - s[:, :1], torch.full_like(s[:, 1:], -1.0))
            far_any = (gap >= 128 * M.VS)
            if bool(far_any.any()):
                e = r["out"]["edges"][r["is_lab"]]
                idxs = torch.searchsorted(e.contiguous(), torch.where(far_any, s[:, 1:], torch.zeros_like(gap)).contiguous()) - 1
                idxs = idxs.clamp(0, e.shape[1] - 2)
                pb = torch.sigmoid(torch.gather(r["out"]["logit_b"][r["is_lab"]], 1, idxs)) > 0.5
                fars.append(pb[far_any].float())
        acc["surf_recall_far128"] = float(torch.cat(fars).mean()) if fars else float("nan")
        return acc

    @torch.no_grad()
    def holdout(model_, step):
        states = held.fixed_states("exp6-V52M|holdout|sc", cfg["holdout_views"], per_scene=True)
        g = torch.Generator(device=dev); g.manual_seed(12345)
        res, ls_, lf_ = [], [], []
        for ti, t, dlt, rows in states:
            r = state_forward(model_, held, hlabels, ti, t, dlt, rows, 256, 256, g, train=False)
            ls, lf = losses([r])
            ls_.append(float(ls)); lf_.append(float(lf))
            res.append({k: (v.detach() if torch.is_tensor(v) else {kk: vv.detach() for kk, vv in v.items()}) for k, v in r.items()})
        stt = stats_of(res)
        stt.update({"step": step, "loss_surface": float(np.mean(ls_)), "loss_firsthit": float(np.mean(lf_))})
        stt["select_score"] = stt.get("nll_unknown", float("inf")) + 2.0 * stt.get("hc_false_nohit_unknown", 1.0) + 2.0 * stt.get("known_passthrough", 1.0)
        TR.jl(ev_p, stt)
        say("holdout", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stt.items()}))
        return stt

    def save_last(step, why):
        rs = {"step": step, "np": rng.bit_generator.state, "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "gen": gen.get_state()}
        atomic_save(rs, rng_p(rank))
        if ddp:
            dist.barrier()
        if rank == 0:
            atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step, "world_size": world, "accum": accum,
                         "oom_retries": oom_retries, "best": best, "config": cfg, "model_cfg": mcfg, "git": run["git"], "params": npar, "why": why,
                         "time": time.ctime()}, last_p)
            json.dump({"step": step, "world_size": world, "why": why, "time": time.ctime(), "sha_last": None}, open(os.path.join(ckdir, "last_manifest.json"), "w"))
        if ddp:
            dist.barrier()

    def save_tag(step, name=None):
        p = os.path.join(ckdir, name or f"step_{step:06d}.pt")
        atomic_save({"ema": ema.state_dict(), "step": step, "model_cfg": mcfg, "arch": "v52m", "lineage": f"V5.2-M scratch (seed {cfg['seed']}) EMA@{step}"}, p)
        return p

    if a.smoke == "none" and step0 == 0 and rank == 0:
        holdout(ema, 0)
    if ddp:
        dist.barrier()
    model.train()
    t_last, losses_hist, first_loss, seen = time.time(), [], None, step0 * cfg["batch_views"]
    data_s = 0.0
    step = step0
    while step < steps:
        lr = cfg["lr"] if a.smoke == "overfit" else lr_at(step, cfg)
        for gp in opt.param_groups:
            gp["lr"] = lr
        want = (step + 1) % 50 == 0 or step == step0
        td = time.time()
        batch = fixed if fixed is not None else data.sample_batch(rng, bv, bs, dcfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], weights)
        # micro-batches (OOM fallback: accum > 1 splits the states / views of this rank)
        micro = []
        for ti, t, dlt, rows in batch:
            k = max(1, accum // max(1, len(batch)))
            for part in np.array_split(np.asarray(rows), min(k, len(rows))):
                micro.append((ti, t, dlt, part))
        data_s += time.time() - td
        opt.zero_grad(set_to_none=True)
        err, terms, res_keep = 0.0, {}, []
        try:
            nmb = len(micro)
            for mb in micro:
                r = state_forward(model, data, labels, *mb, cfg["rays_label"], cfg["rays_fh"], gen)
                ls, lfh = losses([r])
                loss = (ls + beta * lfh) / nmb / world
                if not torch.isfinite(loss):
                    err = 2.0
                    terms = {"surf": float(ls), "firsthit": float(lfh)}
                    break
                if a.smoke == "gradscale" and step >= steps - 20:
                    gs = {}
                    for nme, term in (("surf", ls), ("firsthit", lfh)):
                        gg = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                        gs[nme] = float(torch.sqrt(sum((g_.float() ** 2).sum() for g_ in gg if g_ is not None)))
                    gscale.append(gs)
                loss.backward()
                terms["surf"] = terms.get("surf", 0.0) + float(ls) / nmb
                terms["firsthit"] = terms.get("firsthit", 0.0) + float(lfh) / nmb
                if want:
                    res_keep.append({k: (v.detach() if torch.is_tensor(v) else {kk: vv.detach() for kk, vv in v.items()}) for k, v in r.items()})
                del r
        except torch.cuda.OutOfMemoryError:
            err = 1.0
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        flag = torch.tensor([err], device=dev)
        if ddp:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if float(flag) >= 2.0:
            if rank == 0:
                json.dump({"step": step, "terms": terms, "batch": [(int(x[0]), int(x[1]), int(x[2]), np.asarray(x[3]).tolist()) for x in batch]},
                          open(os.path.join(a.out, f"nan_dump_step{step}.json"), "w"), default=float)
            say(f"STOP: non-finite loss at step {step} {terms}")
            sys.exit(EXIT_NAN)
        if float(flag) >= 1.0:
            oom_retries += 1
            if oom_retries > cfg["oom_max_retries"]:
                say(f"STOP: CUDA OOM at step {step}, fallback exhausted ({oom_retries - 1} retries, accum {accum})")
                save_last(step, "oom_exhausted")
                sys.exit(EXIT_OOM)
            accum *= 2
            model.ray_chunk = max(2048, model.ray_chunk // 2); model.dec_chunk = max(512, model.dec_chunk // 2)
            ema.ray_chunk, ema.dec_chunk = model.ray_chunk, model.dec_chunk
            say(f"CUDA OOM at step {step}: retry {oom_retries}/{cfg['oom_max_retries']} with per-rank micro-batches x{accum} (global batch kept), ray chunk {model.ray_chunk}, decoder chunk {model.dec_chunk}")
            continue
        if ddp:
            for p_ in params:
                if p_.grad is None:
                    p_.grad = torch.zeros_like(p_)
            flat = torch._utils._flatten_dense_tensors([p_.grad for p_ in params])
            dist.all_reduce(flat)
            for g_, s_ in zip(params, torch._utils._unflatten_dense_tensors(flat, [p_.grad for p_ in params])):
                g_.grad.copy_(s_)
        gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm.detach(), 1.0 - cfg["ema"])
        step += 1
        seen += cfg["batch_views"]
        tot = terms["surf"] + beta * terms["firsthit"]
        losses_hist.append(tot)
        first_loss = first_loss if first_loss is not None else tot
        if want:
            dt = (time.time() - t_last) / (50 if step - 1 > step0 else 1)
            t_last = time.time()
            rec = {"step": step, "loss": float(np.mean(losses_hist[-50:])), **terms, "lr": lr, "grad_norm": float(gn), "s_per_step": dt,
                   "data_s_per_step": data_s / (50 if step - 1 > step0 else 1), "views_per_s": cfg["batch_views"] / max(dt, 1e-9), "seen_views": seen,
                   "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "world": world, "accum": accum}
            data_s = 0.0
            if res_keep:
                rec.update(stats_of(res_keep))
            if rank == 0:
                TR.jl(log_p, rec)
            if step % 500 == 0 or step - 1 == step0:
                say(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
        # ---------------- checkpoints / holdout / stop requests (all ranks reach the same decisions)
        if a.smoke in ("none", "resume"):
            if a.smoke == "none" and (step % cfg["holdout_every"] == 0 or step == steps):
                if rank == 0:
                    stt = holdout(ema, step)
                    if best is None or stt["select_score"] < best["select_score"]:
                        best = {"step": step, "select_score": stt["select_score"], "nll_unknown": stt.get("nll_unknown"), "hc_false_nohit_unknown": stt.get("hc_false_nohit_unknown"), "known_passthrough": stt.get("known_passthrough")}
                        save_tag(step, "best_train_holdout.pt")
                        json.dump(best, open(os.path.join(ckdir, "best_train_holdout.json"), "w"), indent=1)
                    model.train()
                if ddp:
                    dist.barrier()
            if a.smoke == "none" and rank == 0 and (step in tag_steps or step % cfg["tag_every"] == 0 or step == steps):
                save_tag(step)
            stop = 0.0
            if step % 50 == 0:
                if a.stop_file and os.path.exists(a.stop_file):
                    stop = 1.0
                if a.deadline and time.time() > a.deadline:
                    stop = 1.0
                fl = torch.tensor([stop], device=dev)
                if ddp:
                    dist.all_reduce(fl, op=dist.ReduceOp.MAX)
                stop = float(fl)
            if step % cfg["ckpt_every"] == 0 or step == steps or stop:
                save_last(step, "stop_request" if stop else ("final" if step == steps else "periodic"))
            if stop:
                say(f"checkpointed at step {step} on a stop request / deadline; exiting {EXIT_CKPT}")
                if ddp:
                    dist.destroy_process_group()
                sys.exit(EXIT_CKPT)
    if rank == 0:
        run["finished"] = time.ctime()
        run["final_loss_mean50"] = float(np.mean(losses_hist[-50:])) if losses_hist else None
        if a.smoke == "overfit":
            run["overfit"] = {"first_loss": first_loss, "last20_loss": float(np.mean(losses_hist[-20:])), "drop": 1 - float(np.mean(losses_hist[-20:])) / first_loss}
            say("overfit", run["overfit"])
        if a.smoke == "short":
            run["short"] = {"first50": float(np.mean(losses_hist[:50])), "last50": float(np.mean(losses_hist[-50:]))}
            say("short", run["short"])
        if a.smoke == "gradscale":
            med = {k: float(np.median([x[k] for x in gscale])) for k in gscale[0]}
            run["gradscale"] = {"median_grad_norm_all_params": med, "per_step": gscale}
            say("gradscale median", med)
        json.dump(run, open(os.path.join(a.out, f"run_{a.smoke}.json"), "w"), indent=1, default=str)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
