"""Unified outdoor-only benchmark (experiments/exp_unified_outdoor): arm registry + run-time patches applied ON TOP of
the UNCHANGED historical trainers. No historical file is edited; every patch replaces a module attribute in the current
process only, after checking that the attribute exists (a renamed attribute fails loudly instead of being ignored).

Trainers (imported, their main() is called unchanged):
  v2    experiments/exp5_future_view/task1/scripts/t12_train.py      (V2 Original)
  v4    experiments/exp6_v4/scripts/train_v4.py                       (V4 = V2 + RayXAttn)
  v3t1  experiments/exp6_v3.1/scripts/trainer_outdoor.py              (V3 outdoor loop) with xattn enabled (= V3-T1 model)
  v5    experiments/exp6_v5/scripts/train_v5.py --arm v5|v4c          (V5 warm start / V4-Continue)

Patches
  ExactResume   per-rank RNG snapshot at the start of every checkpoint step (numpy sampler PCG64, torch CPU, torch CUDA,
                V5 query generator) + exact restore on resume. Hooked on the per-step LR function, which every trainer
                calls exactly once per step (V5: once per parameter group with the same step) BEFORE drawing the batch;
                the trainer's live objects (rng, gen) are read from its main() frame and mutated in place. The
                historical trainers save only rank 0's RNG and, under DDP, re-seed every rank from (seed, rank, step0)
                on resume (t12_train.py:203-205, train_v4.py:121-123, trainer_outdoor.py:124-125, train_v5.py:257-260):
                reproducible but different from an uninterrupted run. With the snapshot the resumed run continues the
                uninterrupted RNG streams exactly; if any rank lacks the snapshot of the resume step, the trainer's own
                re-seed is kept for ALL ranks and the resume is logged as inexact (resume_log.jsonl).
  data roots    v2: v2_data.ViewData default root (t12_train passes none; v2_data.py:34) -> cfg data_root
                v4: none (train_v4 reads cfg data_root, train_v4.py:78)
                v3t1: seen2scene.v3.data.shard_path / CACHE -> cfg data_root / tsdf_cache_root
                v5: train_v5.OUTD (train_v5.py:49) -> cfg data_root; train_v5.V4CK / V4SHA (train_v5.py:51-52) -> the
                    unified V4@48k parent (sha256 recorded in <out>/PARENT.json at the first launch, checked afterwards)
  v3t1 model    seen2scene.v3.trainer.make_model sets mcfg["xattn"]["enabled"] = True before construction (exactly what
                the historical V3 trainer does for --arm t1, trainer.py:105-108); trainer_outdoor's assert on the config
                file (trainer_outdoor.py:101) still sees the unchanged model_v3.json (enabled = false)
  v3t1 sampler  "v2_pool": every rank loads ALL Train-6-minus-S-c trajectories in the frozen V2 split order
                (v2_data.load_split) instead of the sorted [rank::world] shard (trainer_outdoor.py:60), and the per-rank
                step batch is drawn by ONE V2-rule call with batch_views / world views (identical rng consumption and
                identical (state, views, staleness) as V2 / V4 for the same seed and rank), then split at a state
                boundary into `accum` micro-batches (minimal largest part) for memory.
"""
import functools
import hashlib
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
CFG = os.path.join(EXP, "configs")
WU = os.path.join(REPO, "work_dirs", "exp_unified_outdoor")
RUNS = os.path.join(WU, "runs")
S2E = "/bsuscratch/jiahuizhang/envs/seen2scene"

DATA_ROOT = {"s10": "work_dirs/exp5_future_view/task1/data/views",
             "d100": "work_dirs/exp5_future_view/task3_full_depth_p1/e2/out_views_d100"}
TSDF_ROOT = {"d100": "work_dirs/exp6_v3/shared_cache/tsdf4",
             "s10": "work_dirs/exp_unified_outdoor/cache/tsdf4_s10"}
V5_LABELS = "work_dirs/exp6_v5/labels/v1"

TRAINER_DIR = {"v2": "experiments/exp5_future_view/task1/scripts",
               "v4": "experiments/exp6_v4/scripts",
               "v3t1": "experiments/exp6_v3.1/scripts",
               "v5": "experiments/exp6_v5/scripts"}
TRAINER_MOD = {"v2": "t12_train", "v4": "train_v4", "v3t1": "trainer_outdoor", "v5": "train_v5"}

ARMS = {}
for _d in ("s10", "d100"):
    ARMS[f"v2_{_d}"] = {"trainer": "v2", "density": _d, "final_step": 60000}
    ARMS[f"v4_{_d}"] = {"trainer": "v4", "density": _d, "final_step": 60000}
    ARMS[f"v3t1_{_d}"] = {"trainer": "v3t1", "density": _d, "final_step": 60000}
    ARMS[f"v4c12k_from48k_{_d}"] = {"trainer": "v5", "v5_arm": "v4c", "density": _d, "final_step": 12000, "parent_arm": f"v4_{_d}"}
    ARMS[f"v5warm12k_from48k_{_d}"] = {"trainer": "v5", "v5_arm": "v5", "density": _d, "final_step": 12000, "parent_arm": f"v4_{_d}"}
for _k, _v in ARMS.items():
    _v["config"] = os.path.join(CFG, f"{_k}.json")
PARENT_STEP = 48000


def rp(p):
    """Repo-relative -> absolute."""
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def env_rank():
    return int(os.environ.get("RANK", "0"))


def setattr_checked(mod, name, value):
    if not hasattr(mod, name):
        raise AttributeError(f"patch target {getattr(mod, '__name__', mod)}.{name} does not exist (historical code changed?)")
    setattr(mod, name, value)


def add_paths(trainer):
    for p in (os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"),
              os.path.join(REPO, "experiments", "exp5_future_view", "task1", "scripts"),
              rp(TRAINER_DIR[trainer]), REPO):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_cfg(arm):
    return json.load(open(ARMS[arm]["config"]))


# ------------------------------------------------------------------------------------------------ exact resume
class ExactResume:
    """Per-rank RNG snapshots at checkpoint steps + exact restore (see module doc)."""

    def __init__(self, log=print):
        self.last, self.ready, self.enabled, self.restored_at = None, False, False, None
        self.log = log
        self.n_snap = 0

    def wrap(self, fn):
        @functools.wraps(fn)
        def hooked(step, *args, **kw):
            s = int(step)
            if s != self.last:
                self.last = s
                self._on_step(s)
            return fn(step, *args, **kw)
        hooked.__wrapped_exact_resume__ = True
        return hooked

    @staticmethod
    def _main_locals():
        f = sys._getframe(3)
        while f is not None:
            if f.f_code.co_name == "main":
                L = f.f_locals
                if "rng" in L and "cfg" in L and "a" in L:
                    return L
            f = f.f_back
        raise RuntimeError("ExactResume: trainer main() frame with rng / cfg / a not found")

    def _path(self, step, rank):
        return os.path.join(self.dir, f"s{step:07d}_r{rank}.pt")

    def _on_step(self, s):
        L = self._main_locals()
        if not self.ready:
            self.ready = True
            a, cfg = L["a"], L["cfg"]
            self.rank, self.world, self.ddp = int(L.get("rank", 0)), int(L.get("world", 1)), bool(L.get("ddp", False))
            self.every = int(cfg["ckpt_every"])
            self.dir = os.path.join(a.out, "rng_states")
            self.out = a.out
            self.enabled = getattr(a, "smoke", "none") in ("none", "resume")
            if self.enabled:
                os.makedirs(self.dir, exist_ok=True)
                step0 = int(L.get("step0", 0))
                if step0 > 0:
                    self._restore(L, step0)
        if self.enabled and s > 0 and s % self.every == 0 and s != self.restored_at:
            self._snapshot(L, s)

    def _snapshot(self, L, s):
        import torch
        gen = L.get("gen")
        st = {"step": s, "rank": self.rank, "world": self.world, "np_rng": L["rng"].bit_generator.state,
              "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
              "gen": gen.get_state() if gen is not None else None, "time": time.time()}
        p = self._path(s, self.rank)
        torch.save(st, p + ".tmp")
        os.replace(p + ".tmp", p)
        self.n_snap += 1

    def _restore(self, L, step0):
        import torch
        p = self._path(step0, self.rank)
        have = os.path.exists(p)
        st = None
        if have:
            st = torch.load(p, map_location="cpu", weights_only=False)
            have = st["step"] == step0 and st["rank"] == self.rank and st["world"] == self.world
        ok = have
        if self.ddp:
            import torch.distributed as dist
            t = torch.tensor([1 if have else 0], device="cuda", dtype=torch.int32)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            ok = bool(int(t.item()))
        if ok:
            L["rng"].bit_generator.state = st["np_rng"]
            torch.set_rng_state(st["torch_rng"])
            if st["cuda_rng"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(st["cuda_rng"])
            if st["gen"] is not None and L.get("gen") is not None:
                L["gen"].set_state(st["gen"])
        self.restored_at = step0
        rec = {"time": time.strftime("%F %T"), "step0": step0, "rank": self.rank, "world": self.world, "exact": ok,
               "note": "per-rank RNG restored from rng_states" if ok else "snapshot missing on >= 1 rank: trainer re-seed (seed, rank, step0) kept on all ranks"}
        with open(os.path.join(self.out, f"resume_log_r{self.rank}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.rank == 0:
            self.log(f"[exact-resume] step {step0}: {'EXACT' if ok else 'INEXACT (re-seeded)'}")


# ------------------------------------------------------------------------------------------------ V3 pooled sampler
def split_states(batch, k):
    """Contiguous partition of a list of states (ti, t, dlt, rows) into k non-empty parts minimising the largest number of
    views of a part (ties: lexicographically smallest cut positions). Deterministic."""
    if k == 1:
        return [list(batch)]
    if len(batch) < k:
        raise RuntimeError(f"cannot split {len(batch)} states into {k} micro-batches")
    nv = [len(x[3]) for x in batch]
    best = None
    for cuts in itertools.combinations(range(1, len(batch)), k - 1):
        b = (0,) + cuts + (len(batch),)
        key = (max(sum(nv[i:j]) for i, j in zip(b[:-1], b[1:])), cuts)
        if best is None or key < best[0]:
            best = (key, [list(batch[i:j]) for i, j in zip(b[:-1], b[1:])])
    return best[1]


def pooled_v3_views_class(D):
    class PooledV3Views(D.V3Views):
        """V3Views whose sample_batch draws the whole per-rank step batch with ONE V2-rule call (n_views x accum views)
        and hands it out as `accum` state-boundary micro-batches on consecutive calls."""

        def __init__(self, domain, items_, log=print, accum=1):
            super().__init__(domain, items_, log)
            self.accum = int(accum)
            self._queue = []
            self.n_draws = 0

        def sample_batch(self, rng, n_views=32, n_states=4, max_cands=8, stale_p=0.5, stale_max=9, weights=None):
            if self.accum == 1:
                return super().sample_batch(rng, n_views, n_states, max_cands, stale_p, stale_max, weights)
            if not self._queue:
                full = super().sample_batch(rng, n_views * self.accum, n_states * self.accum, max_cands, stale_p, stale_max, weights)
                self._queue = split_states(full, self.accum)
                self.n_draws += 1
            return self._queue.pop(0)
    return PooledV3Views


# ------------------------------------------------------------------------------------------------ patch sets
def patch_v2(cfg, say=print):
    import v2_data as VD
    root = rp(cfg["data_root"])
    Orig = VD.ViewData

    class RootedViewData(Orig):
        """v2_data.ViewData reading the unified data_root (t12_train never passes root)."""

        def __init__(self, items, root=None, keys=VD.VIEW_KEYS, log=print):
            super().__init__(items, root=root or RootedViewData.ROOT, keys=keys, log=log)
    RootedViewData.ROOT = root
    setattr_checked(VD, "ViewData", RootedViewData)
    return {"v2_data.ViewData.root": root}


def patch_v3t1(cfg, say=print):
    import trainer_outdoor as TO
    import v2_data as VD
    from seen2scene.v3 import data as D
    from seen2scene.v3 import trainer as T0
    root, cache = rp(cfg["data_root"]), rp(cfg["tsdf_cache_root"])
    orig_shard = D.shard_path

    def shard_path(domain, scene, traj):
        return os.path.join(root, scene, f"{traj}.npz") if domain == "outdoor" else orig_shard(domain, scene, traj)
    setattr_checked(D, "shard_path", shard_path)
    setattr_checked(D, "CACHE", cache)
    setattr_checked(TO, "CFG_DIR", REPO)          # cfg["model"] is repo-relative in the unified configs
    arm = cfg["unified"]["v3_arm"]
    orig_make = T0.make_model

    def make_model(mcfg, seed, dev):
        mcfg["xattn"]["enabled"] = arm == "t1"     # = seen2scene.v3.trainer.model_cfg (trainer.py:105-108)
        return orig_make(mcfg, seed, dev)
    setattr_checked(T0, "make_model", make_model)
    patches = {"seen2scene.v3.data.shard_path(outdoor)": root, "seen2scene.v3.data.CACHE": cache,
               "trainer_outdoor.CFG_DIR": REPO, "seen2scene.v3.trainer.make_model": f"xattn.enabled = {arm == 't1'}"}
    if cfg["unified"].get("sampler") == "v2_pool":
        Pooled = pooled_v3_views_class(D)
        accum = int(cfg.get("accum", 1))

        def load_data(rank, world, smoke, say_):
            train, held = VD.load_split(cfg["train_set"])
            if smoke:                              # optional worst-case memory smoke: UO_SMOKE_SCENE=dunnottar (largest grid)
                sc = os.environ.get("UO_SMOKE_SCENE")
                train, held = ([x for x in train if x[0] == sc][:6], [x for x in held if x[0] == sc][:3]) if sc else (train[:6], held[:3])
            n_all = {"out_train": len(train), "out_sc": len(held)}
            if rank != 0:
                held = held[:1]
            say_(f"[v2_pool] every rank loads all {len(train)} train trajectories (V2 split order); S-c {len(held)} on this rank; accum {accum}")
            return {"out_train": Pooled("outdoor", train, say_, accum=accum), "out_sc": D.V3Views("outdoor", held, say_)}, n_all
        setattr_checked(TO, "load_data", load_data)
        patches["trainer_outdoor.load_data"] = "v2_pool (all train trajectories per rank, V2 split order, one V2-rule draw per step split into accum micro-batches)"
    return patches


def patch_v5(cfg, out, say=print, check_parent=True):
    import train_v5 as T5
    setattr_checked(T5, "OUTD", rp(cfg["data_root"]))
    setattr_checked(T5, "CFG", REPO)              # cfg["model"] is repo-relative in the unified configs
    par = cfg["unified"]["parent"]
    ck = rp(par["ckpt"])
    patches = {"train_v5.OUTD": rp(cfg["data_root"]), "train_v5.CFG": REPO, "train_v5.V4CK": ck}
    if not check_parent:
        return patches
    if not os.path.exists(ck):
        raise SystemExit(f"STOP: parent checkpoint missing: {ck} (train arm {par['arm']} to step {par['step']} first)")
    sha = sha256_file(ck)
    rec_p = os.path.join(out, "PARENT.json")
    if os.path.exists(rec_p):
        rec = json.load(open(rec_p))
        if rec["sha256"] != sha:
            raise SystemExit(f"STOP: parent {ck} changed since the first launch ({rec['sha256'][:12]} -> {sha[:12]})")
    elif env_rank() == 0:
        import torch
        z = torch.load(ck, map_location="cpu", weights_only=False)
        if int(z.get("step", -1)) != int(par["step"]) or not z["model_cfg"].get("ray_xattn", {}).get("enabled"):
            raise SystemExit(f"STOP: parent {ck} is not a V4 EMA at step {par['step']} (step {z.get('step')})")
        os.makedirs(out, exist_ok=True)
        json.dump({"parent_arm": par["arm"], "ckpt": ck, "step": par["step"], "sha256": sha, "recorded": time.strftime("%F %T")},
                  open(rec_p, "w"), indent=1)
    setattr_checked(T5, "V4CK", ck)
    setattr_checked(T5, "V4SHA", sha)
    patches["train_v5.V4SHA"] = sha
    return patches


def install_exact_resume(trainer, er):
    """Hook the per-step LR function of each trainer (looked up as a module attribute at call time)."""
    if trainer in ("v2", "v4"):
        import t12_train as TR            # train_v4 calls TR.lr_at (train_v4.py:154); t12_train calls lr_at (t12_train.py:224)
        setattr_checked(TR, "lr_at", er.wrap(TR.lr_at))
        return "t12_train.lr_at"
    if trainer == "v3t1":
        import trainer_outdoor as TO      # trainer_outdoor.py:176
        setattr_checked(TO, "lr_at", er.wrap(TO.lr_at))
        return "trainer_outdoor.lr_at"
    if trainer == "v5":
        import train_v5 as T5             # train_v5.py:300-303 (via lrs)
        setattr_checked(T5, "lr_cos", er.wrap(T5.lr_cos))
        return "train_v5.lr_cos"
    raise ValueError(trainer)


def git_sha():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception as e:                       # noqa: BLE001
        return f"error: {e}"
