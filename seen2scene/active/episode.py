"""Closed-loop NBV episodes: regional active inspection within an otherwise mapped scene (exp1-task7).

One episode = one target patch (aligned 256^3 box) in one scene:
1. initial observation: the official p_0.01 cameras whose frustum reaches the patch; if there are
   none, 1 valid candidate from step 0;
2. for t = 1..T:
   a. N candidates from CandidateGenerator (seed [2027, scene, episode, t]). Validity rule
      (2026-09-27): after outside-patch occlusion, at least one ray still reaches the patch;
   b. the planner scores them; argmax is chosen, ties go to the first candidate;
   c. the chosen view is rendered like the official data (pyrender 512^2, znear 0.05,
      zfar 20, uint16) and fused into the patch (PatchFusion, official vdbfusion semantics);
   d. metrics (frozen in eval/metrics_known.py) are logged: surface coverage and free-space coverage.
All planners share the initial observation, the candidate sets and their order, the renderers,
the fusion and the metrics. Only `Planner.score` differs.
"""
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from seen2scene import FRONT3D_DIR
from seen2scene.active.camera import CandidateConfig, CandidateGenerator, load_official_cameras, scene_hash
from seen2scene.active.fusion import PatchFusion, quantize_depth
from seen2scene.active.raycast import PatchRaycaster, camera_rays
from seen2scene.active.render import SceneRenderer

VS, TR, KR = 0.011, 0.033, 0.999
EMPTY, UNKNOWN, BAND = 0, 1, 2


@dataclass
class EpisodeConfig:
    T: int = 20
    N: int = 64
    score_res: int = 128   # rays per side for planner scoring and the validity rule
    sensor_res: int = 512  # executed observation, official setting
    fov_deg: float = 60.0
    znear: float = 0.05
    zfar: float = 20.0


def structure_torch(vol: torch.Tensor) -> torch.Tensor:
    s = torch.full_like(vol, UNKNOWN, dtype=torch.int8)
    s[vol >= TR * KR] = EMPTY
    s[vol.abs() < TR * KR] = BAND
    return s


class Episode:
    def __init__(self, scene: str, bbox, gt_dir: str, cfg: EpisodeConfig, episode_id: int = 0, device: str = "cuda"):
        self.scene, self.cfg, self.ep = scene, cfg, episode_id
        self.bbox = np.asarray(bbox, dtype=np.float64)
        self.dev = torch.device(device)
        self.mesh = trimesh.load(os.path.join(FRONT3D_DIR, scene, "scene.ply"), process=False)
        self.meta = json.load(open(os.path.join(FRONT3D_DIR, scene, "meta.json")))
        gt = torch.as_tensor(np.load(os.path.join(gt_dir, "volume.npy")), device=self.dev)
        self.gt_struct = structure_torch(gt)
        self.gt_band = self.gt_struct == BAND
        self.sensor = SceneRenderer(self.mesh, cfg.sensor_res, cfg.sensor_res, cfg.fov_deg, znear=cfg.znear, zfar=cfg.zfar)
        self.scorer = SceneRenderer(self.mesh, cfg.score_res, cfg.score_res, cfg.fov_deg, znear=cfg.znear, zfar=cfg.zfar)
        self.fusion = PatchFusion(self.bbox[:3], device=device)
        self.rc = PatchRaycaster(self.bbox[:3], device=device)
        self.gen = CandidateGenerator(scene, self.mesh, self.meta, self.bbox.tolist(), CandidateConfig(n=cfg.N))
        self.log: List[Dict] = []

    # ---------------------------------------------------------------- observation state
    def observe(self, c2w: np.ndarray):
        self.fusion.integrate(quantize_depth(self.sensor.render_depth(c2w)), c2w)

    def obs_struct(self) -> torch.Tensor:
        return structure_torch(self.fusion.tsdf())

    def coverage(self, obs: Optional[torch.Tensor] = None) -> Dict[str, float]:
        obs = self.obs_struct() if obs is None else obs
        dil = F.max_pool3d((obs == BAND).float()[None, None], 3, 1, 1)[0, 0] > 0
        gt_empty = self.gt_struct == EMPTY
        return {"surface_coverage": float((self.gt_band & dil).sum() / self.gt_band.sum().clamp(min=1)),
                "freespace_coverage": float((gt_empty & (obs != UNKNOWN)).sum() / gt_empty.sum().clamp(min=1))}

    # ---------------------------------------------------------------- candidates
    def outside_tmax(self, c2w: np.ndarray):
        return self.rc.outside_block_tmax(c2w, self.scorer, self.cfg.score_res, self.cfg.fov_deg)

    def reaches_patch(self, c2w: np.ndarray) -> bool:
        """Validity rule: after outside-patch occlusion, at least one ray still enters the patch."""
        t_max, _ = self.outside_tmax(c2w)
        dirs, org, _ = camera_rays(c2w, self.cfg.score_res, self.cfg.fov_deg)
        d = torch.as_tensor(dirs, device=self.dev)
        o = (torch.as_tensor(org, device=self.dev) - self.rc.bmin).expand_as(d)
        tmin, tmax = self.rc._entry_exit(o, d)
        return bool(((tmax > tmin) & torch.isinf(t_max)).any())

    def candidates(self, step: int) -> List[Dict]:
        return self.gen.generate(self.ep, step, self.cfg.N, extra_check=self.reaches_patch)

    def initial_cameras(self) -> List[np.ndarray]:
        cams = load_official_cameras(os.path.join(FRONT3D_DIR, self.scene, "cameras_p_0.01.ply"))
        cams = [c for c in cams if self.gen.frustum_hits_patch(c)]
        if not cams:
            cands = self.gen.generate(self.ep, 0, 1, extra_check=self.reaches_patch)
            cams = [cands[0]["c2w"]] if cands else []
        return cams

    # ---------------------------------------------------------------- loop
    def run(self, planner, out_path: Optional[str] = None) -> List[Dict]:
        t0 = time.time()
        init = self.initial_cameras()
        for c2w in init:
            self.observe(c2w)
        self.log = [{"step": 0, "n_initial_views": len(init), **self.coverage(), "seconds": time.time() - t0}]
        for t in range(1, self.cfg.T + 1):
            ts = time.time()
            cands = self.candidates(t)
            tg = time.time()
            if not cands:
                self.log.append({"step": t, "n_candidates": 0, **self.coverage()})
                continue
            obs = self.obs_struct()
            planner.last_info = {}
            scores = np.asarray(planner.score(self, cands, obs, t), dtype=np.float64)
            tsc = time.time()
            pick = int(np.argmax(scores))
            self.observe(cands[pick]["c2w"])
            self.log.append({"step": t, "n_candidates": len(cands), "pick": pick, "score": float(scores[pick]),
                             "c2w": cands[pick]["c2w"].tolist(), **self.coverage(),
                             "t_candidates": tg - ts, "t_scoring": tsc - tg, "t_execute": time.time() - tsc,
                             **getattr(planner, "last_info", {})})
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fallback = sum(1 for s in self.log if s.get("planner_fallback"))
            json.dump({"scene": self.scene, "bbox": self.bbox.tolist(), "episode": self.ep, "planner": planner.name,
                       "config": asdict(self.cfg), "fallback_count": fallback, "log": self.log},
                      open(out_path, "w"), indent=1)
        return self.log

    def close(self):
        self.sensor.close(); self.scorer.close(); self.gen.close()


# -------------------------------------------------------------------- planners
class RandomPlanner:
    name = "random"

    def score(self, ep: Episode, cands, obs, step):
        rng = np.random.default_rng([2027, scene_hash(ep.scene), ep.ep, step, 7])
        return rng.random(len(cands))


class OraclePlanner:
    """Greedy on the true increment of the evaluation surface coverage, from the ground truth.

    For each candidate, the full scene mesh is rendered at score_res, which also gives the
    outside-patch occlusion. Hits inside the patch are turned into voxels, dilated by 1 voxel
    (the coverage rule), and the GT surface voxels among them that are not covered yet are counted.
    """
    name = "oracle"

    def score(self, ep: Episode, cands, obs, step):
        covered = F.max_pool3d((obs == BAND).float()[None, None], 3, 1, 1)[0, 0] > 0
        target = ep.gt_band & ~covered
        lo = torch.as_tensor(ep.bbox[:3], device=ep.dev)
        out = []
        for c in cands:
            depth = ep.scorer.render_depth(c["c2w"])
            pts = torch.as_tensor(ep.scorer.backproject(depth, c["c2w"]), device=ep.dev)
            ijk = torch.floor((pts - lo) / VS).long()
            ok = ((ijk >= 0) & (ijk < 256)).all(1)
            if not ok.any():
                out.append(0.0); continue
            hit = torch.zeros((256,) * 3, dtype=torch.float32, device=ep.dev)
            ijk = ijk[ok]
            hit[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1.0
            hit = F.max_pool3d(hit[None, None], 3, 1, 1)[0, 0] > 0
            out.append(float((hit & target).sum()))
        return out


class FrontierPlanner:
    """B-Frontier: unique frontier voxels directly visible (definition frozen 2026-09-27, see frontier.py).
    Uses only the current observation; outside-patch occlusion as for G_cov."""
    name = "frontier"

    def score(self, ep: Episode, cands, obs, step):
        from seen2scene.active.frontier import frontier_mask, frontier_score

        fr = frontier_mask(obs)
        out = []
        for c in cands:
            t_max, _ = ep.outside_tmax(c["c2w"])
            out.append(float(frontier_score(ep.rc, obs, fr, c["c2w"], ep.cfg.score_res, ep.cfg.fov_deg, t_max=t_max)))
        return out


class ExpectedCoveragePlanner:
    """S1 (K=1) and S2 (K>1): mean frozen G_cov over K S0-F worlds of the current observation (exp2-task5/6).

    World k at step t uses seed_world(scene, episode, t, k) (k 0-based), so S1's world is world k=0 of S2's set
    for the same state. The world model is the official S0-F completion (world_model.py, checked bit for bit).
    Fallback (exp2 rule, identical for S1, S2 and later VHD): only if the world model cannot produce a valid
    world -- the observation has no surface voxel (the official input grid cannot be built), or a world is
    non-finite or has no surface -- the step takes the Random planner's choice and the log records
    planner_fallback; the episode JSON reports fallback_count.
    """

    def __init__(self, world_model, ks=(0,), name=None, keep_worlds: bool = False):
        self.wm, self.ks = world_model, tuple(ks)
        self.name = name or ("s1" if len(self.ks) == 1 else f"s2_k{len(self.ks)}")
        self.keep_worlds = keep_worlds  # debugging / equivalence checks only
        self.last_info, self.last_worlds, self.last_per_world = {}, None, None

    def score(self, ep: Episode, cands, obs, step):
        """Streaming over the K worlds (exp2-task6, for GPU memory): generate world k -> G_cov of every candidate
        -> accumulate -> free. Checked equal to generating the K worlds at once (task6_streaming_check.py)."""
        from seen2scene.active.world_model import EmptyObservation

        torch.cuda.reset_peak_memory_stats(ep.dev)
        t_all = time.time()
        vol = ep.fusion.tsdf()
        t_max = [ep.outside_tmax(c["c2w"])[0] for c in cands]  # outside-patch occlusion, shared by the K worlds
        per_world, t_world, t_ray, bands, kept = [], 0.0, 0.0, [], []
        unobserved = obs == UNKNOWN
        for k in self.ks:
            t0 = time.time()
            try:
                w = self.wm.generate(ep.scene, vol, episode=ep.ep, step=step, ks=(k,))
                reason = None if bool(torch.isfinite(w).all()) and bool((w.abs() < TR * KR).any()) else "invalid_world"
            except EmptyObservation:
                w, reason = None, "empty_observation"
            t_world += time.time() - t0
            if reason is not None:
                self.last_info = {"planner_fallback": reason, "planner_t_world": t_world}
                self.last_worlds = self.last_per_world = None
                return RandomPlanner().score(ep, cands, obs, step)
            t1 = time.time()
            g = [float(ep.rc.cast(w, c["c2w"], ep.cfg.score_res, ep.cfg.fov_deg, t_max=tm, observed_structure=obs)["g_cov"][0])
                 for c, tm in zip(cands, t_max)]
            t_ray += time.time() - t1
            per_world.append(g)
            if len(self.ks) > 1:
                bands.append((w[0].abs() < TR * KR) & unobserved)
            if self.keep_worlds:
                kept.append(w[0].clone())
            del w
        G = np.asarray(per_world, dtype=np.float64)  # [K, n_candidates]
        scores = G.mean(0)
        info = {"planner_fallback": None, "planner_t_world": t_world, "planner_t_raycast": t_ray,
                "planner_t_total": time.time() - t_all,
                "planner_peak_mem_gb": torch.cuda.max_memory_allocated(ep.dev) / 2 ** 30}
        if len(self.ks) > 1:  # world / planner disagreement between the K worlds
            from scipy.stats import spearmanr

            K, n_unobs = len(self.ks), max(int(unobserved.sum()), 1)
            pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
            info["planner_world_argmax"] = [int(np.argmax(G[k])) for k in range(K)]
            info["planner_argmax_agree_with_mean"] = float(np.mean([np.argmax(G[k]) == np.argmax(scores) for k in range(K)]))
            info["planner_pairwise_spearman"] = float(np.nanmean([spearmanr(G[a], G[b]).correlation for a, b in pairs]))
            info["planner_unobs_state_disagreement"] = float(np.mean([(bands[a] ^ bands[b]).sum().item() / n_unobs for a, b in pairs]))
        self.last_info = info
        self.last_per_world = G
        self.last_worlds = kept if self.keep_worlds else None
        return scores.tolist()


VHD_TAU = 0.10  # m, depth saturation of the A3 per-ray disagreement (exp_vis task1-A, frozen 2026-09-29)


def vhd_ray_terms(t: torch.Tensor, valid: torch.Tensor, tau: float = VHD_TAU):
    """A3 per-ray disagreement of K worlds (frozen definition, preregistered for VHD-v1).

    t [K, ...]: first-hit ray distance in metres (inf = no hit); valid [...]: M_valid. For each world pair:
    1 if exactly one hits, min(1, |t_a - t_b| / tau) if both hit, 0 if neither; c = mean over the pairs.
    Returns hit [K, ...], h (hit/miss part = n_hit (K - n_hit) / n_pairs) and delta (depth part), float64;
    c = h + delta in [0, 1].
    """
    K = t.shape[0]
    hit = torch.isfinite(t) & valid[None]
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    zero = torch.zeros(valid.shape, dtype=torch.float64, device=t.device)
    h, delta = zero.clone(), zero.clone()
    for a, b in pairs:
        both = hit[a] & hit[b]
        h += (hit[a] ^ hit[b]).double()
        dt = torch.where(both, (t[a].double() - t[b].double()).abs(), zero)  # inf - inf only where not both
        delta += torch.where(both, torch.clamp(dt / tau, max=1.0), zero)
    return hit, h / len(pairs), delta / len(pairs)


def vhd_scores(t: torch.Tensor, new: torch.Tensor, valid: torch.Tensor, tau: float = VHD_TAU) -> Dict[str, np.ndarray]:
    """Candidate scores of VHD-v1 (experiments/vhd_v1/preregistration.md).

    t, new [K, n, R] (first-hit distance, G_cov new-surface flag of the hit), valid [n, R]. With R the fixed ray
    budget (128^2) as denominator:
      vhd_main = 1/R sum_r M_valid M_novel c(r),  M_novel(r) = any_k hit_k(r) & new_k(r)   (the planner score)
      vhd_raw  = 1/R sum_r M_valid c(r)                                                    (diagnostic only)
    plus the hit/miss (H) and depth (D) parts of both and the per-candidate ray counts.
    """
    R = valid.shape[-1]
    hit, h, d = vhd_ray_terms(t, valid, tau)
    novel = (hit & new).any(0)
    out = {"n_valid": valid.sum(-1), "n_novel": (valid & novel).sum(-1)}
    for name, m in (("main", valid & novel), ("raw", valid)):
        out[f"vhd_{name}"] = ((h + d) * m).sum(-1) / R
        out[f"H_{name}"] = (h * m).sum(-1) / R
        out[f"D_{name}"] = (d * m).sum(-1) / R
    return {k: v.cpu().numpy() for k, v in out.items()}


class VHDPlanner(ExpectedCoveragePlanner):
    """VHD-v1: novelty-masked future-observation disagreement of the same K=4 worlds as S2 (frozen 2026-10-02).

    Everything that S2 uses is shared: world model and seeds seed_world(scene, episode, t, k), the 64 candidates,
    the 128^2 PatchRaycaster.cast, the outside-patch occlusion (once per candidate), the streaming over the worlds.
    Only the score differs: G_VHD(c) = 1/128^2 sum_r M_valid(r) M_novel(r) c(r) with the A3 per-ray disagreement c,
    M_novel from the frozen G_cov new-surface rule (current observation only, no GT). argmax, ties -> lowest id.
    Fallbacks, logged separately: the S1/S2 world-model rule (`planner_fallback`), and `vhd_zero_fallback` when all
    64 G_VHD are 0; both take the frozen Random planner's choice (never S2). VHD_raw (no novelty mask), the H/D parts,
    the per-world G_cov of every candidate (S2's choice on the same worlds) are logged as diagnostics only.
    """

    def __init__(self, world_model, ks=(0, 1, 2, 3), name: str = "vhd"):
        super().__init__(world_model, ks, name=name)

    def score(self, ep: Episode, cands, obs, step):
        from scipy.stats import spearmanr

        from seen2scene.active.world_model import EmptyObservation

        torch.cuda.reset_peak_memory_stats(ep.dev)
        t_all = time.time()
        res, fov = ep.cfg.score_res, ep.cfg.fov_deg
        vol = ep.fusion.tsdf()
        t_max = [ep.outside_tmax(c["c2w"])[0] for c in cands]  # outside-patch occlusion, shared by the K worlds
        valid = []
        for c, tm in zip(cands, t_max):  # M_valid: the same test as PatchRaycaster.cast
            dirs, org, _ = camera_rays(c["c2w"], res, fov)
            d = torch.as_tensor(dirs, device=ep.dev)
            o = (torch.as_tensor(org, device=ep.dev) - ep.rc.bmin).expand_as(d)
            tmin, tmax = ep.rc._entry_exit(o, d)
            valid.append(torch.minimum(tmax, tm.to(ep.dev)) > tmin)
        valid = torch.stack(valid)
        K, n, R = len(self.ks), len(cands), valid.shape[1]
        T = torch.full((K, n, R), float("inf"), device=ep.dev)
        NEW = torch.zeros((K, n, R), dtype=torch.bool, device=ep.dev)
        G = np.zeros((K, n), dtype=np.float64)
        t_world, t_ray, bands = 0.0, 0.0, []
        unobserved = obs == UNKNOWN
        for ki, k in enumerate(self.ks):
            t0 = time.time()
            try:
                w = self.wm.generate(ep.scene, vol, episode=ep.ep, step=step, ks=(k,))
                reason = None if bool(torch.isfinite(w).all()) and bool((w.abs() < TR * KR).any()) else "invalid_world"
            except EmptyObservation:
                w, reason = None, "empty_observation"
            t_world += time.time() - t0
            if reason is not None:  # world_invalid_fallback (the S1/S2 rule)
                self.last_info = {"planner_fallback": reason, "vhd_zero_fallback": False, "planner_t_world": t_world}
                return RandomPlanner().score(ep, cands, obs, step)
            t1 = time.time()
            for j, (c, tm) in enumerate(zip(cands, t_max)):
                out = ep.rc.cast(w, c["c2w"], res, fov, t_max=tm, observed_structure=obs)
                T[ki, j], NEW[ki, j], G[ki, j] = out["t"][0], out["new_surface"][0], float(out["g_cov"][0])
            t_ray += time.time() - t1
            bands.append((w[0].abs() < TR * KR) & unobserved)
            del w
        t2 = time.time()
        sc = vhd_scores(T, NEW, valid)
        t_vhd = time.time() - t2
        zero = bool(sc["vhd_main"].max() == 0.0)
        scores = RandomPlanner().score(ep, cands, obs, step) if zero else sc["vhd_main"].tolist()
        Gm = G.mean(0)  # S2's score on the same worlds (diagnostic only)
        pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
        n_unobs = max(int(unobserved.sum()), 1)
        info = {"planner_fallback": None, "vhd_zero_fallback": zero, "planner_t_world": t_world, "planner_t_raycast": t_ray,
                "planner_t_vhd": t_vhd, "planner_t_total": time.time() - t_all,
                "planner_peak_mem_gb": torch.cuda.max_memory_allocated(ep.dev) / 2 ** 30,
                "planner_world_argmax": [int(np.argmax(G[k])) for k in range(K)],
                "planner_pairwise_spearman": float(np.nanmean([spearmanr(G[a], G[b]).correlation for a, b in pairs])),
                "planner_unobs_state_disagreement": float(np.mean([(bands[a] ^ bands[b]).sum().item() / n_unobs for a, b in pairs])),
                "s2_equiv_pick": int(np.argmax(Gm)), "s2_equiv_score": float(Gm[int(np.argmax(Gm))]),
                "world_g_cov": G.tolist(),
                **{f"vhd_{k}" if not k.startswith("vhd") else k: v.tolist() for k, v in sc.items()}}
        self.last_info = info
        return scores


PLANNERS = {"random": RandomPlanner, "oracle": OraclePlanner, "frontier": FrontierPlanner}
MODEL_PLANNERS = {"s1": (0,), "s2": (0, 1, 2, 3), "vhd": (0, 1, 2, 3)}  # need a WorldModel; ks of the worlds


def make_planner(name: str, world_model=None):
    if name in MODEL_PLANNERS:
        assert world_model is not None, f"{name} needs a WorldModel"
        if name == "vhd":
            return VHDPlanner(world_model, MODEL_PLANNERS[name], name=name)
        return ExpectedCoveragePlanner(world_model, MODEL_PLANNERS[name], name=name)
    return PLANNERS[name]()
