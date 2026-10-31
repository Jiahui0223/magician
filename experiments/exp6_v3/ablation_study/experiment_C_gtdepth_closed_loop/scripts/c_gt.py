"""EXP6 V3 Ablation Experiment C -- GT query providers (magician env, simulator / evaluation side; GT mesh and the
frozen coverage evaluators). Stateless with respect to the simulator: every call reads the current coverage state and
returns per-candidate numbers; nothing is written back (checked by state_hash before / after each query in the loops).

IndoorGT  (MP3D)       tstar : first GT intersection of the 64 x 114 scoring rays (t11_labels.Renderer, jitter 0,
                               model units k_I) -- the EXP5 S1 renderer, bitwise
                       gains : true one-step C_obs gain of every candidate = |visible GT points newly within eps of the
                               candidate's frame-(t+1) 10 % sample| / |visible| (t12c_oracle / s1_oracle_in rule: frozen
                               sensor render, the sim's "sample" seed of step t + 1, Coverage._near on the remaining set)
OutdoorGT (Macarons++) tstar : same renderer on the scene OBJ x 10 (s1_oracle_out.load_mesh), k_M of the E-M2 frame
                       gains : T7a snapshot / restore evaluator (m_gain_eval): for every legal key restore -> update_camera
                               + capture_image -> process_frame with the seeds of step t + 1 -> monotone new GT points
                               (t25_out_cov rule; PRIMARY) and the official MAGICIAN gain (secondary); the snapshot is
                               restored afterwards (sim.covered replaced by the restored copy, as the T6 oracle method).
"""
import hashlib
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_common as CC  # noqa: E402

sys.path.insert(0, os.path.join(CC.EXP5, "task1", "scripts"))
sys.path.insert(0, os.path.join(CC.EXP5, "task2", "scripts"))
import t11_labels as TL  # noqa: E402  (pytorch3d rasteriser of the scoring rays; magician env)
import t2_common as C2  # noqa: E402


def sha(*arrays):
    return CC.sha_arrays(*arrays)


# ---------------------------------------------------------------------------------------------------- MP3D
class IndoorGT:
    def __init__(self, S, L, sen, cov, card, v, f, dev):
        from pytorch3d.structures import Meshes
        self.S, self.L, self.sen, self.cov, self.card, self.dev = S, L, sen, cov, card, dev
        mesh = Meshes(verts=[torch.as_tensor(C2.zup_m_to_rt(v), dtype=torch.float32, device=dev)],
                      faces=[torch.as_tensor(f, dtype=torch.int64, device=dev)])
        self.ren = TL.Renderer(mesh, dev, 0, None, 8)
        self.k_M = float(card["k_I"])
        self.nvis = max(1, int(cov.visible.sum()))
        self.P_xy = S.rt_to_zup_m(cov.pts)[:, :2]

    def state_hash(self):
        return sha(self.cov.covered, np.asarray(self.cov.value()))

    def tstar(self, legal_R, legal_T):
        n = len(legal_R)
        if n == 0:
            return np.zeros((0, 7296), np.float32)
        return self.ren.tstar(list(legal_R), list(legal_T), [(0.0, 0.0)] * n, self.k_M).astype(np.float32)

    def gains(self, legal_R, legal_T, start, t):
        """True C_obs gains of the candidates at state t (seed of frame t + 1); also the candidate samples (for the
        executed-candidate check) and the per-candidate new-point masks (for the region diagnostic)."""
        rem = self.cov.visible & ~self.cov.covered
        seed = self.S.derived_seed(self.L.key, start, t + 1, "sample")
        gains, pts_all, news = [], [], []
        for R, T in zip(legal_R, legal_T):
            fr = self.sen.render(torch.tensor(np.asarray(R)[None], dtype=torch.float32, device=self.dev),
                                 torch.tensor(np.asarray(T)[None], dtype=torch.float32, device=self.dev))
            pts = self.sen.sample(fr, seed)[0].cpu().numpy()
            new = self.cov._near(self.cov.tree, pts) & rem
            gains.append(float(new.sum() / self.nvis))
            pts_all.append(pts.astype(np.float32))
            news.append(new)
        return np.asarray(gains, np.float64), pts_all, news, rem

    def region_diag(self, center_rt, rem, news):
        """T0.12-C split with the Task 2 region (n_win = 5 around the refresh camera): share of the remaining visible
        GT points inside the planning region and the in-region part of every candidate's gain."""
        lo, hi = CC.region_box_xy(self.card, center_rt, CC.N_WIN)
        inside = np.all((self.P_xy >= lo) & (self.P_xy <= hi), axis=1)
        return {"remaining": float(rem.sum() / self.nvis), "remaining_in_region_share": float((rem & inside).sum() / max(1, rem.sum())),
                "gain_in_region": [float((n & inside).sum() / self.nvis) for n in news], "region_xy": [lo.tolist(), hi.tolist()]}


# ---------------------------------------------------------------------------------------------------- Macarons++
class OutdoorGT:
    def __init__(self, sim, scene, dev):
        import glob
        import exp3b_paths as P
        import m_gain_eval as GE
        from pytorch3d.io import load_obj
        from pytorch3d.structures import Meshes
        assert scene in CC.SPLIT_OF, f"GT GUARD: {scene} is not a Dev-3 / Test-3 evaluation scene"
        objs = sorted(glob.glob(os.path.join(P.scene_dir(scene), "*.obj")))
        assert len(objs) == 1, objs
        verts, faces, _ = load_obj(objs[0], load_textures=False, device=dev)
        self.ren = TL.Renderer(Meshes(verts=[verts * 10.0], faces=[faces.verts_idx]), dev, 0, None, 8)
        self.k_M = float(TL.frame_of(scene).k_M)
        self.sim, self.GE, self.dev = sim, GE, dev

    def state_hash(self):
        s = self.sim
        c = s.gt_scene.scene_coverage(s.covered, surface_epsilon=s.eps)[0]
        cov_off = c.item() if hasattr(c, "item") else float(c)          # MAGICIAN returns a plain 0.0 when nothing is covered
        return hashlib.sha256((s.key() + f"|{len(s.camera.cam_idx_history)}|{cov_off!r}|").encode() + s.covered_mono.tobytes()).hexdigest()

    def tstar(self, legal_R, legal_T):
        n = len(legal_R)
        if n == 0:
            return np.zeros((0, 7296), np.float32)
        return self.ren.tstar(list(legal_R), list(legal_T), [(0.0, 0.0)] * n, self.k_M).astype(np.float32)

    def gains(self, t, keys, cov_t):
        """(gain_mono, gain_official, cov_official_after, sample points) of every legal key at state t."""
        s, GE = self.sim, self.GE
        snap = GE.snapshot(s.camera, s.covered)
        g_mono, g_off, cov_as, pts = [], [], [], []
        for key in keys:
            cs = GE.restore(s.camera, snap)
            s.camera.update_camera(s.camera.get_idx_from_key(key), interpolation_step=1)
            s.camera.capture_image(s.mesh)
            cov_a, _, _, _, _, pc = GE.process_frame(s.camera, s.params, s.dev, s.s, s.k, t + 1, cs, s.gt_scene, s.eps, s.vis)
            p = pc.double().cpu().numpy()
            g_off.append(float(cov_a - cov_t))
            cov_as.append(float(cov_a))
            g_mono.append(float(s.mono_new(p).sum()) / len(s.G) / s.vis)
            pts.append(p.astype(np.float32))
            del cs
        s.covered = GE.restore(s.camera, snap)
        return np.asarray(g_mono, np.float64), np.asarray(g_off, np.float64), cov_as, pts

    def reach_diag(self, legal_center):
        """Share of the remaining (monotone-uncovered) GT points within the sensor range of some legal candidate centre."""
        s = self.sim
        rem = ~s.covered_mono
        n_rem = int(rem.sum())
        if n_rem == 0 or len(legal_center) == 0:
            return {"remaining": float(n_rem / len(s.G) / s.vis), "remaining_reachable_share": None}
        P = s.G[rem]
        ok = np.zeros(len(P), bool)
        for c in np.asarray(legal_center, np.float64).reshape(-1, 3):
            ok |= np.linalg.norm(P - c[None], axis=1) < 70.0
        return {"remaining": float(n_rem / len(s.G) / s.vis), "remaining_reachable_share": float(ok.mean())}
