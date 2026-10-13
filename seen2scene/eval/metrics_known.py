"""Evaluation restricted to the known region of the high-coverage target (exp1-task2).

All metrics use the dense 256^3 TSDF convention of `TSDFPatch.to_dense()`:
BAND |v| < trunc * known_ratio, EMPTY v >= trunc * known_ratio, UNKNOWN otherwise
(unknown voxels are stored as -trunc). "GT" below is the p_1.0 high-coverage
target, not a complete watertight scene.

Regions (all subsets of the GT-known voxels):
- known:      GT != UNKNOWN
- observed:   known and the input partial scan != UNKNOWN
- unobserved: known and the input partial scan == UNKNOWN

Frozen NBV coverage definitions (used by every planner and every experiment):
- surface coverage: fraction of GT BAND voxels that have an observed BAND voxel
  within 1 voxel (3x3x3 neighbourhood);
- free-space coverage: fraction of GT EMPTY voxels that are observed (not UNKNOWN).
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import trimesh
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

EMPTY, UNKNOWN, BAND = 0, 1, 2  # same values as seen2scene.configs.dataset.Voxel
VOXEL_SIZE = 0.011
TRUNCATION = 3 * VOXEL_SIZE
KNOWN_RATIO = 0.999


def structure_from_volume(vol: np.ndarray, trunc: float = TRUNCATION, known_ratio: float = KNOWN_RATIO) -> np.ndarray:
    """Three-state structure with the official thresholds (dataset.py:605-613)."""
    t = trunc * known_ratio
    s = np.full(vol.shape, UNKNOWN, dtype=np.int8)
    s[vol >= t] = EMPTY
    s[np.abs(vol) < t] = BAND
    return s


def region_masks(gt_struct: np.ndarray, input_struct: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    known = gt_struct != UNKNOWN
    masks = {"known": known}
    if input_struct is not None:
        observed = input_struct != UNKNOWN
        masks["observed"] = known & observed
        masks["unobserved"] = known & ~observed
    return masks


def voxel_metrics(pred_vol: np.ndarray, gt_vol: np.ndarray, masks: Dict[str, np.ndarray],
                  trunc: float = TRUNCATION, known_ratio: float = KNOWN_RATIO) -> Dict[str, float]:
    """Surface-band IoU and TSDF L1 (metres) inside each region."""
    t = trunc * known_ratio
    pred_band = np.abs(pred_vol) < t
    gt_band = np.abs(gt_vol) < t
    diff = np.abs(np.clip(pred_vol, -trunc, trunc) - np.clip(gt_vol, -trunc, trunc))
    out = {}
    for name, m in masks.items():
        inter = np.count_nonzero(pred_band & gt_band & m)
        union = np.count_nonzero((pred_band | gt_band) & m)
        out[f"{name}/band_iou"] = inter / union if union > 0 else float("nan")
        out[f"{name}/tsdf_l1"] = float(diff[m].mean()) if m.any() else float("nan")
        out[f"{name}/n_voxels"] = int(m.sum())
    return out


def _points_in_mask(points: np.ndarray, mask: np.ndarray, bbox_min: np.ndarray, voxel_size: float) -> np.ndarray:
    ijk = np.floor((points - bbox_min) / voxel_size).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.array(mask.shape)), axis=1)
    keep = np.zeros(len(points), dtype=bool)
    keep[inside] = mask[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
    return keep


def sample_points(mesh: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    if mesh is None or len(mesh.faces) == 0:
        return np.zeros((0, 3))
    return trimesh.sample.sample_surface(mesh, n, seed=seed)[0]


def mesh_metrics(pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, bbox_min: Sequence[float],
                 masks: Dict[str, np.ndarray], n_points: int = 300_000, n_ref: int = 2_000_000, seed: int = 0,
                 taus_vox: Sequence[int] = (1, 2), voxel_size: float = VOXEL_SIZE,
                 pred_points: Optional[np.ndarray] = None, gt_points: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Chamfer-L1 (mm) and F-score at 1/2 voxels, per region.

    Query points (n_points per mesh) are matched against a denser reference sample
    (n_ref per mesh). This keeps the point-sampling noise floor well below one voxel;
    with equal-size sets the floor alone is F@1 ~ 0.98 / ~4.7 mm on these patches.
    Accuracy uses predicted query points inside the region; completeness uses GT query
    points inside the region. Distances are measured to the full reference set, so
    region borders do not create artificial errors. Seeds are fixed, so results are
    reproducible. If pred_points/gt_points are given, they serve as both query and reference.
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    if pred_points is None:
        pa, pa_ref = sample_points(pred_mesh, n_points, seed), sample_points(pred_mesh, n_ref, seed + 2)
    else:
        pa = pa_ref = pred_points
    if gt_points is None:
        pb, pb_ref = sample_points(gt_mesh, n_points, seed + 1), sample_points(gt_mesh, n_ref, seed + 3)
    else:
        pb = pb_ref = gt_points
    out = {}
    if len(pa) == 0 or len(pb) == 0:
        return {f"{r}/{k}": float("nan") for r in masks for k in ["chamfer_l1_mm"] + [f"F@{t}" for t in taus_vox]}
    d_ab = cKDTree(pb_ref).query(pa, workers=-1)[0]  # pred -> gt (accuracy)
    d_ba = cKDTree(pa_ref).query(pb, workers=-1)[0]  # gt -> pred (completeness)
    for name, m in masks.items():
        ka = _points_in_mask(pa, m, bbox_min, voxel_size)
        kb = _points_in_mask(pb, m, bbox_min, voxel_size)
        if ka.sum() == 0 or kb.sum() == 0:
            out[f"{name}/chamfer_l1_mm"] = float("nan")
            for t in taus_vox:
                out[f"{name}/F@{t}"] = float("nan")
            continue
        acc, comp = d_ab[ka], d_ba[kb]
        out[f"{name}/chamfer_l1_mm"] = 1000 * (acc.mean() + comp.mean()) / 2
        for t in taus_vox:
            tau = t * voxel_size
            p, r = (acc < tau).mean(), (comp < tau).mean()
            out[f"{name}/F@{t}"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return out


def pairwise_chamfer_l1_mm(meshes: List[trimesh.Trimesh], n_points: int = 100_000, seed: int = 0) -> float:
    """Diversity: mean symmetric Chamfer-L1 (mm) over all pairs of samples."""
    pts = [sample_points(m, n_points, seed + i) for i, m in enumerate(meshes)]
    vals = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if len(pts[i]) and len(pts[j]):
                a = cKDTree(pts[j]).query(pts[i], workers=-1)[0].mean()
                b = cKDTree(pts[i]).query(pts[j], workers=-1)[0].mean()
                vals.append(1000 * (a + b) / 2)
    return float(np.mean(vals)) if vals else float("nan")


def observation_violation(pred_vol: np.ndarray, input_vol: np.ndarray, trunc: float = TRUNCATION,
                          known_ratio: float = KNOWN_RATIO, tol_vox: float = 1.0,
                          voxel_size: float = VOXEL_SIZE) -> Dict[str, float]:
    """How much a sample contradicts the observation it was conditioned on.

    - free_violation: fraction of observed-EMPTY voxels where the sample puts a surface band;
    - surface_violation: fraction of observed-BAND voxels where |pred - obs| > tol_vox voxels.
    """
    t = trunc * known_ratio
    inp = structure_from_volume(input_vol, trunc, known_ratio)
    pred_band = np.abs(pred_vol) < t
    obs_empty, obs_band = inp == EMPTY, inp == BAND
    free_v = pred_band[obs_empty].mean() if obs_empty.any() else float("nan")
    dev = np.abs(np.clip(pred_vol, -trunc, trunc) - input_vol)
    surf_v = (dev[obs_band] > tol_vox * voxel_size).mean() if obs_band.any() else float("nan")
    return {"free_violation": float(free_v), "surface_violation": float(surf_v)}


def surface_coverage(observed_struct: np.ndarray, gt_struct: np.ndarray) -> Dict[str, float]:
    """Frozen NBV evaluation coverage (see module docstring)."""
    gt_band = gt_struct == BAND
    obs_band_dil = maximum_filter((observed_struct == BAND).astype(np.uint8), size=3) > 0
    gt_empty = gt_struct == EMPTY
    return {
        "surface_coverage": float((gt_band & obs_band_dil).sum() / max(gt_band.sum(), 1)),
        "freespace_coverage": float((gt_empty & (observed_struct != UNKNOWN)).sum() / max(gt_empty.sum(), 1)),
    }
