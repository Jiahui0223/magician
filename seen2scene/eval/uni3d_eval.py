"""Uni3D feature extraction and Fréchet Distance evaluation.

Model components are adapted from BAAI-Vision/Uni3D (MIT License). See
``THIRD_PARTY_NOTICES.md``.

Provides two main functions:
1. extract_features: point clouds -> Uni3D embeddings
2. compute_uni3d_fd: pred features + gt features -> Fréchet Distance score

Usage:
    conda activate uni3d
    python -m seen2scene.eval.uni3d_eval  # runs a demo with random point clouds
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from pathlib import Path

import timm
from easydict import EasyDict
from pointnet2_ops import pointnet2_utils


# ========== Inlined model components (avoids importing losses/data/open3d) ==========

def _fps(data, number):
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()
    return fps_data


def _square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def _knn_point(nsample, xyz, new_xyz):
    sqrdists = _square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class _Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        batch_size, num_points, _ = xyz.shape
        center = _fps(xyz, self.num_group)
        idx = _knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features


class _Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class _PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, args):
        super().__init__()
        self.trans_dim = args.pc_feat_dim
        self.embed_dim = args.embed_dim
        self.group_size = args.group_size
        self.num_group = args.num_group
        self.group_divider = _Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dim = args.pc_encoder_dim
        self.encoder = _Encoder(encoder_channel=self.encoder_dim)
        self.encoder2trans = nn.Linear(self.encoder_dim, self.trans_dim)
        self.trans2embed = nn.Linear(self.trans_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim),
        )
        self.patch_dropout = nn.Identity()
        self.visual = point_transformer

    def forward(self, pts, colors):
        _, center, features = self.group_divider(pts, colors)
        group_input_tokens = self.encoder(features)
        group_input_tokens = self.encoder2trans(group_input_tokens)
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = x + pos
        x = self.patch_dropout(x)
        x = self.visual.pos_drop(x)
        for blk in self.visual.blocks:
            x = blk(x)
        x = self.visual.norm(x[:, 0, :])
        x = self.visual.fc_norm(x)
        x = self.trans2embed(x)
        return x


class _Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc):
        xyz = pc[:, :, :3].contiguous()
        color = pc[:, :, 3:].contiguous()
        pc_feat = self.point_encoder(xyz, color)
        return pc_feat


# ========== Model loading ==========

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_ckpt_path() -> Path:
    """Resolve the checkpoint without requiring an external Uni3D checkout."""
    if ckpt_path := os.environ.get("UNI3D_CKPT"):
        return Path(ckpt_path).expanduser()

    local_path = PROJECT_ROOT / "checkpoints" / "uni3d-b.pt"
    if local_path.is_file():
        return local_path

    # Backward compatibility with the old external-checkout setup.
    if uni3d_root := os.environ.get("UNI3D_ROOT"):
        return Path(uni3d_root).expanduser() / "checkpoints" / "uni3d-b.pt"

    return local_path


def _load_model(
    ckpt_path: str = None,
    pc_model: str = "eva02_base_patch14_448",
    pc_feat_dim: int = 768,
    embed_dim: int = 1024,
    group_size: int = 64,
    num_group: int = 512,
    pc_encoder_dim: int = 512,
    device: str = "cuda",
):
    """Load Uni3D model from checkpoint."""
    if ckpt_path is None:
        ckpt_path = str(_default_ckpt_path())

    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(
            f"Uni3D checkpoint not found at {ckpt_path}. Pass --uni3d-ckpt, "
            "set UNI3D_CKPT, or place it at checkpoints/uni3d-b.pt."
        )

    args = EasyDict(
        pc_feat_dim=pc_feat_dim,
        embed_dim=embed_dim,
        group_size=group_size,
        num_group=num_group,
        pc_encoder_dim=pc_encoder_dim,
    )

    point_transformer = timm.create_model(pc_model, drop_path_rate=0.0)
    point_encoder = _PointcloudEncoder(point_transformer, args)
    model = _Uni3D(point_encoder=point_encoder)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = checkpoint.get("module", checkpoint)
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

    model = model.to(device).eval()
    print(f"[Uni3D] Loaded checkpoint from {ckpt_path}")
    print(f"[Uni3D] Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model


_MODEL_CACHE = {}


def _get_model(ckpt_path: str = None, device: str = "cuda"):
    """Get or load a cached Uni3D model."""
    cache_key = (ckpt_path, device)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = _load_model(ckpt_path=ckpt_path, device=device)
    return _MODEL_CACHE[cache_key]


# ========== Public API ==========


def extract_features(
    point_clouds: Union[np.ndarray, torch.Tensor, List[np.ndarray]],
    ckpt_path: str = None,
    device: str = "cuda",
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Extract Uni3D features from point clouds.

    Args:
        point_clouds: Point cloud data. Accepted formats:
            - np.ndarray of shape (N, num_points, 6) with [x, y, z, r, g, b]
            - np.ndarray of shape (N, num_points, 3) with [x, y, z] (zero RGB)
            - torch.Tensor of the same shapes
            - List of np.ndarray, each (num_points, 3) or (num_points, 6)
        ckpt_path: Path to Uni3D checkpoint. Defaults to checkpoints/uni3d-b.pt.
        device: Device to use for inference.
        batch_size: Batch size for inference.
        normalize: Whether to L2-normalize the output features.

    Returns:
        np.ndarray of shape (N, 1024): Uni3D embeddings for each point cloud.
    """
    model = _get_model(ckpt_path=ckpt_path, device=device)

    # Convert input to a single numpy array of shape (N, num_points, 6)
    if isinstance(point_clouds, list):
        processed = []
        for pc in point_clouds:
            if isinstance(pc, torch.Tensor):
                pc = pc.cpu().numpy()
            if pc.ndim == 2 and pc.shape[-1] == 3:
                pc = np.concatenate([pc, np.zeros_like(pc)], axis=-1)
            processed.append(pc)
        point_clouds = np.stack(processed, axis=0)
    elif isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()

    if point_clouds.ndim == 2:
        point_clouds = point_clouds[np.newaxis]

    if point_clouds.shape[-1] == 3:
        point_clouds = np.concatenate(
            [point_clouds, np.zeros_like(point_clouds)], axis=-1
        )

    assert point_clouds.ndim == 3 and point_clouds.shape[-1] == 6, (
        f"Expected (N, num_points, 6), got {point_clouds.shape}"
    )

    num_samples = point_clouds.shape[0]
    all_features = []

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch = torch.tensor(
                point_clouds[start:end], dtype=torch.float32, device=device
            )
            features = model.encode_pc(batch)
            if normalize:
                features = F.normalize(features, dim=-1)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def compute_uni3d_fd(
    pred_features: np.ndarray,
    gt_features: np.ndarray,
) -> float:
    """Compute the Fréchet Distance (FD) between two sets of Uni3D features.

    This is analogous to FID but in Uni3D's 3D-CLIP aligned feature space.

    FD = ||mu_pred - mu_gt||^2 + Tr(Sigma_pred + Sigma_gt - 2*sqrt(Sigma_pred @ Sigma_gt))

    Args:
        pred_features: np.ndarray of shape (N1, D), features from generated shapes.
        gt_features: np.ndarray of shape (N2, D), features from ground-truth shapes.

    Returns:
        float: The Fréchet Distance score (lower is better).
    """
    from scipy import linalg

    assert pred_features.ndim == 2 and gt_features.ndim == 2
    assert pred_features.shape[1] == gt_features.shape[1]

    mu_pred = np.mean(pred_features, axis=0)
    mu_gt = np.mean(gt_features, axis=0)
    sigma_pred = np.cov(pred_features, rowvar=False)
    sigma_gt = np.cov(gt_features, rowvar=False)

    diff = mu_pred - mu_gt
    covmean, _ = linalg.sqrtm(sigma_pred @ sigma_gt, disp=False)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"[Warning] Imaginary component {m:.4f} in sqrtm result")
        covmean = covmean.real

    fd = diff @ diff + np.trace(sigma_pred) + np.trace(sigma_gt) - 2 * np.trace(covmean)
    return float(fd)


def evaluate(
    pred_point_clouds: Union[np.ndarray, List[np.ndarray]],
    gt_point_clouds: Union[np.ndarray, List[np.ndarray]],
    ckpt_path: str = None,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """End-to-end evaluation: point clouds -> Uni3D-FD score.

    Args:
        pred_point_clouds: Generated point clouds, (N1, num_points, 3 or 6).
        gt_point_clouds: Ground-truth point clouds, (N2, num_points, 3 or 6).
        ckpt_path: Path to Uni3D checkpoint.
        device: Device for inference.
        batch_size: Batch size for feature extraction.

    Returns:
        dict with keys:
            - "uni3d_fd": The Fréchet Distance score.
            - "pred_features": np.ndarray (N1, 1024)
            - "gt_features": np.ndarray (N2, 1024)
    """
    print("[Uni3D] Extracting features for predictions...")
    pred_features = extract_features(
        pred_point_clouds, ckpt_path=ckpt_path, device=device, batch_size=batch_size
    )
    print(f"[Uni3D] Pred features shape: {pred_features.shape}")

    print("[Uni3D] Extracting features for ground truth...")
    gt_features = extract_features(
        gt_point_clouds, ckpt_path=ckpt_path, device=device, batch_size=batch_size
    )
    print(f"[Uni3D] GT features shape: {gt_features.shape}")

    fd = compute_uni3d_fd(pred_features, gt_features)
    print(f"[Uni3D] Fréchet Distance: {fd:.4f}")

    return {
        "uni3d_fd": fd,
        "pred_features": pred_features,
        "gt_features": gt_features,
    }


# ---- Demo / test ----
if __name__ == "__main__":
    print("=" * 60)
    print("Uni3D Evaluation Demo")
    print("=" * 60)

    num_pred, num_gt, num_points = 50, 50, 10000

    print(f"\nGenerating {num_pred} pred and {num_gt} GT random point clouds "
          f"with {num_points} points each...")

    np.random.seed(42)
    pred_pcs = np.random.randn(num_pred, num_points, 3).astype(np.float32)
    gt_pcs = np.random.randn(num_gt, num_points, 3).astype(np.float32) * 0.8 + 0.1

    results = evaluate(pred_pcs, gt_pcs)
    print(f"\nUni3D-FD Score: {results['uni3d_fd']:.4f}")
    print("(Lower is better, 0 means identical distributions)")
