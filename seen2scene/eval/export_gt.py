"""Export per-patch GT for evaluation (exp1-task2).

For every row of a patch CSV (assets/splits/front3d_heldout/{dev,test}_patches.csv),
writes `<out_dir>/<scene>_<bbox2str>/`:
- volume.npy     dense 256^3 float32 TSDF of the p_1.0 high-coverage target
                 (`TSDFPatch.to_dense()`, unknown = -trunc). Same format the
                 official `eval/compute_iou_l1_l2_tmd.py` expects.
- mesh.ply       complete scene mesh clipped to the patch (official `mesh_gt`, world
                 coordinates). The official CD is computed against this mesh.
- mesh_tsdf.ply  marching-cubes surface of the p_1.0 target (world coordinates).
                 Our known-region F-score uses this mesh.
- bbox.json      bbox, scene_name, dataset, voxel_size, truncation, known_ratio.
- inputs/p_<x>.npz  (optional, --input-keys) partial-scan inputs: dense int8
                 structure (compressed) plus band ijk/values.

GT and meshes are produced by the official Dataset -> custom_collate_fn ->
batch_to_Field path, so formats and coordinates match what the official completion
test exports. Partial inputs are cropped directly from their VDBs with the same crop,
clamp and thresholds as `Dataset.get_patch_data`. They do NOT go through the Dataset,
because the Dataset returns None for a key with zero surface voxels and then silently
substitutes a random scene. That happens for 5/100 dev patches at p_0.01.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m seen2scene.eval.export_gt --split dev \
      --out-dir work_dirs/exp1_infra/task2_gt_eval/gt/dev \
      --input-keys tsdf_p_0.01 tsdf_p_0.05 tsdf_p_0.1 tsdf_p_0.2 tsdf_p_0.5
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from seen2scene.dataset.common import batch_to_Field, custom_collate_fn
from seen2scene.dataset.dataset import Dataset
from seen2scene.tools.common_utils import bbox2str
from seen2scene.tools.vdb_utils import crop_vdb, loadvdb

SPLITS_DIR = "assets/splits/front3d_heldout"
# Official data config (work_dirs/weights/official/.../generator/.../config.yaml, `data:`)
DATA_KW = dict(data_list=["3D-FRONT"], split="test", known_ratio=0.999, patch_shape=(256, 256, 256),
               augmentation=False, voxel_size=0.011, truncation=None, bottom_padding=3.0,
               ceiling_clip=2.2, category_exclude=("floor", "ceiling", "wall"),
               num_categories=150, use_semantic_mapping=False)


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_device(v, device) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--input-keys", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scenes", nargs="*", default=None, help="only export these scenes (default: all)")
    args = ap.parse_args()

    scenes = tuple(open(os.path.join(SPLITS_DIR, f"{args.split}.txt")).read().split())
    csv_path = os.path.join(SPLITS_DIR, f"{args.split}_patches.csv")
    ds = Dataset(**DATA_KW, data_keys=("tsdf_p_1.0", "mesh_gt"), scene_names=scenes, csv_path=csv_path)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    device = torch.device("cuda")
    t0 = time.time()
    for i in range(n):
        want_scene, want_bbox = ds.csv_rows[i][1], ds.csv_rows[i][2]
        if args.scenes is not None and want_scene not in args.scenes:
            continue
        sample = ds.get_sample(i)
        # Guard against the dataset's silent random replacement.
        assert sample["scene_names"] == want_scene, (i, sample["scene_names"], want_scene)
        batch = batch_to_Field(to_device(custom_collate_fn([sample]), device))
        gt, mesh_gt = batch["tsdf_p_1.0"], batch["mesh_gt"]
        bbox = gt.bbox_world[0].cpu().numpy()
        assert np.allclose(bbox, want_bbox.numpy(), atol=1e-5)
        out = os.path.join(args.out_dir, f"{want_scene}_{bbox2str(bbox)}")
        os.makedirs(out, exist_ok=True)

        np.save(os.path.join(out, "volume.npy"), gt.to_dense()[0, 0].cpu().numpy().astype(np.float32))
        mesh_gt.export_mesh()[0].export(os.path.join(out, "mesh.ply"))
        gt.export_mesh()[0].export(os.path.join(out, "mesh_tsdf.ply"))
        with open(os.path.join(out, "bbox.json"), "w") as f:
            json.dump({"bbox": [float(x) for x in bbox], "scene_name": want_scene, "dataset": "3D-FRONT",
                       "voxel_size": gt.voxel_size, "truncation": float(gt.truncation),
                       "known_ratio": 0.999, "source": "tsdf_p_1.0"}, f, indent=1)
        gt_path = ds.scene_lookup[("3D-FRONT", want_scene)][0]["tsdf_p_1.0_path"]
        tr, kr = ds.truncation, ds.known_ratio
        for k in args.input_keys:
            p = k.replace("tsdf_p_", "")
            grid = loadvdb(gt_path.replace("fusion_p_1.0_", f"fusion_p_{p}_"))[0]["tsdf"]
            vol = np.clip(crop_vdb(grid, bbox[:3].tolist(), [256] * 3, cell_center=True), -tr, tr)
            structure = np.full(vol.shape, 1, dtype=np.int8)  # UNKNOWN
            structure[vol >= tr * kr] = 0  # EMPTY
            structure[np.abs(vol) < tr * kr] = 2  # BAND
            ijk = np.argwhere(structure == 2)
            os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
            np.savez_compressed(os.path.join(out, "inputs", f"p_{p}.npz"), structure=structure,
                                band_ijk=ijk.astype(np.int16), band_values=vol[structure == 2].astype(np.float32))
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"{args.split}: {i + 1}/{n} patches exported, {time.time() - t0:.0f}s", flush=True)


def load_input_volume(npz_path: str, trunc: float = 0.033) -> np.ndarray:
    """Dense TSDF of an exported partial-scan input (same convention as to_dense)."""
    d = np.load(npz_path)
    vol = np.full(d["structure"].shape, trunc, dtype=np.float32)
    vol[d["structure"] == 1] = -trunc
    ijk = d["band_ijk"].astype(np.int64)
    vol[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = d["band_values"]
    return vol


if __name__ == "__main__":
    main()
