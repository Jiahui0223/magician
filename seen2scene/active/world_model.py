"""In-process world model for closed-loop NBV: the official S0-F completion, bit for bit (exp2-task5).

`WorldModel.generate(scene, volume, episode, step, ks)` returns the dense TSDF worlds that
`python -m seen2scene.main control task:completion --task.drop-bbox --task.world-seeding
--task.num-inference-steps S ...` would produce for the same partial observation and seeds, because every
step reuses the official code:
- config: built with the same tyro CLI as experiments/exp2_baseline/run_completion.py;
- setup as in main.py under --task.world-seeding: CUBLAS_WORKSPACE_CONFIG + deterministic algorithms (before any
  CUDA call), seed_everything(args.seed), the network built inside the CUDA device context (as
  Trainer.init_module), eval mode, the dict-held VAE on the device (as Net.setup);
- input: dense TSDF -> clamp to +-truncation (as Dataset._load_cropped) -> Dataset.get_patch_data (structure,
  band) with the patch parameters from Dataset.get_patch_params and the frozen CSV bbox (read by the Dataset,
  float32) -> custom_collate_fn -> device -> batch_to_Field;
- generation: CLIP name cache cleared (as test_step), world_rng(seed_world(scene, episode, step, k)),
  torch.inference_mode + torch.autocast(cuda, bfloat16, cache_enabled=False) (as Lightning bf16-mixed),
  Net.generate_patch, field.to_dense().
The official test path uses episode=0, step=0; closed-loop NBV uses (episode, step) of the episode.
k is 0-based: S1 uses k=0, the first world of S2's k=0..K-1.
"""
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

EXP2_CKPT = ["--ae-log", "2025-12-19_01-23-28-525", "--gen-log", "2026-02-23_16-22-25-152",
             "--ckpt-path", "2026-02-26_14-01-23-930"]
SPLITS = "assets/splits/front3d_heldout"


def enable_deterministic():
    """Same as main.py under --task.world-seeding. Call before the first CUDA/cuBLAS use in the process."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


class EmptyObservation(RuntimeError):
    """The observation has no surface voxel; the official pipeline cannot build its input grid."""


class WorldModel:
    def __init__(self, split: str = "dev", steps: int = 10, src_key: str = "tsdf_p_0.1",
                 run_root: str = "work_dirs/exp2_baseline/world_model_runs", device: str = "cuda"):
        enable_deterministic()
        import tyro
        from pytorch_lightning import seed_everything

        from seen2scene.configs.opt import VAE, Control, Generator
        from seen2scene.models.trainer.train_gen import _modules_incl_dicts
        from seen2scene.tools.common_utils import get_obj_from_str
        from seen2scene.tools.lightning import DataModuleFromConfig

        scenes = open(os.path.join(SPLITS, f"{split}.txt")).read().split()
        argv = ["control", "task:completion", "--run-root", run_root, *EXP2_CKPT, "--src-key", src_key,
                "--latent-key", "tsdf_p_1.0", "--precision", "bf16-mixed", "--task.drop-bbox", "--task.world-seeding",
                "--task.num-inference-steps", str(steps), "--data.data-list", "3D-FRONT",
                "--data.scene-names", *scenes, "--data.csv-path", os.path.join(SPLITS, f"{split}_patches.csv"),
                "--seed", "0", "--slurm.cluster", "local"]
        self.args = tyro.cli(VAE | Generator | Control, args=argv, config=(tyro.conf.CascadeSubcommandArgs,))
        self.device = torch.device(device)
        seed_everything(self.args.seed)
        with torch.device(self.device):  # as Trainer.init_module (single device)
            self.net = get_obj_from_str(self.args.target)(self.args)
        self.net.to(self.device)
        self.net.eval()
        for key in self.net.vae.keys():  # as Net.setup
            self.net.vae[key].to(self.device)
        for m in _modules_incl_dicts(self.net):  # as test_step with --task.num-inference-steps
            if hasattr(m, "num_inference_steps") and not isinstance(m, type(self.net)):
                m.num_inference_steps = int(steps)
        self._conditioners = [m for m in _modules_incl_dicts(self.net)
                              if isinstance(getattr(m, "_embedding_cache", None), dict)]
        dm = DataModuleFromConfig(data_config=self.args.data, num_workers=0, splits=["test"],
                                  test_data_keys=self.args.test_data_keys, test_split=self.args.test_split)
        self.ds = dm.datasets["test"]
        self.src_key, self.latent_key = self.net.src_key, self.net.latent_key
        self._rows = {row[1]: row for row in self.ds.csv_rows}
        self.steps = steps

    # ------------------------------------------------------------------ input
    def patch_inputs(self, scene: str):
        """(file_paths, scene_meta, bbox_world, chunk_size, field_params) exactly as Dataset.get_sample (CSV)."""
        dataset_name, scene_name, bbox_world = self._rows[scene]
        file_paths, scene_meta = self.ds.scene_lookup[(dataset_name, scene_name)]
        scene_size = scene_meta["scene_bounds"][3:] - scene_meta["scene_bounds"][:3]
        chunk_size = torch.tensor([scene_size[i] if a is None else a for i, a in enumerate(self.ds.chunk_size)])
        field_params = self.ds.get_patch_params(scene_meta, bbox_world, chunk_size, None)
        return file_paths, scene_meta, bbox_world, chunk_size, field_params

    def official_input_volume(self, scene: str, key: Optional[str] = None) -> torch.Tensor:
        """The clamped crop the official Dataset reads for `key` (default: the source key)."""
        file_paths, _, bbox_world, chunk_size, _ = self.patch_inputs(scene)
        key = key or self.src_key
        patch_shape = torch.round(chunk_size / self.ds.voxel_size).int().tolist()
        return self.ds._load_cropped(file_paths[f"{key}_path"], key, bbox_world.tolist()[:3], patch_shape)

    def make_batch(self, scene: str, volume) -> Dict:
        """Official batch for a dense observation volume ([256]^3, TSDF, unobserved = -truncation)."""
        from seen2scene.dataset.common import batch_to_Field, custom_collate_fn

        file_paths, scene_meta, bbox_world, chunk_size, field_params = self.patch_inputs(scene)
        vol = torch.as_tensor(volume).detach().to("cpu", torch.float32)
        vol = torch.clamp(vol, min=-self.ds.truncation, max=self.ds.truncation)  # as Dataset._load_cropped
        loader = self.ds._load_cropped
        self.ds._load_cropped = lambda *a, **k: vol  # reuse Dataset.get_patch_data for structure/band
        try:
            data = self.ds.get_patch_data(file_paths, self.src_key, bbox_world, chunk_size, None)
        finally:
            self.ds._load_cropped = loader
        if data is None:
            raise EmptyObservation(scene)
        sample = {self.src_key: data, "scene_names": scene_meta["scene_name"], "field_params": field_params,
                  "keep_region": scene_meta["keep_region"]}
        batch = custom_collate_fn([sample])
        batch = _to_device(batch, self.device)
        batch = batch_to_Field(batch)
        batch[self.latent_key] = batch[self.src_key]  # generate_patch only reads the patch kwargs from it
        return batch

    # ------------------------------------------------------------------ generation
    @torch.inference_mode()
    def generate(self, scene: str, volume, episode: int, step: int, ks: Sequence[int] = (0,),
                 return_fields: bool = False) -> torch.Tensor:
        """Dense worlds [len(ks), 256, 256, 256] (float32, on the device), one per seed_world(scene, episode, step, k)."""
        from seen2scene.tools.seeding import seed_world, world_rng

        batch = self.make_batch(scene, volume)
        for m in self._conditioners:  # as test_step under --task.world-seeding
            m._embedding_cache.clear()
        worlds, fields = [], []
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            for k in ks:
                with world_rng(seed_world(scene, episode, step, k)):
                    field = self.net.generate_patch(batch)
                worlds.append(field.to_dense()[0, 0].float())
                if return_fields:
                    fields.append(field)
        out = torch.stack(worlds)
        return (out, fields) if return_fields else out


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    return x
