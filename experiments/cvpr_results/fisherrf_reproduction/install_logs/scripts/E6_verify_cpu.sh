#!/bin/bash
# CPU-only import checks (no GPU is touched: CUDA_VISIBLE_DEVICES="" from common.sh)
source "$(dirname "$0")/common.sh"
export PYTHONDONTWRITEBYTECODE=1
echo "[start] $(date -Is) host=$(hostname) CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
conda activate "$ENV" || exit 1
P="$ENV/bin/python"
echo "== 1. extensions + torch"
"$P" - <<'PY'
import os, sys, glob, subprocess
import torch, torchvision
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda, "| torchvision", torchvision.__version__,
      "| cuda.is_available()", torch.cuda.is_available(), "(expected False: CUDA_VISIBLE_DEVICES empty)")
import diff_gaussian_rasterization as dgr
from diff_gaussian_rasterization import _C as dgr_C, GaussianRasterizer, GaussianRasterizationSettings
import inspect
print("diff_gaussian_rasterization:", dgr.__file__)
print("  _C:", dgr_C.__file__, "| symbols:", sorted(n for n in dir(dgr_C) if not n.startswith("_")))
print("  GaussianRasterizer(raster_settings, backward_power=...):", "backward_power" in inspect.signature(GaussianRasterizer.__init__).parameters)
import simple_knn
from simple_knn import _C as sk_C
print("simple_knn._C:", sk_C.__file__, "| symbols:", sorted(n for n in dir(sk_C) if not n.startswith("_")))
PY
echo "== 2. sm arch + glibc of compiled .so"
source /etc/profile.d/modules.sh; module load cuda11.8/toolkit/11.8.0
for so in $(ls "$ENV"/lib/python3.9/site-packages/diff_gaussian_rasterization/_C*.so "$ENV"/lib/python3.9/site-packages/simple_knn/_C*.so); do
  echo "$so"; echo "  archs: $(cuobjdump --list-elf "$so" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
  echo "  max GLIBC: $(objdump -T "$so" | grep -o 'GLIBC_[0-9.]*' | sort -V -u | tail -1)  max GLIBCXX: $(objdump -T "$so" | grep -o 'GLIBCXX_[0-9.]*' | sort -V -u | tail -1)"
  echo "  ldd not found: $(ldd "$so" | grep -c 'not found')"
done
echo "== 3. habitat / other deps"
"$P" - <<'PY'
import habitat_sim, habitat, habitat_baselines
print("habitat_sim", habitat_sim.__version__, "| habitat", habitat.__version__, "| habitat_baselines", getattr(habitat_baselines, "__version__", "?"))
for m in ["gym", "hydra", "omegaconf", "cv2", "yacs", "einops", "kornia", "torchmetrics", "tensorboardX", "wandb", "rich", "sklearn",
          "matplotlib", "IPython", "yaml", "pytorch_msssim", "lpips", "plyfile", "quaternion", "numba", "scipy", "imageio", "numpy"]:
    mod = __import__(m)
    print(f"  {m:16s} {getattr(mod, '__version__', '?')}")
try:
    import open3d
    print("  open3d", open3d.__version__)
except Exception as e:
    print("  open3d: IMPORT FAILED ->", type(e).__name__, e)
PY
echo "== 4. official FisherRF modules (from the untouched clone, no bytecode written)"
cd "$AM"
"$P" - <<'PY'
import sys, os, traceback
sys.path.insert(0, os.environ["AM"])
for name in ["models.SLAM.utils.recon_helpers", "models.SLAM.utils.slam_external", "models.SLAM.utils.slam_helpers",
             "models.SLAM.utils.eval_helpers", "models.SLAM.gaussian", "planning.astar", "tester"]:
    try:
        __import__(name)
        print("  import OK  ", name)
    except Exception as e:
        print("  import FAIL", name, "->", type(e).__name__, str(e)[:300])
PY
echo "[end] $(date -Is)"
