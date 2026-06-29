# MAGICIAN 环境安装记录（jiahui）

> 本文件由 Claude 在 2026-06-14 配置环境时生成，记录**所有关键路径、遇到的问题、以及实际执行的命令**，方便复现和排查。

---

## 0. 一句话总结

- **环境位置**：`/bsuscratch/jiahuizhang/envs/magician`（conda，Python 3.9.23）
- **激活方式**：`source activate_magician.sh`（在仓库根目录）
- **运行**：`python test_magician_planning.py`
- 已在 `gpu101`（V100）和 `gpu106`（P100）上**端到端验证通过**（真实跑起 beam search 规划）。
- 直接照搬 `environment.yml` **装不上**，原因和修法见第 2 节。
- ⚠️ **重要**：你的终端运行在 **glibc 2.36 的容器里**，但 host 是 glibc 2.17。这个分裂是你"一直出错"的根因 —— **详见第 8 节（最关键，建议先看）**。env 已编译成 host/容器**两边通用**（glibc 2.17 + sm_60/sm_70）。

---

## 1. 系统环境（HPC 现状）

| 项目 | 值 |
|---|---|
| 集群节点 | `gpu101`（GPU 计算节点，2× Tesla V100-PCIE-16GB，cc 7.0 / sm_70） |
| 操作系统 | CentOS 7（内核 el7），**glibc 2.17** ← 关键约束 |
| CPU / 内存 | 48 核 / 376 GB |
| conda | 23.1.0，base 在 `/cm/shared/apps/conda/23.1.0`（只读，不能改） |
| conda 环境目录 | `/bsuscratch/jiahuizhang/envs`（用户默认 envs_dirs） |
| CUDA 模块 | `module load cuda11.8/toolkit/11.8.0`（nvcc 11.8.89，**正好匹配 torch cu118**） |
| gcc | 9.2.0（默认在 PATH 上） |
| 网络 | 计算节点**可联网**（github / pypi / conda / pytorch / huggingface / google drive 都通） |

诊断时用到的命令：
```bash
hostname; nproc; free -h; nvidia-smi
conda --version; conda env list; conda config --show solver channels envs_dirs
source /etc/profile.d/modules.sh; module avail cuda; module avail gcc
ldd --version                      # → glibc 2.17
conda info | grep glibc            # → __glibc=2.17=0（虚拟包）
```

---

## 2. 为什么 environment.yml 装不上（你之前一直报错的根源）

`environment.yml` 是在一台 **glibc ≥ 2.28** 的机器上导出的，但本集群是 **glibc 2.17**：

1. **conda 部分**：写死了每个包的 build hash（如 `zstd=1.5.7=h11fc155_0`、`libgcc=15.2.0`），这些 build 要求 `__glibc>=2.28` → `conda env create` 直接 `UnsatisfiableError`。
2. **pip 部分**：`torch==2.7.1+cu118` 的 wheel 是 `manylinux_2_28` 标签（要 glibc 2.28）→ pip 报 "No matching distribution"。
3. `pandas==2.3.3` 没有 Python 3.9（cp39）的预编译 wheel。

**修法**：不照搬 yml，改为
- 最小 conda 环境（`python=3.9 pip`，**不锁 build** → conda 自动选 glibc 2.17 兼容版）
- 其余用 pip 装（torch 降到 cu118 在 glibc 2.17 上的最高版 **2.6.0**，pandas 降到 2.2.3）

**glibc 2.17 上的版本天花板**：torch 最高 `2.6.0+cu118`（2.7+ 的 cu118 wheel 全是 manylinux_2_28）。

---

## 3. 实际执行的完整步骤与命令

### 步骤 1 — 初始化子模块

```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN
git submodule update --init --recursive
```
RaDe-GS、glm、diff-gaussian-rasterization、tetra_triangulation 都成功；**只有 `simple-knn` 失败**
（`gitlab.inria.fr/bkerbl/simple-knn.git` 的 git 协议在本集群拉不动）。

**simple-knn 替代方案**：从官方 INRIA gitlab 下载锁定 commit 的归档 tarball（可信来源 + 精确版本）：
```bash
cd RaDe-GS/submodules
curl -sL "https://gitlab.inria.fr/bkerbl/simple-knn/-/archive/44f764299fa305faf6ec5ebd99939e0508331503/simple-knn-44f764299fa305faf6ec5ebd99939e0508331503.tar.gz" -o /tmp/simple-knn.tar.gz
tar -xzf /tmp/simple-knn.tar.gz -C /tmp/
mkdir -p simple-knn && cp -r /tmp/simple-knn-44f7642*/. simple-knn/
```

### 步骤 2 — 创建 conda 环境（最小化）

```bash
# ❌ 这一步会失败（仅作记录）：conda env create -f environment.yml
# ✅ 实际用：
conda create -y -p /bsuscratch/jiahuizhang/envs/magician -c conda-forge python=3.9 pip setuptools wheel
/bsuscratch/jiahuizhang/envs/magician/bin/pip install --upgrade pip   # → pip 26.0.1
```

### 步骤 3 — 安装 torch + 核心依赖（pip）

脚本：`scripts/install_pip.sh`（依赖清单 `scripts/reqs_core.txt`）
```bash
PIP=/bsuscratch/jiahuizhang/envs/magician/bin/pip
# torch 栈（cu118 在 glibc 2.17 的上限 = 2.6.0）
$PIP install --prefer-binary torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
# 95 个核心包（从 environment.yml 的 pip 段提取，去掉 torch/nvidia-*/pyembree，pandas 改 2.2.3）
$PIP install --prefer-binary -r reqs_core.txt
# torch 要求 sympy==1.13.1
$PIP install "sympy==1.13.1"
```

### 步骤 4 — 编译 CUDA 扩展（diff_gaussian_rasterization + simple_knn）

脚本：`scripts/build_ext.sh`
```bash
source /etc/profile.d/modules.sh
module load cuda11.8/toolkit/11.8.0
export CUDA_HOME=/cm/shared/apps/cuda11.8/toolkit/11.8.0
export TORCH_CUDA_ARCH_LIST="7.0"   # V100
export MAX_JOBS=8 FORCE_CUDA=1
PIP=/bsuscratch/jiahuizhang/envs/magician/bin/pip
$PIP install --no-build-isolation RaDe-GS/submodules/diff-gaussian-rasterization
$PIP install --no-build-isolation RaDe-GS/submodules/simple-knn
```

### 步骤 5 — 编译 pytorch3d（源码，最耗时 ~20 分钟）

脚本：`scripts/build_pytorch3d.sh`
```bash
module load cuda11.8/toolkit/11.8.0
export CUDA_HOME=/cm/shared/apps/cuda11.8/toolkit/11.8.0
export TORCH_CUDA_ARCH_LIST="7.0" FORCE_CUDA=1 MAX_JOBS=12
PIP=/bsuscratch/jiahuizhang/envs/magician/bin/pip
$PIP install --prefer-binary fvcore
$PIP install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
```

### 步骤 6 — 下载数据集 + 权重

脚本：`scripts/download_data.sh`
```bash
# 数据集（HuggingFace，单个 Macarons++.zip ~1.2GB）
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download(repo_id='sli016/Macarons-plus-plus', filename='Macarons++.zip', repo_type='dataset', local_dir='./data')"
unzip -o -q ./data/Macarons++.zip -d ./data
mv data/macarons++ data/Macarons++        # ⚠️ 解压出来是小写，必须改成大写以匹配 config
rm -f data/Macarons++.zip

# 权重（Google Drive 文件夹，需要 gdown）
/bsuscratch/jiahuizhang/envs/magician/bin/pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1wyc9_QFmcxOz4oerE8kCQ3I8LO5zioZL" -O ./weights
```

### 步骤 7 — 修复 rtree（缺 libspatialindex）

运行时 trimesh 做射线-网格相交（碰撞检测）会报 `OSError: Could not load libspatialindex_c library`，
因为 pip 装的 rtree 不自带原生库，且 rtree 在 Linux 上不搜 `$CONDA_PREFIX/lib`。
```bash
conda install -y -p /bsuscratch/jiahuizhang/envs/magician -c conda-forge libspatialindex
# 用 conda activate.d 钩子持久化环境变量（每次 conda activate 自动设）
mkdir -p /bsuscratch/jiahuizhang/envs/magician/etc/conda/activate.d
cat > /bsuscratch/jiahuizhang/envs/magician/etc/conda/activate.d/zz_magician_env.sh <<'EOF'
export SPATIALINDEX_C_LIBRARY="${CONDA_PREFIX}/lib/libspatialindex_c.so"
EOF
```

---

## 4. 关键路径速查

| 用途 | 路径 |
|---|---|
| 项目根目录 | `/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN` |
| conda 环境 | `/bsuscratch/jiahuizhang/envs/magician` |
| 环境 python / pip | `/bsuscratch/jiahuizhang/envs/magician/bin/{python,pip}` |
| 激活脚本（自建） | `<项目根>/activate_magician.sh` |
| 数据集 | `<项目根>/data/Macarons++/<scene>/` |
| 权重（运行用的） | `<项目根>/weights/macarons/trained_macarons.pth` |
| 其他权重 | `<项目根>/weights/{resnet,scone}/...` |
| CUDA 工具链 | `/cm/shared/apps/cuda11.8/toolkit/11.8.0`（`module load cuda11.8/toolkit/11.8.0`） |
| 安装脚本+日志原件 | `/bsuscratch/jiahuizhang/envs/magician/_setup/`（本文件夹 `scripts/`、`logs/` 是副本） |

---

## 5. 怎么运行

```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN
source activate_magician.sh                 # 激活环境（运行时无需 cuda 模块，见 8.6）
python test_magician_planning.py            # 完整跑很久（14 场景 × ~100 步）
python evaluation_lmdb.py                    # 跑完后算指标
```
> 正式做指标统计前，建议先清掉我中断试跑留下的不完整结果：
> `rm -rf results/scene_exploration/magician_lmdb`

---

## 6. 安装的关键版本

| 包 | 版本 | 备注 |
|---|---|---|
| python | 3.9.23 | |
| torch / torchvision / torchaudio | 2.6.0 / 0.21.0 / 2.6.0（+cu118） | glibc 2.17 上 cu118 的上限 |
| pytorch3d | 0.7.8 | 源码编译，sm_70 |
| diff_gaussian_rasterization / simple_knn | 源码编译 | sm_70 |
| numpy / scipy / pandas | 1.24.3 / 1.9.1 / 2.2.3 | pandas 原 2.3.3 无 cp39 wheel |
| sympy | 1.13.1 | torch 要求 |
| libspatialindex | 2.1.0（conda） | 修 rtree |

完整冻结清单见 `pip_freeze.txt`。

---

## 7. 没装的（MAGICIAN 运行用不到，有意跳过）

- `tetra_triangulation`（RaDe-GS 的 marching tetrahedra，要 CGAL，glibc 2.17 上编译最麻烦）
- `open3d` / `opencv-python` / `pycolmap` / `pyembree`（代码运行路径里 0 处导入）

如果以后需要 RaDe-GS 的网格提取功能，再单独编译 tetra_triangulation 即可。
```

---

## 8. ⭐ 重要更新：host / 容器分裂（2026-06-14 续）

排查"为什么我自己一直配不好"时发现的**核心问题**。

### 8.1 现象
- 你的交互终端（code-server）跑在 **apptainer 容器里**：`ldd → glibc 2.36 (Debian 12)`，`/.singularity.d` 存在。
- 但 GPU 节点的 **host 是 CentOS 7 / glibc 2.17**。
- 两个环境工具链完全不同：

| | host (CentOS 7) | 容器 (Debian 12) |
|---|---|---|
| glibc | 2.17 | 2.36 |
| gcc | 有（9.2，默认 PATH） | **没有**（只 bind-mount 了 `/cm/shared/apps/gcc`） |
| git | 有 | **没有** |
| glibc 开发头（features.h） | 有 | **没有**（容器只装了运行库，没 libc6-dev） |
| nvcc | `module load cuda11.8` | `/cm/shared/apps/cuda11.8/...`（bind-mount，可用） |

### 8.2 为什么你怎么配都出错
- 在 **host** 上 `conda env create -f environment.yml` → glibc 2.17 太旧，装不上（第 2 节）。
- 在 **容器** 里装 → glibc 够新，但**没 gcc / 没 git / 没系统头**，编译 CUDA 扩展时报
  `fatal error: features.h: No such file or directory`。
- **两边各缺一块**，所以单靠任一边都配不成。

### 8.3 解决办法：用 conda 自带工具链 + sysroot（容器里也能编）
```bash
conda install -c conda-forge gxx_linux-64=11 gcc_linux-64=11 sysroot_linux-64=2.17
```
- `sysroot_linux-64=2.17` 提供完整的 glibc 2.17 头文件（含 features.h），编译器默认就用它。
- 配合 bind-mount 的 nvcc(11.8)，**在容器里也能编译**，且产出的二进制是 **glibc 2.17 兼容**的 → host 和容器都能跑。
- 编译时设：
  ```bash
  conda activate /bsuscratch/jiahuizhang/envs/magician   # 设好 CC/CXX/CONDA_BUILD_SYSROOT
  export CUDA_HOME=/cm/shared/apps/cuda11.8/toolkit/11.8.0
  export PATH="$CUDA_HOME/bin:$PATH"
  export NVCC_PREPEND_FLAGS="-ccbin $CXX"      # nvcc 用 conda 的 host 编译器（带 sysroot）
  export TORCH_CUDA_ARCH_LIST="6.0;7.0"         # P100 + V100，两种卡通用
  ```
  参考脚本：`jiahui/build_pytorch3d_conda.sh`。

### 8.4 多 GPU 架构（很关键）
- gpu101 = **V100 (sm_70)**，gpu106 = **P100 (sm_60)** —— 不一样！
- 只按 sm_70 编的扩展，在 P100 上会报假 OOM：`Tried to allocate 131072.00 GiB`（整数溢出，不是真没显存）。
- 必须 `TORCH_CUDA_ARCH_LIST="6.0;7.0"` 同时编两种架构。
- **坑**：重编前要删源码目录里残留的 `build/`，否则 setuptools 复用旧 .o（旧架构）：
  ```bash
  rm -rf RaDe-GS/submodules/{diff-gaussian-rasterization,simple-knn}/build
  pip install --no-build-isolation --no-cache-dir --force-reinstall <submodule_dir>
  ```
- 验证架构：`cuobjdump <_C*.so> | grep 'arch ='` 应同时出现 sm_60 和 sm_70。

### 8.5 运行时的两个配置点
1. **GPU 索引**：`configs/test/test_in_default_scenes_config.json` 里的 `numGPU` 被当作 GPU 索引用
   （`cuda:<numGPU>`）。SLURM 通常只给你 1 块卡（`CUDA_VISIBLE_DEVICES=0`），所以已改成 **`numGPU: 0`**。
   （gpu101 那次分到 2 块卡，用 1 才没报错。）
2. **rtree**：见第 7 节，`SPATIALINDEX_C_LIBRARY` 由 activate.d 钩子自动设。

### 8.6 现在的状态
- env 已编译为 **glibc 2.17 + sm_60/sm_70**，host 和容器、P100 和 V100 **全部通用**。
- 在 gpu106 容器内 P100 上 `python test_magician_planning.py` 已跑通（beam search 正常推进）。
- 运行时**不需要** `module load cuda`（torch wheel 自带 CUDA 11.8 运行时，只要有 NVIDIA 驱动 / 容器 `--nv`）。
- 额外参考脚本：`jiahui/build_pytorch3d_conda.sh`、`jiahui/rebuild_env.sh`。
