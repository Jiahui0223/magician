# MAGICIAN 项目总结（环境 + 复现）

> 作者：jiahui（Boise State HPC）　最后更新：2026-06-22
> 本文是**权威总结**：完整记录了系统/软件版本、为什么官方 `environment.yml` 装不上、**如何从零重配环境**、怎么跑、以及复现结果对比。
> 详细的安装踩坑过程见同目录 `INSTALL_NOTES.md`（本文是它的浓缩 + 更新版）。

---

## 0. 一句话

- 项目：MAGICIAN（CVPR 2026 主动建图 / active mapping，beam-search 探索）。
- 环境：`/bsuscratch/jiahuizhang/envs/magician`（conda，Python 3.9.23）。
- 激活：在仓库根目录 `source activate_magician.sh`。
- 跑：`python -u test_magician_planning.py`（提交脚本 `run.sh`，跑在 `gpu-l40`）。
- 评测：`python evaluation_lmdb.py`。
- **复现成功**：整体 AUC 0.702 / Max Coverage 0.907，约在论文（0.721 / 0.919）的 ~2% 以内。

---

## 1. 系统环境完整记录

| 项目 | 值 | 说明 |
|---|---|---|
| 集群 | Boise State HPC（Borah） | 文档 https://hpc.boisestate.edu/en/latest/scheduling/ |
| 操作系统 (host) | **CentOS 7**（内核 3.10 el7） | |
| **host glibc** | **2.17** | ⭐ 最关键约束，决定 wheel/conda 包的版本天花板 |
| 终端容器 | apptainer / Debian 12，**glibc 2.36** | code-server 跑在容器里（`/.singularity.d`）；与 host 分裂 |
| conda | 23.1.0 | base 在 `/cm/shared/apps/conda/23.1.0`（只读） |
| conda envs 目录 | `/bsuscratch/jiahuizhang/envs` | |
| CUDA toolkit | **11.8.0**（nvcc 11.8.89） | `module load cuda11.8/toolkit/11.8.0`；正好匹配 torch cu118 |
| host gcc | 9.2.0 | 但容器里**没有** gcc/git/glibc 头 → 用 conda 工具链解决（见 §4） |
| NVIDIA 驱动 | 支持 CUDA 11.8 运行时 | 运行时**不需要** module load cuda（torch wheel 自带 11.8 运行时） |

### 跑过 / 支持的 GPU（按算力 sm）

| 节点 / 卡 | 算力 | 状态 |
|---|---|---|
| `gpu101` Tesla V100-PCIE-16GB | sm_70 | ✅ 验证通过 |
| `gpu106` Tesla P100 | sm_60 | ✅ 验证通过 |
| `gpu-l40` 分区 L40（4×，64 核/节点） | sm_89 | ✅ **正式复现就在这里跑的**（`run.sh`） |
| A100 | sm_80 | ✅ 已编入，可用 |

> 最终 CUDA 扩展编译架构 = **`6.0;7.0;8.0;8.9`**（P100/V100/A100/L40 全覆盖）。
> ⚠️ 不同卡 sm 不同：只按某一架构编的 `.so` 在别的卡上会报假错（P100 上的假 OOM `Tried to allocate 131072.00 GiB`，或 L40 上 `no kernel image is available`）。所以一次把四种架构都编进去。

---

## 2. 软件版本完整记录

| 包 | 版本 | 备注 |
|---|---|---|
| python | 3.9.23 | conda-forge，不锁 build |
| torch | **2.6.0+cu118** | ⭐ glibc 2.17 上 cu118 的**最高版**（2.7+ 是 manylinux_2_28，装不上） |
| torchvision | 0.21.0+cu118 | |
| torchaudio | 2.6.0+cu118 | |
| pytorch3d | 0.7.8 | 源码编译（GitHub `@V0.7.8` tarball） |
| diff_gaussian_rasterization | 源码编译 | RaDe-GS 子模块 |
| simple_knn | 源码编译 | INRIA gitlab 归档 tarball（git 协议拉不动） |
| numpy / scipy / pandas | 1.24.3 / 1.9.1 / **2.2.3** | pandas 原 2.3.3 无 cp39 wheel → 降 2.2.3 |
| sympy | 1.13.1 | torch 2.6 要求精确这个版本 |
| libspatialindex | 2.1.0（conda） | 修 rtree（见 §6 坑 4） |
| pip | 23.0.1 | 见 §6 坑 5（pip 曾被搞坏，已修复） |

> 完整冻结清单：`jiahui/pip_freeze.txt`。
> **没装**（运行路径用不到）：tetra_triangulation、open3d、opencv-python、pycolmap、pyembree。

---

## 3. 为什么官方 `environment.yml` 装不上

`environment.yml` 是在一台 **glibc ≥ 2.28** 的机器上导出的，本集群 host 是 **glibc 2.17**：

1. **conda 段**写死了 build hash（如 `libgcc=15.2.0`），要求 `__glibc>=2.28` → `conda env create` 直接 `UnsatisfiableError`。
2. **pip 段** `torch==2.7.1+cu118` 是 `manylinux_2_28` wheel（要 glibc 2.28）→ `No matching distribution`。
3. `pandas==2.3.3` 没有 Python 3.9（cp39）的预编译 wheel。

**修法**：不照搬 yml。用**最小 conda 环境（不锁 build）+ pip**，并把几个包降到 glibc 2.17 能用的版本（torch 2.6.0、pandas 2.2.3）。

**host/容器分裂**额外加了一层坑：host 有 gcc/git 但 glibc 太旧；容器 glibc 够新但**没 gcc/git/glibc 头**（编译报 `features.h: No such file or directory`）。两边各缺一块 → 用 **conda 自带工具链 + sysroot 2.17** 一招解决（见 §4 步骤 5）。

---

## 4. ⭐ 从零重新配置环境（完整命令）

> 这套流程**容器里、host 上都能编**，产出 glibc 2.17 兼容 + 四种 GPU 架构通用的二进制。
> 把 `ENV`、`PROJ` 设好后逐段执行。建议在 GPU 节点（能联网、有 nvcc）上做。

```bash
ENV=/bsuscratch/jiahuizhang/envs/magician
PROJ=/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN
source /cm/shared/apps/conda/23.1.0/etc/profile.d/conda.sh
```

### 步骤 1 — 子模块
```bash
cd "$PROJ"
git submodule update --init --recursive   # simple-knn 会失败，下一步手动补
```

### 步骤 2 — simple-knn（git 协议拉不动，用官方归档 tarball）
```bash
cd "$PROJ/RaDe-GS/submodules"
COMMIT=44f764299fa305faf6ec5ebd99939e0508331503
curl -sL "https://gitlab.inria.fr/bkerbl/simple-knn/-/archive/$COMMIT/simple-knn-$COMMIT.tar.gz" -o /tmp/simple-knn.tar.gz
tar -xzf /tmp/simple-knn.tar.gz -C /tmp/
mkdir -p simple-knn && cp -r /tmp/simple-knn-$COMMIT/. simple-knn/
cd "$PROJ"
```

### 步骤 3 — 最小 conda 环境（关键：**不锁 build**）
```bash
conda create -y -p "$ENV" -c conda-forge python=3.9 pip setuptools wheel
```

### 步骤 4 — torch 栈 + 核心依赖（pip）
```bash
"$ENV/bin/pip" install --prefer-binary \
    torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
"$ENV/bin/pip" install --prefer-binary -r "$PROJ/jiahui/scripts/reqs_core.txt"  # environment.yml 的 pip 段(去 torch/nvidia-*/pyembree, pandas→2.2.3)
"$ENV/bin/pip" install "sympy==1.13.1"
```

### 步骤 5 — conda 编译工具链（解决容器无 gcc/无系统头）
```bash
conda install -y -p "$ENV" -c conda-forge \
    gxx_linux-64=11 gcc_linux-64=11 sysroot_linux-64=2.17
# sysroot_linux-64=2.17 提供 glibc 2.17 头文件(含 features.h)，编出来的二进制 host/容器通用
```

### 步骤 6 — 编译 3 个 CUDA 扩展（四架构一次编全）
```bash
conda activate "$ENV"                              # 设好 CC/CXX/CONDA_BUILD_SYSROOT
export CUDA_HOME=/cm/shared/apps/cuda11.8/toolkit/11.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"             # nvcc 用 conda 的 host 编译器(带 sysroot)
export TORCH_CUDA_ARCH_LIST="6.0;7.0;8.0;8.9"       # P100 V100 A100 L40
export FORCE_CUDA=1 MAX_JOBS=12

# 重编前务必删残留 build/，否则 setuptools 复用旧架构 .o
rm -rf "$PROJ"/RaDe-GS/submodules/{diff-gaussian-rasterization,simple-knn}/build

pip install --no-build-isolation --no-cache-dir --force-reinstall "$PROJ/RaDe-GS/submodules/diff-gaussian-rasterization"
pip install --no-build-isolation --no-cache-dir --force-reinstall "$PROJ/RaDe-GS/submodules/simple-knn"
pip install --prefer-binary fvcore
pip install --no-build-isolation --no-cache-dir --force-reinstall "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"  # 最慢 ~20min
```
> 现成脚本：`/bsuscratch/jiahuizhang/envs/magician/_setup/recompile_arch.sh`（已含上述全部 + 架构验证）。
> 验证：`cuobjdump <_C*.so> | grep -oE 'sm_[0-9]+'` 应出现 `sm_60 sm_70 sm_80 sm_89`。

### 步骤 7 — 修 rtree（缺 libspatialindex）
```bash
conda install -y -p "$ENV" -c conda-forge libspatialindex
mkdir -p "$ENV/etc/conda/activate.d"
cat > "$ENV/etc/conda/activate.d/zz_magician_env.sh" <<'EOF'
export SPATIALINDEX_C_LIBRARY="${CONDA_PREFIX}/lib/libspatialindex_c.so"
EOF
```

### 步骤 8 — 数据 + 权重
```bash
# 数据集(HuggingFace, Macarons++.zip ~1.2GB)
"$ENV/bin/python" -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download(repo_id='sli016/Macarons-plus-plus', filename='Macarons++.zip', repo_type='dataset', local_dir='./data')"
unzip -o -q ./data/Macarons++.zip -d ./data
mv data/macarons++ data/Macarons++      # ⚠️ 解压是小写，必须改大写匹配 config
rm -f data/Macarons++.zip

# 权重(Google Drive 文件夹)
"$ENV/bin/pip" install gdown
"$ENV/bin/gdown" --folder "https://drive.google.com/drive/folders/1wyc9_QFmcxOz4oerE8kCQ3I8LO5zioZL" -O ./weights
```

### 步骤 9 — 运行前的配置改动
- `configs/test/test_in_default_scenes_config.json`：`"numGPU": 0`
  （**numGPU 是 GPU 索引不是数量**：`device = torch.device("cuda:"+str(numGPU))`；SLURM 给一块卡 → 必须 0）。

---

## 5. 运行 / 复现 / 评测

### 交互式
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN
source activate_magician.sh           # 运行时无需 module load cuda
python -u test_magician_planning.py   # 14 场景 × 5 起点 × ~100 步，很久
python evaluation_lmdb.py             # 算指标
```

### SLURM 提交（正式复现用的）
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN
sbatch run.sh        # -p gpu-l40, --gres=gpu:L40:4, 日志在 jiahui/log/magician_<jobid>.{out,err}
```
> `run.sh` 跑的就是 `test_magician_planning.py` —— **它就是复现入口**。
> 干净统计前先清中断残留：`rm -rf results/scene_exploration/magician_lmdb`。
> `-u` 不缓冲，进度实时写进 `.out`（否则重定向到文件时会块缓冲，看着"没输出"）。

---

## 6. 关键路径 + 常见坑速查

### 关键路径
| 用途 | 路径 |
|---|---|
| 项目根 | `/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN` |
| conda 环境 | `/bsuscratch/jiahuizhang/envs/magician` |
| 激活脚本 | `<根>/activate_magician.sh` |
| 提交脚本 | `<根>/run.sh`　日志 `<根>/jiahui/log/` |
| 数据 | `<根>/data/Macarons++/<scene>/` |
| 运行权重 | `<根>/weights/macarons/trained_macarons.pth` |
| 评测结果 LMDB | `<根>/results/scene_exploration/magician_lmdb` |
| 安装脚本/日志原件 | `<env>/_setup/`（`recompile_arch.sh` 等） |
| 可视化输出 | `<根>/work_dir/vis/` |

### 常见坑
1. **GPU 架构不匹配** → 假 OOM / `no kernel image`：四架构一次编全 + 删 `build/` + `--no-cache-dir --force-reinstall`。
2. **环境名大小写**：解压出 `macarons++` 必须改 `Macarons++`。
3. **numGPU 是索引**：SLURM 单卡 → `numGPU: 0`，否则 `invalid device ordinal`。
4. **rtree 缺 libspatialindex**：conda 装 + activate.d 钩子设 `SPATIALINDEX_C_LIBRARY`。
5. **pip 被搞坏**（`cannot import RequirementInformation`）：`pip install --upgrade pip`(→26) 后又被 conda 部分回退到 22，文件混版。修：`rm -rf $ENV/lib/python3.9/site-packages/pip pip-*.dist-info` 再 `python -m ensurepip --upgrade`（→干净的 23.0.1）。**教训：环境定型后别再升级 pip。**
6. **`.out` 没输出**：`python -u`。

---

## 7. 复现结果对比（vs 论文）

| 指标 | 复现 | 论文 | 说明 |
|---|---|---|---|
| 整体 AUC | **0.702** | 0.721 | 100 步覆盖率曲线的 trapz |
| 整体 Max Coverage | **0.907** | 0.919 | |

- **结论：复现成功**（~2% 以内）。
- 个别场景偏低（如 pantheon 平均 0.743 vs 论文 0.842）几乎全由**单条卡住的轨迹**拉低：pantheon 5 条起点里 traj0/1/3/4 ≈ 0.84，只有 **traj2 = 0.35**（困在穹顶内壁）。
- 归因：困难起点 + **跨硬件浮点不确定性**让 beam search 在那一条上走偏；作者报告的是 5 个起点的平均。

### 可视化工具（自建）
- `visualize_exploration.py` —— 单条轨迹的完整动图 + 全步网格 → `work_dir/vis/<scene>_traj<t>.{gif,_grid.png}`
  例：`python visualize_exploration.py --scene pantheon --traj 2 --every 1`
- `compare_trajectories.py` —— 一张图并排五条轨迹（带覆盖率标注）→ `work_dir/vis/<scene>_compare.png`
  例：`python compare_trajectories.py --scene pantheon`
