# FisherRF 复现 · 环境（ENVIRONMENT）

> **状态（2026-11-11 04:46）**：新 env `fisherrf`（`/bsuscratch/jiahuizhang/envs/fisherrf`，13 GB）**已建好**。
> - 已就绪：torch 2.2.2+cu118、habitat-sim / habitat-lab / habitat-baselines 0.2.4，以及两个 CUDA 扩展（FisherRF 改造版 `diff_gaussian_rasterization` 和 `simple_knn`，sm_89）。
> - 已通过 CPU 检查：两个扩展的 `_C` 都能 import，`backward_power` 接口存在，`pip check` 无冲突。
> - **BLOCKER B1：open3d 装不上**（见 §5）。官方的 `models/SLAM/gaussian.py:7` 和 `tester.py` 都在顶层 `import open3d`，所以按原样 import 会报 `ModuleNotFoundError: No module named 'open3d'`。我没有用其它方法绕开，等用户决定。
> - 另外，`models/SLAM/utils/eval_helpers.py:19` 在 import 时就要求 GPU（见 §5 B2）。这不是安装问题，但意味着完整的 import 检查只能在 GPU 上做。

日期 2026-11-11，执行节点 gpu114（宿主机 CentOS 7、glibc 2.17、gcc 9.2.0 模块、NVIDIA 驱动 550、`__cuda = 12.4`）。
**全程不使用 GPU**：所有命令都设 `CUDA_VISIBLE_DEVICES=""`；编译 CUDA 扩展不需要 GPU。没有提交 Slurm 作业；没有改动任何已有 env（seen2scene、magician、macarons 等）。只读用途：用 seen2scene 和 magician 的 python 跑过 `pip install --dry-run`，用 magician 的 PyTorch3D 在 CPU 上测过内参。

> **授权说明**：最初的任务是"只做 dry-run，把命令交给用户执行"。约 04:20，协调方转达："用户明确授权由我用 conda 实际创建并安装新 env `/bsuscratch/jiahuizhang/envs/fisherrf`"。因此下文的 Executed 一节是我实际执行的命令。仍然遵守的约束：只新建，不碰已有 env；装不上的步骤立即停止，记录错误，不换方法绕开。

---

## 1. 选型与理由

| 组件 | 选择 | 理由与证据 |
|---|---|---|
| Python | 3.9（conda-forge 3.9.23） | habitat-sim 0.2.4 只有 py3.9 的构建（`conda search -c aihabitat habitat-sim=0.2.4`）；官方 `AM/environment.yml:81` 也是 python 3.9.19 |
| habitat-sim | 0.2.4 `py3.9_headless_linux_f179b584…`，来自 **aihabitat 频道** | 官方指定 v0.2.4（`AM/README.md:57`）。**conda-forge 上没有 habitat-sim**（`conda search -c conda-forge habitat-sim` 返回 No match）。headless 版用 EGL，集群上没有 X；FisherRF 用不到物理引擎，所以选 nobullet |
| habitat-lab / habitat-baselines | 0.2.4（PyPI 上的版本号为 `0.2.420230405`），用 `--no-deps` 安装 | 官方要求 v0.2.4；`tester.py` 顶层 import `habitat`，`models.UPEN → planning/ddppo_policy.py:7` import `habitat_baselines` |
| torch | **2.2.2+cu118**、torchvision 0.17.2+cu118（pip 安装） | 与官方 `AM/environment.yml:343, 347` 的 pip 锁定版本**完全相同**。驱动 550（CUDA 12.4）可以运行 CUDA 11.8 的程序；magician env 的 torch 2.6.0+cu118 在同一集群上已经跑通（`MAGICIAN/jiahui/INSTALL_NOTES.md:54, 191`）。wheel 内二进制最高需要 GLIBC_2.16（见 §2 D3） |
| nvcc | `module load cuda11.8/toolkit/11.8.0`（11.8.89） | 与 torch cu118 精确匹配；CPU 上已编译通过（§2 D5）。协调方说"没有可用模块时再用 conda cuda-nvcc"，模块可用，所以没有用 conda nvcc |
| host 编译器 | conda-forge `gxx_linux-64=11`、`sysroot_linux-64=2.17`，配合 `NVCC_PREPEND_FLAGS="-ccbin $CXX"` | CUDA 11.8 支持 gcc ≤ 11；conda python 的 sysconfig 指向 `x86_64-conda-linux-gnu-*` 编译器；有 sysroot 2.17 时，产物与宿主 glibc 2.17 兼容。这套做法 magician env 已在本集群用过（`INSTALL_NOTES.md:235-249`） |
| 架构 | `TORCH_CUDA_ARCH_LIST="8.9"` | L40S（sm_89），按协调方要求 |
| 其余 pip 依赖 | 尽量使用官方 `environment.yml` 的版本 | 见 §3 E4。例外：`faster-fifo` 改用 1.5.2 的 wheel，因为 1.4.5 只有 sdist，需要编译；habitat-baselines 只要求 ≥1.4.2 |
| open3d | **装不上（BLOCKER）** | 见 §5 B1 |

## 2. Dry-run 和只读检查的证据（安装前）

原始输出在 `install_logs/` 和 scratch 目录 `/bsuhome/jiahuizhang/.claude/jobs/fde6a1ae/tmp/`。

- **D1 conda dry-run A：成功。**
  - 命令：`conda create --dry-run -p <tmp> --override-channels -c conda-forge -c aihabitat python=3.9 pip setuptools wheel habitat-sim=0.2.4=py3.9_headless_linux_f179b584… gcc_linux-64=11 gxx_linux-64=11 sysroot_linux-64=2.17`
  - 用时 7 分 33 秒；`success: true`；共 110 个包，其中 108 个来自 conda-forge，2 个来自 aihabitat（habitat-sim、habitat-sim-mutex）。
  - conda 识别到的虚拟包为 `__glibc=2.17`、`__cuda=12.4`。
  - 完整清单：`install_logs/D1_conda_dryrun_habitat_ok.txt`。
- **D2 conda dry-run B：失败。** 在 A 的基础上加 `open3d=0.18.0`，得到 `UnsatisfiableError`：open3d 0.18 → qt6-main → libpng / libxcb 等，与 habitat-sim 0.2.4 的 `libxcb >=1.13,<1.14`、`xorg-libx11` 约束冲突。完整报错：`install_logs/D2_conda_dryrun_open3d_conflict.txt`。conda-forge 上 py39 的 open3d 只有 0.18.0 和 0.19.0。
- **D3 habitat-sim 二进制。** 下载 `habitat-sim-0.2.4-py3.9_headless_linux_f179….tar.bz2`（md5 fc62e466…），对 3 个 `.so` 做 `objdump -T`：
  - 最高 GLIBC_2.14、GLIBCXX_3.4.19；
  - 需要 libEGL.so.1 和 libOpenGL.so.0，宿主机的 `ldconfig -p` 中都有（`/cm/local/apps/cuda-driver/libs/550.127.05/lib64/`），也有 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`；
  - 所以在 glibc 2.17 上可以安装。
- **D4 torch 2.2.2+cu118 wheel（cp39，linux_x86_64）。** `libtorch_cuda.so` 和 `libc10.so` 最高需要 GLIBC_2.14；`libtorch_cpu.so` 和 `libtorch_python.so` 最高需要 GLIBC_2.16。
- **D5 nvcc 11.8 + gcc 9.2，仅编译。** `forward.cu`、`backward.cu`、`rasterizer_impl.cu` 都以 `-gencode arch=compute_89,code=sm_89` 编译成功，只有 glm 的 warning；`cuobjdump` 显示 `arch = sm_89`。
- **D6 pip 原生 dry-run。** 用 py3.9 解释器在本机执行 `pip install --dry-run --ignore-installed --report`，最终依赖清单共 127 个包：
  - 二进制 wheel **全部**满足 ≤ manylinux_2_17 或纯 Python；
  - 只有 4 个 sdist：gym 0.23.0、moviepy 1.0.3、antlr4-python3-runtime 4.9.3（三者都是纯 Python），以及 faster-fifo 1.4.5（要编译，因此改用 1.5.2 的 wheel）；
  - 解析结果（包名、版本、文件名）：`install_logs/D6_pip_dryrun_resolved.txt`，原始输出：`install_logs/D6_pip_dryrun.out`。
- **D7 open3d 的 pip wheel。** PyPI 上所有 cp39 的 wheel 都是 `manylinux_2_27`（0.14.1–0.18.0）或 `manylinux_2_31`（0.19.0），glibc 2.17 装不上。
- **D8 PyTorch3D 相机约定。** CPU 探针确认：f = 221.70、cx = 227.5、cy = 127.5，zbuf 是 z-depth。详见 `PROTOCOL_ADAPTATION.md` §3.2。

## 3. 命令清单（实际执行的脚本）

脚本在 `/bsuhome/jiahuizhang/.claude/jobs/fde6a1ae/tmp/install_scripts/`，副本见 `install_logs/scripts/`。所有步骤共用 `common.sh`：

```bash
export ENV=/bsuscratch/jiahuizhang/envs/fisherrf
export CONDA_PKGS_DIRS=/bsuscratch/jiahuizhang/cache/conda/pkgs
export PIP_CACHE_DIR=/bsuhome/jiahuizhang/.claude/jobs/fde6a1ae/tmp/pipcache
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
export AM=/bsuhome/jiahuizhang/scratch/projects/occupancy/FisherRF/FisherRF-active-mapping
source /cm/shared/apps/conda/23.1.0/etc/profile.d/conda.sh     # conda 23.1，classic solver，channel_priority strict
```

**E1 创建 conda env**（conda-forge 加 aihabitat；aihabitat 只提供 habitat-sim）

```bash
conda create -y -p $ENV --override-channels -c conda-forge -c aihabitat \
  python=3.9 pip "setuptools=68.2.2" wheel \
  "habitat-sim=0.2.4=py3.9_headless_linux_f179b584bcd713c5a2a998132211e2cae881d6d1" \
  gcc_linux-64=11 gxx_linux-64=11 sysroot_linux-64=2.17
```

**E2 安装 torch**（官方锁定版本）

```bash
$ENV/bin/python -m pip install --prefer-binary "torch==2.2.2+cu118" "torchvision==0.17.2+cu118" \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

**E3 安装 habitat-lab / habitat-baselines 0.2.4**（`--no-deps`；它们的依赖在 E4 中显式安装，以免与 conda 已装的 numba、quaternion、imageio 重复）

```bash
$ENV/bin/python -m pip install --no-deps "habitat-lab==0.2.420230405" "habitat-baselines==0.2.420230405"
```

**E4 安装其余依赖**

```bash
$ENV/bin/python -m pip install --prefer-binary "numpy==1.26.4" "setuptools==68.2.2" \
  "gym==0.23.0" "hydra-core==1.3.2" "omegaconf==2.3.0" "opencv-python==4.9.0.80" "protobuf==3.20.1" \
  "threadpoolctl==3.4.0" "moviepy==1.0.3" "faster-fifo==1.5.2" "ifcfg==0.24" "tensorboard==2.8.0" \
  "lmdb==1.4.1" "webdataset==0.1.40" \
  "yacs==0.1.8" "einops==0.7.0" "kornia==0.7.3" "torchmetrics==1.3.2" "tensorboardX==2.6.2.2" \
  "wandb==0.16.6" "rich==13.7.1" "scikit-learn==1.4.2" "matplotlib==3.5.3" "ipython==8.18.1" \
  "pyyaml==6.0.1" "pytorch-msssim==1.0.0" "lpips==0.1.4" "plyfile==1.0.3"
```

`matplotlib==3.5.3` 必须装，因为 `tester.py:699` 用到 `mpl.colormaps`，而 conda 解出的是 3.3.2。

**E5 编译 CUDA 扩展**（从克隆的副本编译，克隆本身不动）

```bash
B=$ENV/_build; mkdir -p $B
cp -r $AM/thirdparty/diff-gaussian-rasterization-modified $AM/thirdparty/simple-knn $B/
conda activate $ENV                                   # 设置 CC / CXX（conda gcc / g++ 11.4，sysroot 2.17）
source /etc/profile.d/modules.sh; module load cuda11.8/toolkit/11.8.0
export CUDA_HOME=/cm/shared/apps/cuda11.8/toolkit/11.8.0 PATH=$CUDA_HOME/bin:$PATH
export NVCC_PREPEND_FLAGS="-ccbin $CXX" TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 MAX_JOBS=8
(cd $B/diff-gaussian-rasterization-modified && python -m pip install --no-build-isolation --no-cache-dir -v .)
(cd $B/simple-knn && python -m pip install --no-build-isolation --no-cache-dir -v .)
```

**E6 只用 CPU 的 import 检查**（`E6_verify_cpu.sh`）。检查内容：
- torch 与两个 `_C` 扩展能否 import；
- `cuobjdump` 看 sm 架构；`objdump` 看 GLIBC 版本；`ldd` 看有没有缺库；
- habitat_sim / habitat / habitat_baselines 与其它依赖；
- 从克隆中 import 官方模块，并设 `PYTHONDONTWRITEBYTECODE=1`，不往克隆里写字节码。

**E7 导出版本**：

```bash
conda list -p $ENV --export > env_fisherrf_conda_list.txt
pip freeze > env_fisherrf_pip_freeze.txt
pip check
```

## 4. Executed（实际执行记录）

执行节点均为 gpu114，时区 -0700。每一步的完整输出在 `install_logs/E*.log`；E1 日志里的 conda 进度条已删除，其余内容原样保留。

| 步骤 | 开始 → 结束 | 退出码 | 结果摘要（摘自日志） |
|---|---|---|---|
| D1–D8 dry-run 与只读检查 | 03:53 → 04:25 | — | 见 §2 |
| **E1** conda create | 04:25:57 → 04:36:51 | **0** | 第一次用 current_repodata 求解失败，自动改用完整 repodata 后成功；"NEW packages" 共 110 个，下载 398.5 MB。主要包：python 3.9.23、habitat-sim 0.2.4（aihabitat，headless_nobullet）、numpy 1.26.4、numba 0.60.0、quaternion 2023.0.4、scipy 1.13.1、gcc / g++ 11.4.0、sysroot 2.17、libstdcxx 16.2.0。只有一条与结果无关的报错：aihabitat 的 notices.json 读取超时 |
| **E2** pip torch | 04:37:10 → 04:39:26 | **0** | `Successfully installed … torch-2.2.2+cu118 torchvision-0.17.2+cu118 triton-2.2.0 nvidia-cudnn-cu11-8.7.0.84 …` 等 nvidia-*-cu11 包 |
| **E3** habitat-lab / baselines | 04:39:42 → 04:39:45 | **0** | `Successfully installed habitat-baselines-0.2.420230405 habitat-lab-0.2.420230405` |
| **E4** pip 依赖 | 04:39:45 → 04:41:12 | **0** | 共 86 个包，含 gym 0.23.0、moviepy 1.0.3、faster-fifo 1.5.2、kornia 0.7.3、torchmetrics 1.3.2、wandb 0.16.6、matplotlib 3.5.3（`Attempting uninstall: matplotlib`，替换掉 conda 的 3.3.2）、opencv-python 4.9.0.80、scikit-learn 1.4.2 等。gym、moviepy、antlr4 是纯 Python sdist，本地构建成功 |
| **E5** CUDA 扩展编译 | 04:41:23 → 04:44:13 | **0 / 0** | `CC/CXX = x86_64-conda-linux-gnu-{cc,c++}`，conda-forge gcc 11.4.0；nvcc 11.8.89；torch 2.2.2+cu118 / CUDA 11.8；nvcc 参数 `-gencode=arch=compute_89,code=sm_89 -ccbin <conda c++> -I…/third_party/glm/`；输出 `Successfully installed diff-gaussian-rasterization-0.0.0` 和 `Successfully installed simple-knn-0.0.0`。有两个无害的 UserWarning：没有 ninja 所以退回 distutils；conda g++ 没有对应 CUDA 11.8 的版本上下界 |
| **E6** CPU import 检查（`CUDA_VISIBLE_DEVICES=""`） | 04:44:51 → 04:45:23 | 0 | 详见下面的列表 |
| **E6b** `.so` 检查（重跑 E6 第 2 节） | 04:45:58 | 0 | 两个 `_C*.so`：只有 **sm_89 的 SASS，没有 PTX**；最高 GLIBC_2.14；GLIBCXX_3.4.29 和 CXXABI_1.3.13 由 env 自带的 libstdc++ 16.2 提供；`ldd` 中唯一的 not found 是 torch 的库，它们在 import 时由 torch 加载。habitat_sim 和 _magnum 的 `.so` 在 `ldd` 中没有缺失项 |
| **E7** 导出版本 | 04:46:06 → 04:46:15 | 0 | `env_fisherrf_conda_list.txt`（222 行）、`env_fisherrf_pip_freeze.txt`（135 行）；`pip check` 输出 **No broken requirements found.**；env 总大小 13 GB |

E6 的结果（`install_logs/E6_verify_cpu.log`）：
- torch 2.2.2+cu118，`torch.cuda.is_available() = False`，符合预期，因为 GPU 被屏蔽；
- `diff_gaussian_rasterization._C` 能 import，导出 `mark_visible`、`rasterize_gaussians`、`rasterize_gaussians_backward`，并且 `GaussianRasterizer.__init__` 带 `backward_power` 参数，确认装上的是 **FisherRF 改造版**；
- `simple_knn._C` 能 import，导出 `distCUDA2`；
- `habitat_sim 0.2.4 | habitat 0.2.4 | habitat_baselines 0.2.4` 能 import；
- 其余 23 个依赖都能 import；
- `open3d`：`ModuleNotFoundError: No module named 'open3d'`，即 B1；
- 从克隆中 import 官方模块：
  - 成功：`models.SLAM.utils.recon_helpers`、`slam_external`、`slam_helpers`、`planning.astar`；
  - 失败：`models.SLAM.utils.eval_helpers`，报 `RuntimeError: No CUDA GPUs are available`，即 B2；
  - 失败：`models.SLAM.gaussian` 和 `tester`，报 `ModuleNotFoundError: No module named 'open3d'`，即 B1；
- 克隆检查完仍然干净（`git status --porcelain` 为空），因为设置了 `PYTHONDONTWRITEBYTECODE=1`。

还要记录两件事：
- E6 第 2 节第一次运行失败了，`ls … Not a directory`。原因是 `/etc/profile.d/modules.sh` 会覆盖 shell 变量 `ENV`（modules 用它指向 `init/profile.sh`）。E6b 改用另一个变量名重跑，结果正常。E5 不受影响，因为它在加载模块之前就已经用完了 `$ENV`。
- E6 import `eval_helpers` 时，torchmetrics 把 `alexnet-owt-7be5be79.pth`（233 MB）下载到了已有的 `TORCH_HOME=/bsuscratch/jiahuizhang/cache/torch/hub/checkpoints/`，这是共享缓存目录。
- 安装结束后，删除了我在 home scratch 里的大文件：habitat-sim tarball、pip 缓存（约 5 GB）。

## 5. Blocker 与已知问题

| # | 问题 | 证据 | 影响 | 可选方案（**都没有执行**，需要用户决定） |
|---|---|---|---|---|
| **B1** | **open3d 装不上** | conda 侧：`open3d=0.18.0` 与 habitat-sim 0.2.4 同装时报 `UnsatisfiableError`（`install_logs/D2_conda_dryrun_open3d_conflict.txt`）；conda-forge 上 py39 只有 0.18.0 / 0.19.0。pip 侧：cp39 的 wheel 全是 manylinux_2_27 或 2_31（D7）。E6 的报错：`ModuleNotFoundError: No module named 'open3d'` | `AM/models/SLAM/gaussian.py:7` 和 `models/SLAM/utils/gs_helpers.py:2` 都在顶层 import open3d，所以 import `models.SLAM.gaussian` 或 `tester` 都会失败。但 `gaussian.py` 里**一处 `o3d.` 都没用到**（grep 无结果），`gs_helpers.py` 也不在核心 import 链上 | (a) 在适配器进程里预先放一个空的 `open3d` 桩模块（`sys.modules`），官方代码一行不改。由于 open3d 只被 import、从未被调用，这对 FisherRF 的计算没有影响。这是最便宜的办法，但属于"换方法绕开"，需要用户批准。(b) 用 apptainer（1.2.5，glibc ≥ 2.27 的镜像）在容器里 pip 安装 open3d 0.18.0。(c) 在 glibc 2.17 上从源码编译 open3d 0.18，工作量大。(d) 另建一个不含 habitat-sim 的 env，用 conda-forge 的 open3d，再用桩模块替代 habitat，等于把问题转移到别处 |
| B2 | `eval_helpers.py:19` 在 import 时执行 `LearnedPerceptualImagePatchSimilarity(...).cuda()` | E6 报错：`RuntimeError: No CUDA GPUs are available` | 不是安装问题。没有 GPU 时无法 import `models.SLAM.gaussian`；第一次 import 还会下载 alexnet 权重（233 MB）到 `TORCH_HOME`。完整的 import / 运行验证只能放在 F1 的 GPU 作业里，**本次没有做** | 不需要处理，F1 在 GPU 上验证即可 |
| B3 | 扩展只编了 sm_89，没有 PTX | E6b：`cuobjdump --list-elf` 只有 sm_89，`--list-ptx` 为空 | 只能在 L40S 上运行，A30（sm_80）、V100 等都不行 | 如果要在 A30 上跑，用 `TORCH_CUDA_ARCH_LIST="8.0;8.9"` 重编 E5，只需几分钟 |
| B4 | 使用了 aihabitat 频道，偏离"只用 conda-forge"的约定 | conda-forge 上没有 habitat-sim（`No match found`）；官方安装方式就是 `-c conda-forge -c aihabitat` | E1 里只有 habitat-sim 和 habitat-sim-mutex 来自 aihabitat，其余 108 个包都来自 conda-forge，strict 优先级 | 如果不能接受，就得用桩模块替代 habitat（适配器本身不需要 Habitat 模拟器）。但那样就无法运行官方的 Habitat 实验 |
| B5 | 官方 Habitat 实验的数据不在本地 | 本地没有 Gibson、HM3D、`pointnav/*/v1/val/val.json.gz`；本地 MP3D 是原始 `v1/scans`，没有 habitat 用的 `.glb` 和 navmesh（已查 `~/scratch/data`） | 跑不了官方的 Habitat 实验。我们的适配器协议不需要这些数据 | 需要的话，要先签 Gibson / HM3D 的数据协议再下载，按 `AM/README.md:5-27` 组织目录 |
| B6 | env 里有两套编译器：模块里的 gcc 9.2，以及 conda 的 gcc 11.4（`conda activate` 会设置 CC / CXX） | E5 日志 | 以后在这个 env 里再编扩展，记得保留 `NVCC_PREPEND_FLAGS="-ccbin $CXX"`，与 E5 保持一致 | — |

**使用方法**（用于 F1 及以后）：

```bash
source /cm/shared/apps/conda/23.1.0/etc/profile.d/conda.sh && conda activate fisherrf
export PYTHONDONTWRITEBYTECODE=1 SLURM_JOB_QOS=kd-high
```

运行时不需要加载 CUDA 模块，torch 的 wheel 自带 CUDA 11.8 运行时。只有重新编译扩展时，才需要 `module load cuda11.8/toolkit/11.8.0`。

## 6. 以后在 GPU 上运行的验证片段（**这次没有运行**：gpu114 的 GPU 都被占用，而且 F0 规定不碰 GPU）

在空闲的 L40S 上，例如 F1 的 1-GPU 作业中运行：

```bash
source /cm/shared/apps/conda/23.1.0/etc/profile.d/conda.sh && conda activate /bsuscratch/jiahuizhang/envs/fisherrf
export PYTHONDONTWRITEBYTECODE=1 SLURM_JOB_QOS=kd-high      # 后者让 FisherRF 的 ClusterStateManager 失效（PROTOCOL_ADAPTATION.md R5）
python - <<'PY'
import sys, torch, numpy as np
sys.path.insert(0, "/bsuhome/jiahuizhang/scratch/projects/occupancy/FisherRF/FisherRF-active-mapping")
from models.SLAM.utils.recon_helpers import setup_camera
from diff_gaussian_rasterization import GaussianRasterizer as R
print(torch.cuda.get_device_name(0))
H, W = 256, 456; K = np.array([[221.7025, 0, 227.5], [0, 221.7025, 127.5], [0, 0, 1]])
cam = setup_camera(W, H, K, np.eye(4))
n = 2000; g = torch.Generator(device="cuda").manual_seed(0)
xyz = (torch.rand(n, 3, device="cuda", generator=g) - 0.5) * torch.tensor([4., 2., 0.], device="cuda") + torch.tensor([0., 0., 5.], device="cuda")
rv = dict(means3D=xyz.clone().requires_grad_(True), colors_precomp=torch.rand(n, 3, device="cuda", generator=g).requires_grad_(True),
          rotations=torch.tensor([[1., 0, 0, 0]], device="cuda").repeat(n, 1).requires_grad_(True),
          opacities=torch.full((n, 1), 0.5, device="cuda").requires_grad_(True),
          scales=torch.full((n, 3), 0.03, device="cuda").requires_grad_(True),
          means2D=torch.zeros(n, 3, device="cuda", requires_grad=True))
im, radius, depth = R(raster_settings=cam, backward_power=2)(**rv)
im.backward(torch.ones_like(im) * 1e-3)
Hm = torch.cat([rv["means3D"].grad, rv["opacities"].grad], 1)
print("render", tuple(im.shape), "visible", int((radius > 0).sum()), "| H min", Hm.min().item(), "max", Hm.max().item(), "nonzero", int((Hm.abs().sum(1) > 0).sum()))
assert Hm.min().item() >= 0, "power-2 accumulation must be non-negative"
import habitat_sim; print("habitat_sim", habitat_sim.__version__, "cuda_enabled", getattr(habitat_sim, "cuda_enabled", "?"))
PY
```

判据：H 非负、有非零项、渲染形状为 (3, 256, 456)。habitat-sim 的 EGL 渲染要到 F1 才会用到；我们的适配器根本不创建 Habitat 模拟器，只需要它能被 import。
