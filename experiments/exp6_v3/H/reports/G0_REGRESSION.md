# G0_REGRESSION — H 回归（barts_s0；自动生成，`scripts/g0_regression.py`）

| 检查 | 结果 |
|---|---|
| h0 与 Table 1 V3 记录：执行动作 / 官方覆盖逐位一致 | True / True |
| h1deg 与 Table 1 V3 记录：执行动作 / 官方覆盖逐位一致 | True / True |
| h1deg 每步分数与 h0 冻结 DistP1：逐位相同的步数 / 最大相对差 / 每步选择相同 | 41/100 / 3.30e-16（浮点求和顺序：numpy 与 torch） / True |
| h3d1（H3 代码、深度 1）与 h2：执行动作 / 官方覆盖逐位一致 | True / True |
| H2 分数 == mean − 0.25·std（ddof 0），每步最大绝对差 | 0.0 |
| H2 记录的 std 与样本值重算（ddof 0）逐位相同 | True |
| H2：风险项改变同一状态的 argmax（相对 H1 均值）的步数 | 2 / 68 个非回退步 |
| λ = 0 恒等 | 代码中 H2 = mean − λ·std，λ = 0 时就是 H1 的 mean（同一个 np.mean 结果）；固定状态上的 λ = 0 变体与 H1 的选择见 G2_FIXED_STATE.md |
| H3 冒烟：步数 / 每步第二层唯一视角数均值 / 每步决策中位数（s） | 100 / 30.9 / 3.84 |
| 位姿表与历史服务记录逐位核对（14 个场景） | 39243 行，5600 条记录，不符 0 |

| 运行 | h_policy.py sha256 | FROZEN_CONFIG.yaml sha256 |
|---|---|---|
| h0 | ccfdf4af2783… | ab8aff7077dd… |
| h1deg | ccfdf4af2783… | ab8aff7077dd… |
| h2 | ccfdf4af2783… | ab8aff7077dd… |
| h3 | ccfdf4af2783… | ab8aff7077dd… |
| h3d1 | ccfdf4af2783… | ab8aff7077dd… |

Table 1 V3 记录的 checkpoint sha256：`639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb`；回归运行的规划器报告的 sha256：`639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb`。
