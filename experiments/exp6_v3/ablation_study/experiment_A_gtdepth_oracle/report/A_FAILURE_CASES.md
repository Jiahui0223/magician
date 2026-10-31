# Experiment A — 失败案例（自动规则，协议 §2；不渲染，只列状态与分数）

## cl / indoor — T1
| 类型 | 状态 | sid | G\* pp | 选点 | regret T1 / GTD pp |
|---|---|---|---|---|---|
| 预测失败（模型归一化 regret ≥ 0.5 且 GTD 选对，取 regret 最大） | oLBMNvg9in8_L1 s36 V2O t=5 (early) | 8600cd7151360a35 | 2.692 | T1: 候选 8 分 2117.0 增益 0.854 pp；GTD: 候选 0 增益 2.692 pp；oracle 最优 候选 0 分（T1）1740.5 GTD 1697.8 | 1.838 / 0.000 |
| 评分失败（GTD 归一化 regret ≥ 0.5，取最大） | TbHJrupSAjP_L1 s112 V2O t=0 (early) | 40c4905becd638db | 4.620 | T1: 候选 1 分 1237.9 增益 2.045 pp；GTD: 候选 1 增益 2.045 pp；oracle 最优 候选 4 分（T1）1125.5 GTD 1222.4 | 2.575 / 2.575 |
| 停滞状态且 G\* > 0.5 pp，取模型 regret 最大 | TbHJrupSAjP_L1 s112 V2O t=105 (late) | dfe3e3f9c7303c8d | 1.180 | T1: 候选 1 分 202.1 增益 0.325 pp；GTD: 候选 0 增益 1.180 pp；oracle 最优 候选 0 分（T1）198.7 GTD 426.6 | 0.855 / 0.000 |

## cl / indoor — B2
| 类型 | 状态 | sid | G\* pp | 选点 | regret B2 / GTD pp |
|---|---|---|---|---|---|
| 预测失败（模型归一化 regret ≥ 0.5 且 GTD 选对，取 regret 最大） | oLBMNvg9in8_L1 s36 V2M t=5 (early) | 47c66f1abae914ab | 4.339 | B2: 候选 5 分 1363.1 增益 1.960 pp；GTD: 候选 4 增益 4.339 pp；oracle 最优 候选 4 分（B2）1185.0 GTD 1448.8 | 2.379 / 0.000 |
| 评分失败（GTD 归一化 regret ≥ 0.5，取最大） | TbHJrupSAjP_L1 s112 V2O t=0 (early) | 40c4905becd638db | 4.620 | B2: 候选 1 分 1210.8 增益 2.045 pp；GTD: 候选 1 增益 2.045 pp；oracle 最优 候选 4 分（B2）804.1 GTD 1222.4 | 2.575 / 2.575 |
| 停滞状态且 G\* > 0.5 pp，取模型 regret 最大 | TbHJrupSAjP_L1 s112 V2O t=105 (late) | dfe3e3f9c7303c8d | 1.180 | B2: 候选 0 分 227.3 增益 1.180 pp；GTD: 候选 0 增益 1.180 pp；oracle 最优 候选 0 分（B2）227.3 GTD 426.6 | 0.000 / 0.000 |

## cl / outdoor — T1
| 类型 | 状态 | sid | G\* pp | 选点 | regret T1 / GTD pp |
|---|---|---|---|---|---|
| 预测失败（模型归一化 regret ≥ 0.5 且 GTD 选对，取 regret 最大） | barts s3 V2O t=15 (early) | 7178f5f0d9f5844c | 10.052 | T1: 候选 2 分 935.2 增益 2.854 pp；GTD: 候选 4 增益 10.052 pp；oracle 最优 候选 4 分（T1）799.7 GTD 3303.7 | 7.198 / 0.000 |
| 评分失败（GTD 归一化 regret ≥ 0.5，取最大） | colosseum s4 V2M t=15 (early) | cb5c88a984b0c379 | 7.633 | T1: 候选 0 分 1192.9 增益 1.217 pp；GTD: 候选 3 增益 2.925 pp；oracle 最优 候选 1 分（T1）1183.8 GTD 1315.3 | 6.416 / 4.709 |

## cl / outdoor — B2
| 类型 | 状态 | sid | G\* pp | 选点 | regret B2 / GTD pp |
|---|---|---|---|---|---|
| 预测失败（模型归一化 regret ≥ 0.5 且 GTD 选对，取 regret 最大） | barts s2 V2O t=5 (early) | 68d844ab2d7d7ad8 | 18.088 | B2: 候选 1 分 508.0 增益 1.785 pp；GTD: 候选 4 增益 18.088 pp；oracle 最优 候选 4 分（B2）275.6 GTD 3254.2 | 16.303 / 0.000 |
| 评分失败（GTD 归一化 regret ≥ 0.5，取最大） | colosseum s4 V2M t=15 (early) | cb5c88a984b0c379 | 7.633 | B2: 候选 1 分 1437.6 增益 7.633 pp；GTD: 候选 3 增益 2.925 pp；oracle 最优 候选 1 分（B2）1437.6 GTD 1315.3 | 0.000 / 4.709 |

## fx / indoor — T1
| 类型 | 状态 | sid | G\* pp | 选点 | regret T1 / GTD pp |
|---|---|---|---|---|---|
| 预测失败（模型归一化 regret ≥ 0.5 且 GTD 选对，取 regret 最大） | zsNo4HB9uLZ_L0 s16 VALSEL t=0 (early) | b02b928dff45f644 | 2.938 | T1: 候选 4 分 680.3 增益 0.591 pp；GTD: 候选 5 增益 2.938 pp；oracle 最优 候选 5 分（T1）425.2 GTD 765.0 | 2.347 / 0.000 |
| 评分失败（GTD 归一化 regret ≥ 0.5，取最大） | X7HyMhZNoso_L0 s30 VALSEL t=20 (early) | 2dfa3b62f2208a5e | 1.105 | T1: 候选 1 分 709.7 增益 0.371 pp；GTD: 候选 5 增益 0.159 pp；oracle 最优 候选 4 分（T1）265.5 GTD 277.5 | 0.734 / 0.946 |
| 停滞 | 无 G\* > 0.5 pp 的停滞状态 | | | | |

## fx / indoor — B2
| 类型 | 状态 | sid | G\* pp | 选点 | regret B2 / GTD pp |
|---|---|---|---|---|---|
| 预测失败（模型归一化 regret ≥ 0.5 且 GTD 选对，取 regret 最大） | EU6Fwq7SyZv_L0 s67 VALSEL t=5 (early) | 36e9405a66ce5f15 | 3.379 | B2: 候选 7 分 2618.2 增益 1.616 pp；GTD: 候选 4 增益 3.379 pp；oracle 最优 候选 4 分（B2）2552.2 GTD 2153.4 | 1.763 / 0.000 |
| 评分失败（GTD 归一化 regret ≥ 0.5，取最大） | X7HyMhZNoso_L0 s30 VALSEL t=20 (early) | 2dfa3b62f2208a5e | 1.105 | B2: 候选 5 分 775.1 增益 0.159 pp；GTD: 候选 5 增益 0.159 pp；oracle 最优 候选 4 分（B2）286.2 GTD 277.5 | 0.946 / 0.946 |
| 停滞 | 无 G\* > 0.5 pp 的停滞状态 | | | | |

## fx / outdoor — T1
| 类型 | 状态 | sid | G\* pp | 选点 | regret T1 / GTD pp |
|---|---|---|---|---|---|
| 预测失败（模型归一化 regret ≥ 0.5 且 GTD 选对，取 regret 最大） | barts s1 V2M t=20 (mid) | 5ed9edc55723455a | 2.814 | T1: 候选 1 分 222.9 增益 0.287 pp；GTD: 候选 3 增益 2.814 pp；oracle 最优 候选 3 分（T1）202.2 GTD 419.1 | 2.527 / 0.000 |
| 评分失败（GTD 归一化 regret ≥ 0.5，取最大） | colosseum s0 V2M t=15 (early) | fdbfeda83bedfaf2 | 1.607 | T1: 候选 5 分 1932.0 增益 1.296 pp；GTD: 候选 2 增益 0.718 pp；oracle 最优 候选 3 分（T1）736.8 GTD 693.0 | 0.311 / 0.889 |

## fx / outdoor — B2
| 类型 | 状态 | sid | G\* pp | 选点 | regret B2 / GTD pp |
|---|---|---|---|---|---|
| 预测失败（模型归一化 regret ≥ 0.5 且 GTD 选对，取 regret 最大） | barts s4 V2M t=10 (early) | 65ea4e837b19444d | 12.264 | B2: 候选 6 分 903.7 增益 3.134 pp；GTD: 候选 4 增益 12.264 pp；oracle 最优 候选 4 分（B2）426.5 GTD 3561.5 | 9.130 / 0.000 |
| 评分失败（GTD 归一化 regret ≥ 0.5，取最大） | colosseum s0 V2M t=15 (early) | fdbfeda83bedfaf2 | 1.607 | B2: 候选 2 分 1935.1 增益 0.718 pp；GTD: 候选 2 增益 0.718 pp；oracle 最优 候选 3 分（B2）894.4 GTD 693.0 | 0.889 / 0.889 |
