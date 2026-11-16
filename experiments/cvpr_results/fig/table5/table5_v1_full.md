| Prediction / combination | Hypotheses to planner | CRPS ↓ | Energy score ↓ | Uncertainty | AUC@100 (out) | Final (out) | Final (out, unseen-6) | Final (in) |
|---|---|---|---|---|---|---|---|---|
| Deterministic U-Net | 1 (regressed mean) | 5.28† | 7.55† | — | **0.765** | 0.900† | 0.915† | 0.663 |
| Single generated sample | 1 | 6.89† | 9.34† | — | 0.756† | 0.900 | 0.907† | 0.669† |
| Average of 4 samples (b, y) | 4 → 1 | 6.15† | 8.24† | — | 0.759 | 0.898† | 0.922 | **0.699** |
| Average of 4 samples (depth) | 4 → 1 | 6.10† | 8.33† | — | 0.764 | 0.904 | 0.921 | 0.687 |
| Ours: 4 hypotheses, aggregated by CDGR | 4 | **4.25** | **5.47** | **ρ = 0.94** | 0.763 | **0.911** | **0.930** | **0.699** |

Same V3 network, training data, D1 (CDGR) planner and protocol in every row; only the prediction head (D0) or the way the K = 4 generated samples are used changes. CRPS / energy score: distributional accuracy of the predicted first-hit surface on 755 planning states (lower is better). Uncertainty: rank correlation between sample spread and actual error. AUC@100 / Final: closed-loop coverage, Macarons++ 14 scenes x 5 starts (out), 6 scenes no method was trained on (unseen-6), MP3D 22 levels x 2 starts (in). † = difference to Ours with 95% bootstrap CI excluding 0. Bold = best.
