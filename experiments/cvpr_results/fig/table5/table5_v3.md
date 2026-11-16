| Prediction strategy | CRPS ↓ | Energy score ↓ | Cov@80 ↑ | Cov@100 ↑ |
|---|---|---|---|---|
| Deterministic | 6.77† | 7.15† | 0.912 | 0.915† |
| Single sample | 8.54† | 8.57† | 0.910† | 0.919 |
| Mean (b, y) | 7.29† | 7.47† | 0.913 | 0.922 |
| Mean depth | 7.35† | 7.25† | 0.914† | 0.921 |
| Seen2Map (Ours) | **5.07** | **4.93** | **0.925** | **0.930** |

Macarons++ Unseen-6 (6 scenes no method was trained on) x 5 official starts, 100 steps, official coverage; same network family, training data, CDGR planner and protocol in every row. Single sample: mean over noise maps 0-3 (the maps of the 4-hypothesis rows). CRPS / energy score: accuracy of the predicted distribution of the future first-hit surface on held-out planning states of unseen scenes (lower is better). Cov@t: coverage after t steps. Early coverage is comparable across rows (Cov@20 / Cov@50 differences to Seen2Map not significant; full time course in the supplementary). † = difference to Seen2Map with 95% paired bootstrap CI excluding 0. Bold = best.
