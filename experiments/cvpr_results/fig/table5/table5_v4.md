| Prediction strategy | Energy score ↓ | Cov@100 ↑ |
|---|---|---|
| Deterministic | 7.15† | 0.915† |
| Single sample | 8.57† | 0.919 |
| Mean (b, y) | 7.47† | 0.922 |
| Mean depth | 7.25† | 0.921 |
| Seen2Map (Ours) | **4.93** | **0.930** |

Macarons++ Unseen-6 (6 scenes no method was trained on) x 5 official starts, 100 steps, official coverage; same network family, training data, CDGR planner and protocol in every row. Single sample: mean over noise maps 0-3 (the maps of the 4-hypothesis rows). Energy score: accuracy of the predicted distribution of the whole future surface seen from a candidate view (first-hit depth along every ray into unknown space, compared with the ground truth; lower is better), on held-out planning states of unseen scenes. Cov@t: coverage after t steps. Earlier coverage is comparable across rows (Cov@20 / Cov@50 differences to Seen2Map not significant; full time course in the supplementary). † = difference to Seen2Map with 95% paired bootstrap CI excluding 0. Bold = best.
