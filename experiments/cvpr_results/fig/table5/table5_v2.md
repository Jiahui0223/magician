| Prediction strategy | CRPS ↓ | Energy score ↓ | AUC@100 ↑ | AUC@50–100 ↑ | Final ↑ |
|---|---|---|---|---|---|
| Deterministic | 6.77† | 7.15† | **0.784** | 0.908 | 0.915† |
| Single sample | 8.54† | 8.57† | 0.778 | 0.904† | 0.919 |
| Mean (b, y) | 7.29† | 7.47† | **0.784** | 0.906 | 0.922 |
| Mean depth | 7.35† | 7.25† | **0.784** | 0.910† | 0.921 |
| Seen2Map (Ours) | **5.07** | **4.93** | 0.783 | **0.914** | **0.930** |

Macarons++ Unseen-6 (6 scenes no method was trained on) x 5 official starts, 100 steps, official coverage; same network family, training data, CDGR planner and protocol in every row. Single sample: mean over noise maps 0-3 (the maps of the 4-hypothesis rows). CRPS / energy score: accuracy of the predicted distribution of the future first-hit surface on held-out planning states of unseen scenes (lower is better). AUC@50-100: mean coverage over steps 50-100. † = difference to Seen2Map with 95% paired bootstrap CI excluding 0. Bold = best.
