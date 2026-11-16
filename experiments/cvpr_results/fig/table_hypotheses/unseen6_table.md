| Prediction / combination | Hypotheses to planner | AUC@20 ↑ | Cov@20 ↑ | AUC@100 ↑ | Final ↑ |
|---|---|---|---|---|---|
| Deterministic U-Net | 1 (regressed mean) | **0.446** | 0.694 | **0.784** | 0.915† |
| Single generated sample | 1 | 0.438 | **0.700** | 0.778 | 0.907† |
| Average of 4 samples (b, y) | 4 → 1 | 0.435 | 0.693 | **0.784** | 0.922 |
| Average of 4 samples (depth) | 4 → 1 | 0.422 | 0.679 | **0.784** | 0.921 |
| Ours: 4 hypotheses + CDGR | 4 | 0.432 | 0.676 | 0.783 | **0.930** |

Macarons++, 6 scenes no method was trained on (alhambra, neuschwanstein, eiffel, bridge, barts, sestino museum) x 5 official starts, 100 steps, official coverage. Same network, training data, CDGR planner and protocol in every row. † = difference to Ours with 95% paired scene-bootstrap CI excluding 0. Bold = best.
