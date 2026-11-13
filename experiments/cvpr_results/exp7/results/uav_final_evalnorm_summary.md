### UAV-final
| Method | tiles | AUC@20 | AUC@50 | AUC@100 | Cov@20 | Cov@50 | Final |
|---|---|---|---|---|---|---|---|
| Random Walk | 6/6 | 0.196 | 0.275 | 0.351 | 0.266 | 0.374 | 0.467 |
| MAGICIAN | 6/6 | 0.304 | 0.469 | 0.651 | 0.440 | 0.722 | 0.879 |
| Ours | 6/6 | 0.433 | 0.650 | 0.773 | 0.672 | 0.862 | 0.910 |

Ours - MAGICIAN: AUC@20 +0.129 [+0.103, +0.150] (tiles 6/6, episodes 56/60); AUC@100 +0.123 [+0.107, +0.142] (tiles 6/6, episodes 58/60); Final +0.031 [+0.018, +0.044] (tiles 6/6, episodes 56/60)

Ours - Random Walk: AUC@20 +0.237 [+0.205, +0.268] (tiles 6/6, episodes 60/60); AUC@100 +0.423 [+0.386, +0.453] (tiles 6/6, episodes 60/60); Final +0.443 [+0.383, +0.496] (tiles 6/6, episodes 60/60)

MAGICIAN - Random Walk: AUC@20 +0.108 [+0.089, +0.127] (tiles 6/6, episodes 55/60); AUC@100 +0.300 [+0.254, +0.335] (tiles 6/6, episodes 58/60); Final +0.412 [+0.353, +0.456] (tiles 6/6, episodes 60/60)

invalid / missing: []
max coverage after normalisation (evaluator): uav_AMtown_11 0.955, uav_AMtown_12 0.947, uav_HKairport_00 0.908, uav_HKairport_01 0.859, uav_AMvalley_01_v1 0.934, uav_HKisland_00_v1 0.942

### UAV-v0-all
| Method | tiles | AUC@20 | AUC@50 | AUC@100 | Cov@20 | Cov@50 | Final |
|---|---|---|---|---|---|---|---|
| Random Walk | 6/6 | 0.195 | 0.271 | 0.342 | 0.268 | 0.369 | 0.453 |
| MAGICIAN | 6/6 | 0.293 | 0.469 | 0.656 | 0.429 | 0.726 | 0.896 |
| Ours | 6/6 | 0.437 | 0.663 | 0.790 | 0.683 | 0.884 | 0.926 |

Ours - MAGICIAN: AUC@20 +0.144 [+0.124, +0.164] (tiles 6/6, episodes 59/60); AUC@100 +0.133 [+0.114, +0.153] (tiles 6/6, episodes 59/60); Final +0.030 [+0.016, +0.045] (tiles 6/6, episodes 54/60)

Ours - Random Walk: AUC@20 +0.242 [+0.203, +0.279] (tiles 6/6, episodes 60/60); AUC@100 +0.447 [+0.412, +0.487] (tiles 6/6, episodes 60/60); Final +0.473 [+0.409, +0.559] (tiles 6/6, episodes 60/60)

MAGICIAN - Random Walk: AUC@20 +0.098 [+0.078, +0.119] (tiles 6/6, episodes 57/60); AUC@100 +0.314 [+0.278, +0.352] (tiles 6/6, episodes 59/60); Final +0.443 [+0.386, +0.518] (tiles 6/6, episodes 60/60)

invalid / missing: []
max coverage after normalisation (evaluator): uav_AMtown_11 0.955, uav_AMtown_12 0.947, uav_AMvalley_01 0.998, uav_HKairport_00 0.908, uav_HKairport_01 0.859, uav_HKisland_00 0.989

