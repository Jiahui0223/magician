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
| Random Walk | 5/6 | 0.208 | 0.290 | 0.366 | 0.288 | 0.395 | 0.484 |
| MAGICIAN | 5/6 | 0.312 | 0.485 | 0.667 | 0.454 | 0.740 | 0.893 |
| Ours | 5/6 | 0.464 | 0.684 | 0.797 | 0.720 | 0.885 | 0.918 |

Ours - MAGICIAN: AUC@20 +0.151 [+0.131, +0.169] (tiles 5/6, episodes 50/50); AUC@100 +0.130 [+0.107, +0.153] (tiles 5/6, episodes 49/50); Final +0.025 [+0.012, +0.042] (tiles 5/6, episodes 47/50)

Ours - Random Walk: AUC@20 +0.256 [+0.220, +0.288] (tiles 5/6, episodes 50/50); AUC@100 +0.430 [+0.403, +0.453] (tiles 5/6, episodes 50/50); Final +0.434 [+0.398, +0.467] (tiles 5/6, episodes 50/50)

MAGICIAN - Random Walk: AUC@20 +0.104 [+0.085, +0.125] (tiles 5/6, episodes 48/50); AUC@100 +0.300 [+0.269, +0.331] (tiles 5/6, episodes 49/50); Final +0.409 [+0.375, +0.436] (tiles 5/6, episodes 50/50)

invalid / missing: [('uav_HKisland_00', 'RANDOM', 0, 'no_visibility_official'), ('uav_HKisland_00', 'RANDOM', 1, 'no_visibility_official'), ('uav_HKisland_00', 'RANDOM', 2, 'no_visibility_official'), ('uav_HKisland_00', 'RANDOM', 3, 'no_visibility_official'), ('uav_HKisland_00', 'RANDOM', 4, 'no_visibility_official'), ('uav_HKisland_00', 'RANDOM', 5, 'no_visibility_official'), ('uav_HKisland_00', 'RANDOM', 6, 'no_visibility_official'), ('uav_HKisland_00', 'RANDOM', 7, 'no_visibility_official'), ('uav_HKisland_00', 'RANDOM', 8, 'no_visibility_official'), ('uav_HKisland_00', 'RANDOM', 9, 'no_visibility_official'), ('uav_HKisland_00', 'MAG-B', 0, 'no_visibility_official'), ('uav_HKisland_00', 'MAG-B', 1, 'no_visibility_official'), ('uav_HKisland_00', 'MAG-B', 2, 'no_visibility_official'), ('uav_HKisland_00', 'MAG-B', 3, 'no_visibility_official'), ('uav_HKisland_00', 'MAG-B', 4, 'no_visibility_official'), ('uav_HKisland_00', 'MAG-B', 5, 'no_visibility_official'), ('uav_HKisland_00', 'MAG-B', 6, 'no_visibility_official'), ('uav_HKisland_00', 'MAG-B', 7, 'no_visibility_official'), ('uav_HKisland_00', 'MAG-B', 8, 'no_visibility_official'), ('uav_HKisland_00', 'MAG-B', 9, 'no_visibility_official'), ('uav_HKisland_00', 'OURS', 0, 'no_visibility_official'), ('uav_HKisland_00', 'OURS', 1, 'no_visibility_official'), ('uav_HKisland_00', 'OURS', 2, 'no_visibility_official'), ('uav_HKisland_00', 'OURS', 3, 'no_visibility_official'), ('uav_HKisland_00', 'OURS', 4, 'no_visibility_official'), ('uav_HKisland_00', 'OURS', 5, 'no_visibility_official'), ('uav_HKisland_00', 'OURS', 6, 'no_visibility_official'), ('uav_HKisland_00', 'OURS', 7, 'no_visibility_official'), ('uav_HKisland_00', 'OURS', 8, 'no_visibility_official'), ('uav_HKisland_00', 'OURS', 9, 'no_visibility_official')]
max coverage after normalisation (evaluator): uav_AMtown_11 0.955, uav_AMtown_12 0.947, uav_AMvalley_01 0.998, uav_HKairport_00 0.908, uav_HKairport_01 0.859

