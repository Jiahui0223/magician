# G2 item 3 — closed-loop temporal analysis (protocol G2-v1 §3)

ΔC(t) = scene-equal mean coverage difference (official), 95 % pointwise band = scene bootstrap (10,000, seed 0).

## All-14 (14 scenes)

| Pair | crossover t (ΔC ≤ 0 from t on) | last sign change | ΔC@20 | ΔC@50 | ΔC@100 | max ΔC (t) | min ΔC (t) | late gain C[100]−C[50] diff [CI] |
|---|---|---|---|---|---|---|---|---|
| D0 − G4 | 55 | 55 | +0.0212 | -0.0010 | -0.0108 | +0.0253 (16) | -0.0114 (99) | -0.0099 [-0.0315, +0.0152] |
| D0 − G1 | 94 | 94 | +0.0182 | +0.0106 | -0.0001 | +0.0221 (18) | -0.0009 (96) | -0.0107 [-0.0277, +0.0074] |

Steps with the band above 0 / below 0 (D0 − G4): 14 / 33; (D0 − G1): 28 / 0.

## Dev+Test-8 (8 scenes)

| Pair | crossover t (ΔC ≤ 0 from t on) | last sign change | ΔC@20 | ΔC@50 | ΔC@100 | max ΔC (t) | min ΔC (t) | late gain C[100]−C[50] diff [CI] |
|---|---|---|---|---|---|---|---|---|
| D0 − G4 | 66 | 66 | +0.0215 | +0.0112 | -0.0134 | +0.0289 (14) | -0.0136 (99) | -0.0245 [-0.0493, -0.0029] |
| D0 − G1 | None | 29 | +0.0057 | +0.0141 | +0.0019 | +0.0189 (57) | -0.0044 (25) | -0.0122 [-0.0369, +0.0149] |

Steps with the band above 0 / below 0 (D0 − G4): 3 / 23; (D0 − G1): 13 / 0.

## Unseen-6 (6 scenes)

| Pair | crossover t (ΔC ≤ 0 from t on) | last sign change | ΔC@20 | ΔC@50 | ΔC@100 | max ΔC (t) | min ΔC (t) | late gain C[100]−C[50] diff [CI] |
|---|---|---|---|---|---|---|---|---|
| D0 − G4 | 65 | 65 | +0.0183 | +0.0049 | -0.0152 | +0.0324 (14) | -0.0160 (98) | -0.0201 [-0.0492, +0.0016] |
| D0 − G1 | None | 49 | -0.0061 | +0.0049 | +0.0080 | +0.0213 (3) | -0.0169 (24) | +0.0031 [-0.0190, +0.0289] |

Steps with the band above 0 / below 0 (D0 − G4): 2 / 19; (D0 − G1): 1 / 2.

