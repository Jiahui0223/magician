# AUC of full scenes (All-14 re-evaluated; MAGICIAN App. Table 6 style)

> AUC of full scenes on Macarons++ (style of MAGICIAN App. Table 6). Left: values reported in the MAGICIAN paper (App. Table 6); different runs, starts and evaluation code, shown for reference only. Right: all methods re-evaluated by us under one frozen protocol (official MAGICIAN code, weights and evaluator; settings.json starts 0-4; 100 moves; official coverage, eps = 1.0; AUC@100 = trapz(C[0..100]) / 100); mean of 5 starts per scene, scene-equal averages. Bold / underline = best / second best within the re-evaluated group only. St. Sofia Church cannot be loaded by the frozen evaluator (loader bug fixed upstream only in 2026-08), so all re-evaluated averages are All-14; the paper's averages are shown both over the same 14 scenes and as reported (15 scenes).

| Scene | Rand. Walk (paper) | SCONE (paper) | MACARONS (paper) | FisherRF (paper) | MAGICIAN (paper) | MAGICIAN (re-eval.) | B2 | V3-D1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dunnottar Castle | 0.149 | 0.366 | 0.618 | 0.500 | 0.745 | 0.721 | <u>0.731</u> | **0.765** |
| Colosseum | 0.219 | 0.589 | 0.656 | 0.551 | 0.704 | **0.710** | <u>0.700</u> | 0.649 |
| Bannerman Castle | 0.192 | 0.559 | 0.575 | 0.595 | 0.761 | **0.764** | 0.724 | <u>0.749</u> |
| Pantheon | 0.198 | 0.465 | 0.601 | 0.270 | 0.644 | 0.561 | <u>0.582</u> | **0.609** |
| Christ the Redeemer | 0.439 | 0.772 | 0.859 | 0.727 | 0.793 | 0.784 | **0.883** | <u>0.874</u> |
| Statue of Liberty | 0.323 | 0.632 | 0.711 | 0.553 | 0.797 | 0.792 | <u>0.798</u> | **0.813** |
| Pisa Cathedral | 0.290 | 0.486 | 0.678 | 0.486 | 0.723 | 0.721 | **0.751** | <u>0.750</u> |
| Fushimi Castle | 0.279 | 0.689 | 0.718 | 0.565 | 0.766 | <u>0.781</u> | 0.742 | **0.811** |
| Alhambra Palace | 0.126 | 0.369 | 0.567 | 0.462 | 0.631 | 0.616 | <u>0.643</u> | **0.664** |
| Neuschwanstein Castle | 0.184 | 0.325 | 0.452 | 0.375 | 0.608 | 0.565 | **0.665** | <u>0.647</u> |
| Eiffel Tower | 0.333 | 0.683 | 0.709 | 0.616 | 0.754 | 0.766 | <u>0.768</u> | **0.808** |
| Manhattan Bridge | 0.258 | 0.632 | 0.750 | 0.637 | 0.705 | 0.732 | <u>0.796</u> | **0.811** |
| St. Sofia Church | 0.280 | 0.532 | 0.621 | 0.608 | 0.710 | n/a | n/a | n/a |
| Barts | 0.214 | 0.551 | 0.660 | 0.673 | 0.831 | 0.828 | <u>0.875</u> | **0.897** |
| Sestino Museum | 0.132 | 0.367 | 0.537 | 0.571 | 0.637 | 0.611 | <u>0.737</u> | **0.864** |
| **Average (All-14)** | 0.238 | 0.535 | 0.649 | 0.541 | 0.721 | 0.711 | <u>0.742</u> | **0.765** |
| Average (All-15, as reported) | 0.241 | 0.534 | 0.647 | 0.546 | 0.721 | n/a | n/a | n/a |
