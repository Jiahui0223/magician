# Final Coverage of full scenes (All-14 re-evaluated; MAGICIAN App. Table 7 style)

> Final Coverage of full scenes on Macarons++ (style of MAGICIAN App. Table 7). Left: values reported in the MAGICIAN paper (App. Table 7); different runs, starts and evaluation code, shown for reference only. Right: all methods re-evaluated by us under one frozen protocol (official MAGICIAN code, weights and evaluator; settings.json starts 0-4; 100 moves; official coverage, eps = 1.0; Final = C[100]); mean of 5 starts per scene, scene-equal averages. Bold / underline = best / second best within the re-evaluated group only. St. Sofia Church cannot be loaded by the frozen evaluator (loader bug fixed upstream only in 2026-08), so all re-evaluated averages are All-14; the paper's averages are shown both over the same 14 scenes and as reported (15 scenes). The paper's Tab. 7 reports a FisherRF average of 0.786, whereas the mean of its 15 rows is 0.772; we show the row means.

| Scene | Rand. Walk (paper) | SCONE (paper) | MACARONS (paper) | FisherRF (paper) | MAGICIAN (paper) | MAGICIAN (re-eval.) | B2 | V3-D1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dunnottar Castle | 0.225 | 0.527 | 0.820 | 0.809 | 0.975 | **0.973** | 0.925 | <u>0.955</u> |
| Colosseum | 0.272 | 0.755 | 0.794 | 0.757 | 0.872 | **0.870** | <u>0.841</u> | 0.771 |
| Bannerman Castle | 0.240 | 0.722 | 0.834 | 0.801 | 0.917 | **0.896** | <u>0.888</u> | 0.878 |
| Pantheon | 0.309 | 0.610 | 0.796 | 0.444 | 0.842 | 0.740 | <u>0.753</u> | **0.795** |
| Christ the Redeemer | 0.581 | 0.924 | 0.967 | 0.876 | 0.973 | 0.977 | **0.982** | <u>0.981</u> |
| Statue of Liberty | 0.443 | 0.819 | 0.909 | 0.819 | 0.947 | **0.947** | 0.943 | <u>0.945</u> |
| Pisa Cathedral | 0.353 | 0.566 | 0.865 | 0.776 | 0.941 | **0.941** | <u>0.912</u> | <u>0.912</u> |
| Fushimi Castle | 0.449 | 0.844 | 0.853 | 0.814 | 0.931 | **0.940** | 0.879 | <u>0.935</u> |
| Alhambra Palace | 0.162 | 0.473 | 0.775 | 0.615 | 0.852 | 0.849 | **0.863** | <u>0.861</u> |
| Neuschwanstein Castle | 0.223 | 0.444 | 0.551 | 0.582 | 0.848 | 0.797 | **0.866** | <u>0.844</u> |
| Eiffel Tower | 0.541 | 0.856 | 0.915 | 0.827 | 0.923 | <u>0.931</u> | 0.907 | **0.932** |
| Manhattan Bridge | 0.356 | 0.781 | 0.924 | 0.877 | 0.955 | **0.957** | <u>0.954</u> | 0.949 |
| St. Sofia Church | 0.331 | 0.619 | 0.795 | 0.865 | 0.891 | n/a | n/a | n/a |
| Barts | 0.240 | 0.677 | 0.768 | 0.878 | 0.996 | <u>0.996</u> | **0.998** | 0.990 |
| Sestino Museum | 0.141 | 0.430 | 0.713 | 0.842 | 0.924 | 0.878 | <u>0.990</u> | **0.991** |
| **Average (All-14)** | 0.324 | 0.673 | 0.820 | 0.766 | 0.921 | 0.906 | <u>0.907</u> | **0.910** |
| Average (All-15, as reported) | 0.324 | 0.670 | 0.819 | 0.786 | 0.919 | n/a | n/a | n/a |
