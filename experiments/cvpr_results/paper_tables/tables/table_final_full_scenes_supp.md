# Final Coverage of full scenes (All-14 re-evaluated; MAGICIAN App. Table 7 style)

> Final Coverage of full scenes on Macarons++ (style of MAGICIAN App. Table 7). Left: values reported in the MAGICIAN paper (App. Table 7); different runs, starts and evaluation code, shown for reference only. Right: all methods re-evaluated by us under one frozen protocol (official MAGICIAN code, weights and evaluator; settings.json starts 0-4; 100 moves; official coverage, eps = 1.0; Final = C[100]); mean of 5 starts per scene, scene-equal averages. Bold / underline = best / second best within the re-evaluated group only. St. Sofia Church cannot be loaded by the frozen evaluator (loader bug fixed upstream only in 2026-08), so all re-evaluated averages are All-14; the paper's averages are shown both over the same 14 scenes and as reported (15 scenes). The paper's Tab. 7 reports a FisherRF average of 0.786, whereas the mean of its 15 rows is 0.772; we show the row means.

| Scene | Rand. Walk (paper) | SCONE (paper) | MACARONS (paper) | FisherRF (paper) | MAGICIAN (paper) | MAGICIAN (re-eval.) | MAGICIAN-Greedy (re-eval.) | B0 | B2 | V3-T1 (H0) | V3-D1 | V3-T0-D1 (no x-attn) | V3-YBEST-H3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dunnottar Castle | 0.225 | 0.527 | 0.820 | 0.809 | 0.975 | **0.973** | 0.966 | 0.944 | 0.925 | 0.941 | 0.955 | <u>0.969</u> | 0.968 |
| Colosseum | 0.272 | 0.755 | 0.794 | 0.757 | 0.872 | **0.870** | **0.870** | 0.790 | <u>0.841</u> | 0.824 | 0.771 | 0.720 | 0.705 |
| Bannerman Castle | 0.240 | 0.722 | 0.834 | 0.801 | 0.917 | <u>0.896</u> | **0.900** | 0.883 | 0.888 | 0.893 | 0.878 | 0.888 | **0.900** |
| Pantheon | 0.309 | 0.610 | 0.796 | 0.444 | 0.842 | 0.740 | 0.790 | 0.776 | 0.753 | 0.758 | <u>0.795</u> | **0.805** | 0.788 |
| Christ the Redeemer | 0.581 | 0.924 | 0.967 | 0.876 | 0.973 | 0.977 | 0.975 | **0.983** | <u>0.982</u> | **0.983** | 0.981 | 0.979 | 0.981 |
| Statue of Liberty | 0.443 | 0.819 | 0.909 | 0.819 | 0.947 | <u>0.947</u> | <u>0.947</u> | 0.939 | 0.943 | 0.943 | 0.945 | 0.945 | **0.950** |
| Pisa Cathedral | 0.353 | 0.566 | 0.865 | 0.776 | 0.941 | **0.941** | 0.927 | 0.914 | 0.912 | 0.932 | 0.912 | 0.926 | <u>0.936</u> |
| Fushimi Castle | 0.449 | 0.844 | 0.853 | 0.814 | 0.931 | **0.940** | 0.934 | <u>0.937</u> | 0.879 | 0.925 | 0.935 | 0.935 | 0.935 |
| Alhambra Palace | 0.162 | 0.473 | 0.775 | 0.615 | 0.852 | 0.849 | 0.832 | 0.860 | **0.863** | 0.852 | <u>0.861</u> | 0.853 | 0.850 |
| Neuschwanstein Castle | 0.223 | 0.444 | 0.551 | 0.582 | 0.848 | 0.797 | <u>0.864</u> | 0.758 | **0.866** | 0.742 | 0.844 | 0.860 | 0.861 |
| Eiffel Tower | 0.541 | 0.856 | 0.915 | 0.827 | 0.923 | 0.931 | <u>0.933</u> | 0.899 | 0.907 | 0.916 | 0.932 | 0.930 | **0.939** |
| Manhattan Bridge | 0.356 | 0.781 | 0.924 | 0.877 | 0.955 | **0.957** | 0.948 | 0.945 | <u>0.954</u> | 0.950 | 0.949 | 0.952 | 0.944 |
| St. Sofia Church | 0.331 | 0.619 | 0.795 | 0.865 | 0.891 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Barts | 0.240 | 0.677 | 0.768 | 0.878 | 0.996 | 0.996 | 0.989 | 0.995 | <u>0.998</u> | <u>0.998</u> | 0.990 | <u>0.998</u> | **0.999** |
| Sestino Museum | 0.141 | 0.430 | 0.713 | 0.842 | 0.924 | 0.878 | 0.747 | 0.938 | 0.990 | 0.987 | <u>0.991</u> | 0.990 | **0.992** |
| **Average (All-14)** | 0.324 | 0.673 | 0.820 | 0.766 | 0.921 | 0.906 | 0.901 | 0.897 | 0.907 | 0.903 | <u>0.910</u> | **0.911** | **0.911** |
| Average (All-15, as reported) | 0.324 | 0.670 | 0.819 | 0.786 | 0.919 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
