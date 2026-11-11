| Component (per decision, t = 10..99, 2 trajectories) | Mean (ms) | Share of decision |
|---|---|---|
| Planning geometry (P1 -> planner input) | 87.5 | 3.1% |
| GeoAnchor: refresh incl. 3D VAE encoder (amortised, every 10 steps) | 0.9 | 0.0% |
| RayForge L1: ray construction + projection features | 41.9 | 1.5% |
| RayForge L1: ray conditioning (latent sampling + 1D ray encoder) | 9.1 | 0.3% |
| RayForge L1: flow sampling (K=4, 10 Euler steps) | 302.6 | 10.7% |
| RayForge L1: UNKNOWN-arc depth decoding + DistP1 | 20.2 | 0.7% |
| CDGR L2: ray construction + projection features | 181.1 | 6.4% |
| CDGR L2: ray conditioning | 43.6 | 1.5% |
| CDGR L2: flow sampling (K=4, 10 Euler steps) | 1637.1 | 57.8% |
| CDGR L2: depth decoding + DistP1 | 87.6 | 3.1% |
| CDGR: novel-surface test + weighted coverage (value_cov) | 408.1 | 14.4% |
| CDGR: expansion + path test + union + aggregation + choice | 12.2 | 0.4% |
| Complete decision (sum) | 2831.9 | 100% |
|   of which 3D VAE encoder per call (18 refreshes) | 3.8 | - |
|   of which 1D ray encoder (all layers) | 41.4 | 1.5% |
|   of which depth decoding (all layers) | 42.1 | 1.5% |
| Online map update (not in the decision) | 303.3 | - |
| Candidates: first layer / second layer predicted views / search nodes | 7.1 / 31.6 / 51.0 | - |
