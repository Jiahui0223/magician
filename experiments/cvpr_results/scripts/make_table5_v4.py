"""Table 5 v4 (2026-11-16, user request): two columns -- Energy score (predictive distribution of the whole future surface of a view) and
Cov@100 (final coverage), Unseen-6. Same data, rows, alignment and statistics as v2 / v3 (make_table5_v3.build; every value is asserted
against table5_v2_metric_panel.csv). Energy score rather than CRPS: it scores each hypothesis as a whole view (the unit the planner scores),
so it penalises incoherent / floating averaged surfaces; CRPS gives the same ranking (all four rows significantly worse than Seen2Map).
The caption keeps the early-coverage parity statement (Cov@20 / Cov@50 lowest for Seen2Map, n.s.). Outputs fig/table5/table5_v4.*"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_table5_v2 as T  # noqa: E402
import make_table5_v3 as V3  # noqa: E402


def main():
    ep = T.episodes()
    cl = T.closed_loop(ep)
    fx_u = T.fixed(lambda t: t.rsplit("_s", 1)[0].endswith(tuple("_" + s for s in T.U6)))
    assert fx_u["states"] == 211 and fx_u["trajectories"] == 16
    V3.build("table5_v4", [("Energy score", "Energy score ↓", "fx", 2), ("Cov@100", "Cov@100 ↑", "cl", 3)], cl, fx_u,
             "Earlier coverage is comparable across rows (Cov@20 / Cov@50 differences to Seen2Map not significant; full time course in "
             "the supplementary). ",
             dist_note=("Energy score: accuracy of the predicted distribution of the whole future surface seen from a candidate view "
                        "(first-hit depth along every ray into unknown space, compared with the ground truth; lower is better), on held-out "
                        "planning states of unseen scenes. "),
             script="make_table5_v4.py")


if __name__ == "__main__":
    main()
