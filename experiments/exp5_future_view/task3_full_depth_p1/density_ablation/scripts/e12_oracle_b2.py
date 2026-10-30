"""EXP5 Task 3 / E1.2 -- evaluation-side oracle (gains, t*) of the S_B2 states with the frozen ablation S1 scripts
(magician env, 1 GPU). Usage: python e12_oracle_b2.py indoor|outdoor --part i/n"""
import sys

import e12_paths as P

A = P.set_stratum("B2")
if __name__ == "__main__":
    d = sys.argv.pop(1)
    mod = __import__("s1_oracle_in" if d == "indoor" else "s1_oracle_out")
    mod.main()
