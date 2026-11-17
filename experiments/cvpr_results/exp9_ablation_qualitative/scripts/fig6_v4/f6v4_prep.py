"""EXP9 Fig. 6 v4 -- mesh preparation (magician env, CPU only).  Usage:  $MAGPY -I -B f6v4_prep.py SHARD NSHARDS
Runs scripts/fig6_v2/f6v2_prep.py UNCHANGED (GT domain D, GT-point / mesh alignment checks, texture inputs, transfer statistics) on the
v4 case list work_dirs/.../fig6_v4/v4_cases.csv, with its input / output directories pointed at fig6_v4/.  Cases whose masks are bitwise
identical to v3 (asserted by f6v4_masks.py) reuse the v3 mesh files through symlinks.  SHARD / NSHARDS split the remaining cases
round-robin so several CPU processes can run in parallel."""
import csv
import json
import os
import sys

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/"
EXP9 = REPO + "experiments/cvpr_results/exp9_ablation_qualitative/"
sys.path.insert(0, EXP9 + "scripts/fig6_v2")
import f6v2_prep as P2  # noqa: E402

V3 = REPO + "work_dirs/cvpr_results/exp9_ablation_qualitative/fig6_v3/"
V4 = REPO + "work_dirs/cvpr_results/exp9_ablation_qualitative/fig6_v4/"


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    P2.SEL, P2.MASKS, P2.OUT = V4 + "v4_cases.csv", V4, V4 + "mesh/"
    os.makedirs(P2.OUT, exist_ok=True)
    same = json.load(open(V4 + "f6v4_masks_summary.json"))["v3_cases_bitwise_identical"]
    todo = []
    for r in csv.DictReader(open(P2.SEL)):
        name = P2.case_name(r)
        if same.get(name) and os.path.exists(V3 + "mesh/" + name + ".npz"):
            if shard == 0:
                for suf in (".npz", "_prep.json"):
                    dst = P2.OUT + name + suf
                    if not os.path.lexists(dst):
                        os.symlink(os.path.realpath(V3 + "mesh/" + name + suf), dst)
                print(name, "-> v3 mesh files (symlink)", flush=True)
        elif not os.path.exists(P2.OUT + name + ".npz"):
            todo.append(name)
    todo = todo[shard::nsh]
    print("shard", shard, "cases", todo, flush=True)
    if todo:
        sys.argv = [sys.argv[0]] + todo
        P2.main()


if __name__ == "__main__":
    main()
