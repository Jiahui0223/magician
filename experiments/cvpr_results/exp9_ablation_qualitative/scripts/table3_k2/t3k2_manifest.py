"""T3K2 -- add the V3-T0-K2D{1,2}-N2 rows to manifest/episodes.csv (CPU). Runs the unchanged build_manifest.main() into a temporary directory
(full validation of every canonical record: 101 values, episode file == record, official replay == stored curve, handshake K / depth / noise
offset / checkpoint), then writes manifest/episodes.csv = the existing rows (asserted byte-identical in the rebuild) + the rows of the two new
ids, in build_manifest's method order; provenance.json gets only the two new entries. Methods registered later but never put into the
manifest (e.g. the T4-R1 *-LR1 ids) are NOT added here (a full run_all.sh rebuild would add them). protocol.json is left unchanged.
Usage: python t3k2_manifest.py            (prints the per-id status of the new rows)"""
import csv
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts"))
import cvpr_lib as L  # noqa: E402

NEW = ("V3-T0-K2D1-N2", "V3-T0-K2D2-N2")


def main():
    man = L.MAN
    old = list(csv.reader(open(os.path.join(man, "episodes.csv"), newline="")))
    tmp = tempfile.mkdtemp(prefix="t3k2_manifest_")
    L.MAN = tmp
    import build_manifest as B
    B.main()
    L.MAN = man
    new = list(csv.reader(open(os.path.join(tmp, "episodes.csv"), newline="")))
    assert new[0] == old[0], "header changed"
    keep_methods = {r[0] for r in old[1:]} | set(NEW)
    sn = set(map(tuple, new))
    for r in old[1:]:
        if r[0] in NEW:
            continue                                                      # a previous run of this script: replaced by the fresh rows
        assert tuple(r) in sn, f"existing row changed in the rebuild: {r[:4]}"
    out = [new[0]] + [r for r in new[1:] if r[0] in keep_methods]
    olds = {tuple(r) for r in old[1:] if r[0] not in NEW}
    assert olds == {tuple(r) for r in out[1:] if r[0] not in NEW}, "existing rows differ"
    with open(os.path.join(man, "episodes.csv.tmp"), "w", newline="") as f:
        csv.writer(f).writerows(out)
    os.replace(os.path.join(man, "episodes.csv.tmp"), os.path.join(man, "episodes.csv"))
    pv_new = json.load(open(os.path.join(tmp, "provenance.json")))
    pp = os.path.join(man, "provenance.json")
    pv = json.load(open(pp))
    for m in NEW:
        pv["methods"][m] = pv_new["methods"][m]
    json.dump(pv, open(pp + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(pp + ".tmp", pp)
    shutil.rmtree(tmp)
    st = {}
    for r in out[1:]:
        if r[0] in NEW:
            st.setdefault(r[0], {}).setdefault(r[12], 0)
            st[r[0]][r[12]] += 1
    print(f"manifest/episodes.csv: {len(old) - 1} existing rows unchanged, {sum(r[0] in NEW for r in out[1:])} rows of {NEW}: {st}")
    for m in NEW:
        if pv["methods"][m]["problems"]:
            print(m, "PROBLEMS", pv["methods"][m]["problems"][:5])


if __name__ == "__main__":
    main()
