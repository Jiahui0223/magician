"""CVPR protocol audit -- function-level diff of the evaluation-relevant MAGICIAN code between the official repo
(github shiyao-li/MAGICIAN, local clone at HEAD e697009) and the fork MAGICIAN_clean_9521d69 that our frozen simulator imports.
Extracts each named function / method with ast (source segment) from both copies and reports identical / differs (+ the
unified diff of differing ones). Read-only. Output: protocol_audit/results/eval_function_diff.{json,md}."""
import ast
import difflib
import json
import os

R = "/bsuscratch/jiahuizhang/projects/occupancy"
OFF = f"{R}/seen2scene/work_dirs/exp5_future_view/task0/magician_comparison/external/MAGICIAN"
FORK = f"{R}/MAGICIAN_clean_9521d69"
OUT = f"{R}/seen2scene/experiments/cvpr_results/protocol_audit/results"
TARGETS = {
    "macarons/utility/macarons_utils.py": ["load_scene", "get_scene_gt_surface", "Scene.scene_coverage", "Scene.fill_cells", "Scene.get_covered_points",
                                           "Camera.compute_partial_point_cloud", "Camera.update_camera", "Camera.capture_image", "Camera.get_idx_from_key",
                                           "Camera.get_key_from_idx", "Camera.__init__", "Settings.__init__", "Camera.get_neighboring_poses", "Camera.get_valid_neighbors"],
    "macarons/testers/magician_planning.py": ["setup_test_scene", "setup_test_camera", "load_current_frame_perfect_depth", "apply_perfect_depth_simple"],
    "macarons/utility/magician_utils.py": ["line_segment_mesh_intersection", "get_valid_neighbors", "get_neighboring_poses"],
}


def functions(path):
    src = open(path).read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{sub.name}"] = ast.get_source_segment(src, sub)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rep, md = {}, ["# 评估相关函数：官方 e697009 vs fork 9521d69（`scripts/diff_eval_functions.py` 自动生成）", "",
                   "| 文件 | 函数 | 官方 | fork | 结果 | 差异行数 |", "|---|---|---|---|---|---|"]
    diffs = []
    for f, names in TARGETS.items():
        fo, ff = functions(os.path.join(OFF, f)), functions(os.path.join(FORK, f))
        for n in names:
            a, b = fo.get(n), ff.get(n)
            if a is None and b is None:
                st, nd = "两边都没有该函数", 0
            elif a is None or b is None:
                st, nd = "只在一边存在", 0
            elif a == b:
                st, nd = "逐字相同", 0
            else:
                d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), "official", "fork", lineterm="", n=1))
                nd = sum(1 for x in d if x[:1] in "+-" and not x.startswith(("+++", "---")))
                st = "不同"
                diffs.append((f, n, d))
            rep[f"{f}::{n}"] = {"official": a is not None, "fork": b is not None, "status": st, "changed_lines": nd}
            md.append(f"| `{f}` | `{n}` | {'有' if a else '无'} | {'有' if b else '无'} | {st} | {nd} |")
    md += ["", "## 不同之处（unified diff，官方 → fork）", ""]
    for f, n, d in diffs:
        md += [f"### `{f}` :: `{n}`", "", "```diff"] + d[:120] + ["```", ""]
    json.dump(rep, open(os.path.join(OUT, "eval_function_diff.json"), "w"), indent=1)
    open(os.path.join(OUT, "eval_function_diff.md"), "w").write("\n".join(md) + "\n")
    for k, v in rep.items():
        print(v["status"], v["changed_lines"], k)


if __name__ == "__main__":
    main()
