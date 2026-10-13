"""Record the code version of a run (exp1-task5).

Writes `git_state.txt` into the run directory: HEAD commit, branch, `git status --short`
and the full uncommitted diff. Any run can then be traced to its exact code.
"""
import os
import subprocess

from seen2scene import PROJECT_ROOT


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                              timeout=60).stdout
    except Exception as e:  # git missing or not a repository: record why, never fail the run
        return f"<git {' '.join(args)} failed: {e!r}>\n"


def write_git_state(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "git_state.txt")
    with open(path, "w") as f:
        f.write("# HEAD\n" + _git("rev-parse", "HEAD"))
        f.write("# branch\n" + _git("rev-parse", "--abbrev-ref", "HEAD"))
        f.write("# git status --short\n" + _git("status", "--short"))
        f.write("# git diff HEAD\n" + _git("diff", "HEAD"))
    return path
