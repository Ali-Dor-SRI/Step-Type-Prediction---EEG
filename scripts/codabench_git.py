#!/usr/bin/env python3
"""Check Codabench source tracking, fetch code, or pull with fast-forward only.

Uses only Python's standard library and Git. It never stages, commits, pushes,
stashes, resets, or copies model/data artifacts.
"""

import argparse
import ast
import subprocess
import sys
from pathlib import Path


ARTIFACT_SUFFIXES = {
    ".zip", ".7z", ".tar", ".gz", ".pt", ".pth", ".ckpt", ".joblib",
    ".pkl", ".pickle", ".npy", ".npz", ".parquet", ".edf", ".bdf",
    ".fif", ".rds",
}
SOURCE_SUFFIXES = {".py", ".sh", ".ps1", ".md", ".txt", ".json", ".yaml", ".yml", ".toml"}
SOURCE_FOLDERS = ("analysis", "scripts", "solvers", "datasets", "config", "prompts", "reports")
MAX_SOURCE_BYTES = 10 * 1024 * 1024


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def tracked_paths(root):
    value = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--", "codabench"],
    )
    return {p.decode("utf-8") for p in value.split(b"\0") if p}


def check_source(root):
    tracked = tracked_paths(root)
    problems = []
    parsed = 0
    if not tracked:
        problems.append("No Codabench source is tracked in this checkout.")
    for name in sorted(tracked):
        path = root / name
        if path.suffix.lower() in ARTIFACT_SUFFIXES:
            problems.append(f"Generated artifact tracked in Git: {name}; store it on NAS.")
            continue
        if not path.is_file():
            problems.append(f"Tracked file missing from working tree: {name}")
            continue
        if path.stat().st_size > MAX_SOURCE_BYTES:
            problems.append(f"Tracked file exceeds 10 MiB: {name}; review for NAS storage.")
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=name)
                parsed += 1
            except (SyntaxError, UnicodeError) as error:
                problems.append(f"Invalid Python source {name}: {error}")

    cb = root / "codabench"
    candidates = list(cb.glob("*"))
    for folder in SOURCE_FOLDERS:
        candidates.extend((cb / folder).rglob("*"))
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        name = path.relative_to(root).as_posix()
        if name not in tracked:
            problems.append(f"Source is not tracked: {name}; review and git add it before pushing.")
    return problems, parsed


def fetch(root, remote, branch):
    if remote not in git(root, "remote").splitlines():
        raise RuntimeError(f"Unknown remote {remote!r}.")
    if branch.startswith("-"):
        raise RuntimeError("Branch must be a branch name, not a Git option.")
    subprocess.run(["git", "check-ref-format", "--branch", branch], check=True, capture_output=True)
    source = f"refs/heads/{branch}"
    target = f"refs/remotes/{remote}/{branch}"
    subprocess.run(["git", "-C", str(root), "fetch", "--no-tags", remote, f"{source}:{target}"], check=True)
    print(f"Fetched {remote}/{branch}: {git(root, 'rev-parse', '--short', target)}")
    return target


def pull(root, remote, branch):
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked files have local changes. Commit or save them before pulling.")
    if not git(root, "branch", "--show-current"):
        raise RuntimeError("Detached HEAD: switch to a branch before pulling.")
    target = fetch(root, remote, branch)
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", "HEAD", target],
    )
    if ancestor.returncode == 1:
        already_included = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", target, "HEAD"],
        )
        if already_included.returncode == 0:
            print("This checkout already contains the fetched code; local commits were preserved.")
            return
        raise RuntimeError(
            "This branch has local commits absent from the remote. "
            "Fetch completed; reconcile the branches explicitly. Working files were not changed."
        )
    if ancestor.returncode:
        raise RuntimeError("Could not compare branch ancestry; working files were not changed.")
    subprocess.run(["git", "-C", str(root), "merge", "--ff-only", target], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "fetch", "pull"))
    parser.add_argument("--remote", default="personal")
    parser.add_argument("--branch", default="feat/codabench-track2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if args.action == "check":
            problems, parsed = check_source(root)
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            print(f"Checked {len(tracked_paths(root))} tracked Codabench files; parsed {parsed} Python files.")
            return int(bool(problems))
        if args.action == "fetch":
            fetch(root, args.remote, args.branch)
        else:
            pull(root, args.remote, args.branch)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
