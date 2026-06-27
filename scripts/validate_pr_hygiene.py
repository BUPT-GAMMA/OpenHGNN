#!/usr/bin/env python3
"""Validate pull-request file hygiene for OpenHGNN.

The check intentionally focuses on high-signal repository hygiene issues:
runtime outputs, bundled datasets, ad-hoc root scripts, and local/debug-only
code that should not enter model PRs.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
from pathlib import Path


DATA_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".csv",
    ".dat",
    ".h5",
    ".hdf5",
    ".mat",
    ".npy",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".tgz",
    ".txt",
    ".zip",
}

ARCHIVE_EXTENSIONS = {".gz", ".tar", ".tgz", ".zip"}
ROOT_SCRIPT_PATTERNS = ("main.py", "debug.py", "check.py", "test*.py")
FORBIDDEN_CONTENT = (
    ("breakpoint(", "debugger breakpoint"),
    ("pdb.set_trace", "pdb debugger"),
    ("ipdb.set_trace", "ipdb debugger"),
    ("/home/", "machine-local absolute path"),
    ("cuda:0", "hard-coded CUDA device"),
    ("gpu=2", "hard-coded GPU id"),
)
WARNING_CONTENT = (
    ("TODO", "TODO marker"),
    ("FIXME", "FIXME marker"),
    ("print(", "direct print statement"),
)


def run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return result


def default_base() -> str:
    for candidate in (
        os.environ.get("GITHUB_BASE_SHA"),
        "origin/main",
        "main",
    ):
        if not candidate:
            continue
        result = run_git(["rev-parse", "--verify", candidate], check=False)
        if result.returncode == 0:
            return candidate
    raise SystemExit("Could not infer a base ref. Pass --base explicitly.")


def default_head() -> str:
    return os.environ.get("GITHUB_SHA") or "HEAD"


def changed_files(base: str, head: str) -> list[str]:
    result = run_git(
        ["diff", "--name-only", "--diff-filter=AMR", f"{base}...{head}"],
        check=False,
    )
    if result.returncode != 0:
        result = run_git(
            ["diff", "--name-only", "--diff-filter=AMR", base, head],
            check=True,
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def path_errors(path: str) -> list[str]:
    errors: list[str] = []
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    suffixes = Path(normalized).suffixes
    suffix = Path(normalized).suffix.lower()

    if normalized.startswith("openhgnn/output/"):
        errors.append("runtime output/docs under openhgnn/output are not allowed")

    if Path(normalized).name.startswith("events.out.tfevents"):
        errors.append("TensorBoard event files are runtime artifacts")

    if len(parts) == 1 and any(fnmatch.fnmatch(normalized, pat) for pat in ROOT_SCRIPT_PATTERNS):
        errors.append("ad-hoc root scripts/tests must live under examples/, scripts/, or tests/")

    if normalized.startswith("openhgnn/dataset/"):
        if suffix in DATA_EXTENSIONS or any(ext in ARCHIVE_EXTENSIONS for ext in suffixes):
            errors.append("dataset payloads must be downloaded/cached, not committed")
        if "/data/" in normalized and suffix in DATA_EXTENSIONS:
            errors.append("raw dataset data directories must not be committed")

    if normalized.startswith("openhgnn/") and any(ext in ARCHIVE_EXTENSIONS for ext in suffixes):
        errors.append("archives must not be committed under openhgnn/")

    if normalized.startswith("openhgnn/") and suffix in {".ckpt", ".pth", ".pt", ".npy", ".npz", ".pkl"}:
        errors.append("model/data artifacts must not be committed under openhgnn/")

    return errors


def file_text_at(ref: str, path: str) -> str | None:
    result = run_git(["show", f"{ref}:{path}"], check=False)
    if result.returncode != 0:
        return None
    try:
        result.stdout.encode("utf-8").decode("utf-8")
    except UnicodeDecodeError:
        return None
    return result.stdout


def content_findings(path: str, head: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not path.endswith(".py"):
        return errors, warnings

    text = file_text_at(head, path)
    if text is None:
        return errors, warnings

    for needle, description in FORBIDDEN_CONTENT:
        if needle in text:
            errors.append(f"contains {description}: {needle}")

    for needle, description in WARNING_CONTENT:
        if needle in text:
            warnings.append(f"contains {description}: {needle}")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=default_base())
    parser.add_argument("--head", default=default_head())
    args = parser.parse_args()

    files = changed_files(args.base, args.head)
    all_errors: list[tuple[str, str]] = []
    all_warnings: list[tuple[str, str]] = []

    for path in files:
        for message in path_errors(path):
            all_errors.append((path, message))
        content_errors, content_warnings = content_findings(path, args.head)
        for message in content_errors:
            all_errors.append((path, message))
        for message in content_warnings:
            all_warnings.append((path, message))

    print(f"Checked {len(files)} changed files for PR hygiene.")
    if all_warnings:
        print("\nWarnings:")
        for path, message in all_warnings:
            print(f"  - {path}: {message}")

    if all_errors:
        print("\nErrors:")
        for path, message in all_errors:
            print(f"  - {path}: {message}")
        return 1

    print("PR hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
