#!/usr/bin/env python3
"""Inspect OpenHGNN registry sizes without importing heavy modules."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def literal_mapping(path: Path, names: tuple[str, ...]) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except Exception as exc:  # pragma: no cover - diagnostic path
                    out[target.id] = {"error": str(exc)}
    return out


def main() -> int:
    items = {
        "models": literal_mapping(ROOT / "openhgnn/models/__init__.py", ("SUPPORTED_MODELS",)).get("SUPPORTED_MODELS", {}),
        "flows": literal_mapping(ROOT / "openhgnn/trainerflow/__init__.py", ("SUPPORTED_FLOWS",)).get("SUPPORTED_FLOWS", {}),
        "tasks": literal_mapping(ROOT / "openhgnn/tasks/__init__.py", ("SUPPORTED_TASKS",)).get("SUPPORTED_TASKS", {}),
        "datasets": literal_mapping(ROOT / "openhgnn/dataset/__init__.py", ("SUPPORTED_DATASETS",)).get("SUPPORTED_DATASETS", {}),
    }
    payload = {
        "root": str(ROOT),
        "counts": {key: len(value) if isinstance(value, dict) else None for key, value in items.items()},
        "items": {key: sorted(value) if isinstance(value, dict) else value for key, value in items.items()},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
