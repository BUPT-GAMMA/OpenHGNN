from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
OPTIONAL_PREFIXES = ("space4hgnn.",)
DATASET_LIST_NAMES = {
    "common",
    "hgbl_datasets",
    "hgbn_datasets",
    "kg_lp_datasets",
    "kg_sub_datasets",
    "kg_subT_datasets",
    "ohgbl_datasets",
    "ohgbn_datasets",
    "hypergraph_datasets",
}
DATASET_SPECIAL_NAMES = {
    "meirec",
    "NL-100",
    "AdapropT",
    "AdapropI",
    "SACN",
    "LTE",
    "dbook",
    "acm4HGMAE",
    "hgprompt_acm_dblp",
    "acm4FedHGNN",
    "acm4HGA",
    "dblp4HGA",
    "dblp4RHINE",
    "dblp4MHGCN",
    "imdb4MHGCN",
    "alibaba4MHGCN",
    "ogbn-mag",
    "LastFM_KGAT",
    "yelp2018_KGAT",
    "amazon-book_KGAT",
    "NBF_WN18RR",
    "NBF_FB15k-237",
    "DisenKGAT_WN18RR",
    "DisenKGAT_FB15k-237",
}


def _evaluate_module_assignments(path: Path, names: set[str]) -> dict[str, object]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    safe_globals = {"__builtins__": {}}
    safe_locals: dict[str, object] = {"range": range}

    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                values[name] = eval(compile(ast.Expression(node.value), str(path), "eval"), safe_globals, safe_locals)
                safe_locals[name] = values[name]
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name in names and name in safe_locals:
                extra = eval(compile(ast.Expression(node.value), str(path), "eval"), safe_globals, safe_locals)
                values[name] = safe_locals[name] + extra
                safe_locals[name] = values[name]

    return values


def _module_path_exists(module_path: str) -> bool:
    if module_path.startswith(OPTIONAL_PREFIXES):
        return True
    module_relpath = Path(*module_path.split("."))
    return (REPO_ROOT / f"{module_relpath}.py").exists()


def get_supported_models() -> list[str]:
    values = _evaluate_module_assignments(PACKAGE_ROOT / "models" / "__init__.py", {"SUPPORTED_MODELS"})
    return sorted(values["SUPPORTED_MODELS"])


def get_supported_model_modules() -> dict[str, str]:
    values = _evaluate_module_assignments(PACKAGE_ROOT / "models" / "__init__.py", {"SUPPORTED_MODELS"})
    return dict(values["SUPPORTED_MODELS"])


def get_supported_flows() -> list[str]:
    values = _evaluate_module_assignments(PACKAGE_ROOT / "trainerflow" / "__init__.py", {"SUPPORTED_FLOWS"})
    return sorted(values["SUPPORTED_FLOWS"])


def get_supported_tasks() -> list[str]:
    values = _evaluate_module_assignments(PACKAGE_ROOT / "tasks" / "__init__.py", {"SUPPORTED_TASKS"})
    return sorted(values["SUPPORTED_TASKS"])


def get_supported_datasets() -> list[str]:
    values = _evaluate_module_assignments(PACKAGE_ROOT / "dataset" / "__init__.py", DATASET_LIST_NAMES | {"CLASS_DATASETS"})
    datasets = set(DATASET_SPECIAL_NAMES)
    for name in DATASET_LIST_NAMES:
        datasets.update(values.get(name, []))
    datasets.update(values.get("CLASS_DATASETS", {}).keys())
    return sorted(datasets)


def get_supported_registry_items(kind: str) -> list[str]:
    mapping = {
        "models": get_supported_models,
        "flows": get_supported_flows,
        "tasks": get_supported_tasks,
        "datasets": get_supported_datasets,
    }
    return mapping[kind]()


def validate_registry() -> dict:
    checks = {
        "models": _evaluate_module_assignments(PACKAGE_ROOT / "models" / "__init__.py", {"SUPPORTED_MODELS"})["SUPPORTED_MODELS"],
        "flows": _evaluate_module_assignments(PACKAGE_ROOT / "trainerflow" / "__init__.py", {"SUPPORTED_FLOWS"})["SUPPORTED_FLOWS"],
        "tasks": _evaluate_module_assignments(PACKAGE_ROOT / "tasks" / "__init__.py", {"SUPPORTED_TASKS"})["SUPPORTED_TASKS"],
        "datasets": _evaluate_module_assignments(PACKAGE_ROOT / "dataset" / "__init__.py", {"SUPPORTED_DATASETS"})["SUPPORTED_DATASETS"],
    }
    issues = []
    for kind, entries in checks.items():
        for name, module_path in entries.items():
            if not _module_path_exists(module_path):
                issues.append(
                    {
                        "kind": kind,
                        "name": name,
                        "module": module_path,
                        "reason": "missing_module",
                    }
                )
    return {
        "ok": not issues,
        "issues": issues,
        "counts": {kind: len(entries) for kind, entries in checks.items()},
    }
