from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

from ._version import __version__


SUPPORTED_PYTHON = {
    "min": "3.10",
    "max": "3.12",
}

RECOMMENDED_ENVIRONMENTS = [
    {
        "name": "py311-torch241-dgl240-cu121",
        "python": "3.11",
        "torch": "2.4.1",
        "dgl": "2.4.0+cu121",
        "cuda": "12.1",
    },
    {
        "name": "py310-torch231-dgl221",
        "python": "3.10",
        "torch": "2.3.1",
        "dgl": "2.2.1",
        "cuda": "12.1",
    },
]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def get_environment_report() -> dict:
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_ok = (3, 10) <= sys.version_info[:2] <= (3, 12)
    return {
        "package_version": __version__,
        "python": {
            "current": current_python,
            "supported": f"{SUPPORTED_PYTHON['min']}-{SUPPORTED_PYTHON['max']}",
            "ok": python_ok,
        },
        "dependencies": {
            "torch": _package_version("torch"),
            "dgl": _package_version("dgl"),
        },
        "recommended_envs": RECOMMENDED_ENVIRONMENTS,
    }
