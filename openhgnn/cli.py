from __future__ import annotations

import argparse
import json

from ._metadata import get_environment_report
from ._version import __version__
from .registry import get_supported_registry_items, validate_registry


def _emit(payload, output_format: str) -> int:
    if output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if isinstance(payload, dict) and "items" in payload:
        for item in payload["items"]:
            print(item)
        return 0
    print(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openhgnn")
    parser.add_argument("--version", action="version", version=f"OpenHGNN {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List supported registry items")
    list_parser.add_argument("kind", choices=["models", "flows", "tasks", "datasets"])
    list_parser.add_argument("--format", choices=["plain", "json"], default="plain")

    env_parser = subparsers.add_parser("env", help="Show current environment against supported matrix")
    env_parser.add_argument("--format", choices=["plain", "json"], default="plain")

    validate_parser = subparsers.add_parser("validate-registry", help="Validate lazy-import registry mappings")
    validate_parser.add_argument("--format", choices=["plain", "json"], default="plain")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return _emit({"kind": args.kind, "items": get_supported_registry_items(args.kind)}, args.format)
    if args.command == "env":
        return _emit(get_environment_report(), args.format)
    if args.command == "validate-registry":
        report = validate_registry()
        _emit(report, args.format)
        return 0 if report["ok"] else 1
    parser.error(f"unsupported command: {args.command}")
    return 2
