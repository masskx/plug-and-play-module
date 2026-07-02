#!/usr/bin/env python3
"""Compare the public top-level interface of two Python module files."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModuleInterface:
    path: str
    imports: list[str]
    classes: list[str]
    functions: list[str]
    constants: list[str]
    has_main: bool


@dataclass
class InterfaceComparison:
    reference: ModuleInterface
    candidate: ModuleInterface
    missing_imports: list[str]
    missing_classes: list[str]
    missing_functions: list[str]
    missing_constants: list[str]
    main_guard_matches: bool
    status: str


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def import_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.extend(f"{module}.{alias.name}" if module else alias.name for alias in node.names)
    return sorted(set(names))


def has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = ast.dump(node.test).replace("'", '"')
        if "__name__" in test and "__main__" in test:
            return True
    return False


def assigned_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Assign):
        values: list[str] = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                values.append(target.id)
        return values
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def module_interface(path: Path) -> ModuleInterface:
    source = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(path))
    classes: list[str] = []
    functions: list[str] = []
    constants: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        else:
            constants.extend(assigned_names(node))

    return ModuleInterface(
        path=str(path),
        imports=import_names(tree),
        classes=classes,
        functions=functions,
        constants=constants,
        has_main=has_main_guard(tree),
    )


def missing(reference: list[str], candidate: list[str]) -> list[str]:
    candidate_set = set(candidate)
    return [item for item in reference if item not in candidate_set]


def compare(reference_path: Path, candidate_path: Path) -> InterfaceComparison:
    reference = module_interface(reference_path)
    candidate = module_interface(candidate_path)
    missing_imports = missing(reference.imports, candidate.imports)
    missing_classes = missing(reference.classes, candidate.classes)
    missing_functions = missing(reference.functions, candidate.functions)
    missing_constants = missing(reference.constants, candidate.constants)
    main_guard_matches = reference.has_main == candidate.has_main
    status = (
        "passed"
        if not missing_imports
        and not missing_classes
        and not missing_functions
        and not missing_constants
        and main_guard_matches
        else "failed"
    )
    return InterfaceComparison(
        reference=reference,
        candidate=candidate,
        missing_imports=missing_imports,
        missing_classes=missing_classes,
        missing_functions=missing_functions,
        missing_constants=missing_constants,
        main_guard_matches=main_guard_matches,
        status=status,
    )


def render_text(result: InterfaceComparison) -> str:
    lines = [
        f"status: {result.status}",
        f"reference: {result.reference.path}",
        f"candidate: {result.candidate.path}",
        f"missing_imports: {', '.join(result.missing_imports) if result.missing_imports else 'none'}",
        f"missing_classes: {', '.join(result.missing_classes) if result.missing_classes else 'none'}",
        f"missing_functions: {', '.join(result.missing_functions) if result.missing_functions else 'none'}",
        f"missing_constants: {', '.join(result.missing_constants) if result.missing_constants else 'none'}",
        f"main_guard_matches: {str(result.main_guard_matches).lower()}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference Python module.")
    parser.add_argument("candidate", type=Path, help="Extracted/generated Python module.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    if not args.reference.exists():
        parser.error(f"reference not found: {args.reference}")
    if not args.candidate.exists():
        parser.error(f"candidate not found: {args.candidate}")

    result = compare(args.reference.resolve(), args.candidate.resolve())
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(render_text(result))
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
