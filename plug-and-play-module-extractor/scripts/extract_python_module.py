#!/usr/bin/env python3
"""Extract selected top-level Python modules into a standalone file."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def node_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.Assign):
        names: list[str] = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
        return names[0] if len(names) == 1 else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def is_import(node: ast.AST) -> bool:
    return isinstance(node, (ast.Import, ast.ImportFrom))


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = ast.dump(node.test).replace("'", '"')
    return "__name__" in test and "__main__" in test


def top_level_defs(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        name = node_name(node)
        if name:
            names.append(name)
    return names


def top_level_node_map(tree: ast.Module) -> dict[str, ast.AST]:
    mapping: dict[str, ast.AST] = {}
    for node in tree.body:
        name = node_name(node)
        if name:
            mapping[name] = node
    return mapping


def referenced_names(node: ast.AST) -> set[str]:
    refs: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            refs.add(child.id)
    return refs


def dependency_closure(names: set[str], definitions: dict[str, ast.AST]) -> set[str]:
    selected = set(names)
    queue = list(names)
    while queue:
        current = queue.pop()
        node = definitions.get(current)
        if node is None:
            continue
        for ref in sorted(referenced_names(node) & set(definitions)):
            if ref not in selected:
                selected.add(ref)
                queue.append(ref)
    return selected


def selected_nodes(
    tree: ast.Module,
    names: set[str],
    include_main: bool,
    include_all_helpers: bool,
    include_dependencies: bool,
) -> list[ast.AST]:
    definitions = top_level_node_map(tree)
    root_names = set(names)
    if include_main:
        for node in tree.body:
            if is_main_guard(node):
                root_names.update(referenced_names(node) & set(definitions))
    selected_names = (
        set(definitions)
        if include_all_helpers
        else dependency_closure(root_names, definitions)
        if include_dependencies
        else root_names
    )
    nodes: list[ast.AST] = []
    for node in tree.body:
        name = node_name(node)
        if is_import(node):
            nodes.append(node)
        elif is_main_guard(node):
            if include_main:
                nodes.append(node)
        elif name in selected_names:
            nodes.append(node)
    return nodes


def source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError(f"Could not recover source for node at line {getattr(node, 'lineno', '?')}")
    return segment.rstrip()


def build_module(
    source_path: Path,
    source: str,
    nodes: list[ast.AST],
    names: list[str],
    basis: str,
) -> str:
    title_names = ", ".join(names) if names else "all top-level definitions"
    source_display = source_path.as_posix()
    header = [
        '"""',
        f"Extracted plug-and-play module from: {source_display}",
        f"Selected symbols: {title_names}",
        f"Implementation basis: {basis}",
        "",
        "Review tensor shapes and dependencies against the source paper/code before reuse.",
        '"""',
        "",
    ]
    body: list[str] = []
    seen_imports: set[str] = set()
    for node in nodes:
        segment = source_segment(source, node)
        if is_import(node):
            if segment in seen_imports:
                continue
            seen_imports.add(segment)
        body.append(segment)
        body.append("")
        body.append("")
    return "\n".join(header + body).rstrip() + "\n"


def parse_names(raw_names: list[str] | None) -> list[str]:
    if not raw_names:
        return []
    names: list[str] = []
    for raw in raw_names:
        names.extend(part.strip() for part in raw.split(",") if part.strip())
    return names


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source Python file.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output Python module path.")
    parser.add_argument("--name", action="append", help="Top-level class/function/constant to extract. Repeat or comma-separate.")
    parser.add_argument("--list", action="store_true", help="List top-level extractable symbols and exit.")
    parser.add_argument("--include-main", action="store_true", help="Include the source file's __main__ smoke test.")
    parser.add_argument(
        "--include-all-helpers",
        action="store_true",
        help="Include all top-level definitions from the source file to preserve local dependencies.",
    )
    parser.add_argument(
        "--no-dependencies",
        action="store_true",
        help="Do not auto-include same-file top-level definitions referenced by selected symbols.",
    )
    parser.add_argument(
        "--basis",
        default="source-code-derived",
        choices=["source-code-derived", "paper-derived", "mixed"],
        help="Implementation basis to write in the module header.",
    )
    args = parser.parse_args(argv)

    source_path = args.source.resolve()
    if not source_path.exists():
        parser.error(f"source file not found: {source_path}")
    if source_path.suffix.lower() != ".py":
        parser.error(f"source file is not Python: {source_path}")

    source = source_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(source_path))
    available = top_level_defs(tree)
    names = parse_names(args.name)

    if args.list:
        for name in available:
            print(name)
        return 0

    if not names and not args.include_all_helpers:
        parser.error("provide --name or use --include-all-helpers")

    missing = sorted(set(names) - set(available))
    if missing:
        parser.error(f"symbols not found: {', '.join(missing)}")

    nodes = selected_nodes(
        tree,
        set(names),
        args.include_main,
        args.include_all_helpers,
        include_dependencies=not args.no_dependencies,
    )
    if not nodes:
        parser.error("no nodes selected")

    output = build_module(source_path, source, nodes, names, args.basis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"Wrote extracted module: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
