#!/usr/bin/env python3
"""Find likely plug-and-play module candidates in a Python source tree."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    "site-packages",
}

NAME_KEYWORDS = {
    "attention",
    "attn",
    "adapter",
    "block",
    "module",
    "layer",
    "loss",
    "norm",
    "filter",
    "fusion",
    "selector",
    "token",
    "prompt",
    "mamba",
    "conv",
    "point",
    "temporal",
    "spatial",
    "revin",
}

BASE_KEYWORDS = {
    "module",
    "nn.module",
    "torch.nn.module",
    "layer",
    "keras.layers.layer",
}


@dataclass
class Candidate:
    path: str
    kind: str
    name: str
    line: int
    reason: str
    bases: list[str]


@dataclass
class FileSummary:
    path: str
    imports: list[str]
    candidates: list[Candidate]
    has_main: bool
    parse_error: str | None = None


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return dotted_name(node.value)
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    return ""


def normalize(value: str) -> str:
    return value.replace("_", "").replace("-", "").lower()


def import_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                names.append(f"{module}.{alias.name}" if module else alias.name)
    return sorted(set(names))


def has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.dump(node.test).replace("'", '"')
        if "__name__" in test and "__main__" in test:
            return True
    return False


def candidate_reason(name: str, bases: list[str], kind: str) -> str | None:
    lowered = name.lower()
    compact = normalize(name)
    matched_names = [word for word in NAME_KEYWORDS if word in lowered or word in compact]
    matched_bases = [base for base in bases if base.lower() in BASE_KEYWORDS or base.lower().endswith(".module")]

    if matched_bases and matched_names:
        return f"{kind} inherits module-like base and name matches {', '.join(sorted(matched_names))}"
    if matched_bases:
        return f"{kind} inherits module-like base"
    if matched_names:
        return f"{kind} name matches {', '.join(sorted(matched_names))}"
    return None


def scan_file(path: Path, root: Path) -> FileSummary:
    relative = str(path.relative_to(root))
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8", errors="replace")

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return FileSummary(
            path=relative,
            imports=[],
            candidates=[],
            has_main=False,
            parse_error=f"{exc.msg} at line {exc.lineno}",
        )

    candidates: list[Candidate] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = [dotted_name(base) for base in node.bases if dotted_name(base)]
            reason = candidate_reason(node.name, bases, "class")
            if reason:
                candidates.append(Candidate(relative, "class", node.name, node.lineno, reason, bases))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            reason = candidate_reason(node.name, [], "function")
            if reason:
                candidates.append(Candidate(relative, "function", node.name, node.lineno, reason, []))

    return FileSummary(
        path=relative,
        imports=import_names(tree),
        candidates=sorted(candidates, key=lambda item: (item.line, item.kind, item.name)),
        has_main=has_main_guard(tree),
    )


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def render_text(summaries: list[FileSummary]) -> str:
    lines: list[str] = []
    for summary in summaries:
        if not summary.candidates and not summary.parse_error and not summary.has_main:
            continue
        lines.append(f"{summary.path}")
        if summary.parse_error:
            lines.append(f"  parse_error: {summary.parse_error}")
        if summary.imports:
            lines.append(f"  imports: {', '.join(summary.imports[:12])}")
        if summary.has_main:
            lines.append("  has_main: yes")
        for candidate in summary.candidates:
            base_text = f" bases={candidate.bases}" if candidate.bases else ""
            lines.append(f"  {candidate.kind} {candidate.name}:{candidate.line}{base_text}")
            lines.append(f"    reason: {candidate.reason}")
        lines.append("")
    return "\n".join(lines).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Python source root to scan.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.exists():
        parser.error(f"source root not found: {root}")

    summaries = [scan_file(path, root) for path in sorted(iter_python_files(root))]
    if args.json:
        print(json.dumps([asdict(summary) for summary in summaries], ensure_ascii=False, indent=2))
    else:
        text = render_text(summaries)
        print(text if text else "No obvious Python module candidates found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
