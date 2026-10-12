#!/usr/bin/env python3
"""Extract selected top-level Python symbols into a standalone file.

Same-file dependencies are always followed. Dependencies that live in *other files of
the same repository* (``from .ops.act import HSwish``, ``from pkg.utils import f``) are
inlined too, so the result runs without the rest of the project. Third-party imports are
kept as imports; unresolved project-local names are reported instead of silently left
broken.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import sys
from dataclasses import dataclass, field
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


def is_plain_import(node: ast.AST) -> bool:
    return isinstance(node, (ast.Import, ast.ImportFrom))


def is_import_guard(node: ast.AST) -> bool:
    """``try: import x / except ImportError: ...`` blocks made only of imports."""
    if not isinstance(node, ast.Try):
        return False
    return bool(node.body) and all(is_plain_import(child) for child in node.body)


def is_import(node: ast.AST) -> bool:
    return is_plain_import(node) or is_import_guard(node)


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


REGISTRY_HINTS = ("register", "build")


def is_registry_decorator(decorator: ast.AST) -> bool:
    """Framework registry decorators (``@MODELS.register_module()``) do not belong in a standalone file."""
    text = ast.unparse(decorator).lower()
    return any(hint in text for hint in REGISTRY_HINTS)


def without_registry_decorators(node: ast.AST) -> ast.AST:
    decorators = getattr(node, "decorator_list", None)
    if not decorators or not any(is_registry_decorator(dec) for dec in decorators):
        return node
    clone = copy.copy(node)
    clone.decorator_list = [dec for dec in decorators if not is_registry_decorator(dec)]  # type: ignore[attr-defined]
    return clone


def referenced_names(node: ast.AST) -> set[str]:
    refs: set[str] = set()
    for child in ast.walk(without_registry_decorators(node)):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            refs.add(child.id)
    return refs


def import_bound_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            for alias in child.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(child, ast.ImportFrom):
            for alias in child.names:
                names.add(alias.asname or alias.name)
    return names


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


def source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError(f"Could not recover source for node at line {getattr(node, 'lineno', '?')}")
    kept = [
        dec
        for dec in getattr(node, "decorator_list", [])
        if not is_registry_decorator(dec)
    ]
    if kept:
        decorators = [f"@{ast.get_source_segment(source, dec) or ast.unparse(dec)}" for dec in kept]
        segment = "\n".join(decorators + [segment])
    return segment.rstrip()


# --------------------------------------------------------------------------------------
# Cross-file resolution
# --------------------------------------------------------------------------------------


@dataclass
class LocalRef:
    """A name imported from another file of the same repository."""

    target: Path
    original: str  # name inside the target file ("*" is never resolved)


@dataclass
class ParsedFile:
    path: Path
    source: str
    tree: ast.Module
    defs: dict[str, ast.AST]
    local_refs: dict[str, LocalRef] = field(default_factory=dict)
    unresolved_local: dict[str, str] = field(default_factory=dict)  # bound name -> import text


def module_candidates(base: Path, dotted: str) -> list[Path]:
    rel = Path(*dotted.split(".")) if dotted else Path()
    return [base / rel.with_suffix(".py"), base / rel / "__init__.py"] if dotted else [base / "__init__.py"]


def resolve_import_file(root: Path, current: Path, level: int, module: str | None) -> Path | None:
    dotted = module or ""
    bases: list[Path] = []
    if level > 0:
        base = current.parent
        for _ in range(level - 1):
            base = base.parent
        bases.append(base)
    else:
        bases.extend([root, current.parent])
    for base in bases:
        for candidate in module_candidates(base, dotted):
            if candidate.is_file():
                return candidate.resolve()
    return None


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.files: dict[Path, ParsedFile] = {}
        self.warnings: list[str] = []

    def load(self, path: Path) -> ParsedFile:
        path = path.resolve()
        cached = self.files.get(path)
        if cached:
            return cached
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
        parsed = ParsedFile(path=path, source=source, tree=tree, defs=top_level_node_map(tree))
        self.files[path] = parsed
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                self._register_from_import(parsed, node)
            elif isinstance(node, ast.Import):
                self._register_plain_import(parsed, node)
        return parsed

    def _register_from_import(self, parsed: ParsedFile, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module and node.module.split(".")[0] in {"__future__"}:
            return
        target = resolve_import_file(self.root, parsed.path, node.level, node.module)
        text = ast.unparse(node)
        if target is None:
            if node.level > 0:
                for alias in node.names:
                    parsed.unresolved_local[alias.asname or alias.name] = text
            return
        # `from . import sibling` may name a module file rather than a symbol.
        for alias in node.names:
            bound = alias.asname or alias.name
            if alias.name == "*":
                parsed.unresolved_local[bound] = text
                continue
            parsed.local_refs[bound] = LocalRef(target=target, original=alias.name)

    def _register_plain_import(self, parsed: ParsedFile, node: ast.Import) -> None:
        for alias in node.names:
            if resolve_import_file(self.root, parsed.path, 0, alias.name):
                parsed.unresolved_local[(alias.asname or alias.name).split(".")[0]] = ast.unparse(node)


@dataclass
class Plan:
    """Ordered code pieces to emit, plus the bookkeeping needed to explain them."""

    file_order: list[Path] = field(default_factory=list)
    selected: dict[Path, set[str]] = field(default_factory=dict)
    aliases: dict[Path, dict[str, str]] = field(default_factory=dict)  # file -> {alias: original}
    unresolved: list[str] = field(default_factory=list)


def build_plan(
    workspace: Workspace,
    main_path: Path,
    names: set[str],
    include_main: bool,
    include_all_helpers: bool,
    include_dependencies: bool,
    resolve_local: bool,
) -> Plan:
    plan = Plan()
    main = workspace.load(main_path)
    root_names = set(names)
    if include_main:
        for node in main.tree.body:
            if is_main_guard(node):
                root_names.update(referenced_names(node) & set(main.defs))
    visiting: set[tuple[Path, str]] = set()
    done: set[tuple[Path, str]] = set()

    def finish_file(path: Path) -> None:
        if path not in plan.file_order:
            plan.file_order.append(path)

    def need(path: Path, name: str, via: str | None = None) -> None:
        key = (path, name)
        if key in done or key in visiting:
            return
        parsed = workspace.load(path)
        visiting.add(key)
        if name in parsed.defs:
            if include_dependencies or path != main.path:
                for ref in sorted(referenced_names(parsed.defs[name])):
                    follow(parsed, ref, name)
            plan.selected.setdefault(path, set()).add(name)
            finish_file(path)
        elif name in parsed.local_refs:
            ref = parsed.local_refs[name]
            need(ref.target, ref.original, via=name)
        else:
            plan.unresolved.append(f"{path.name}: `{name}` is not a top-level definition in this file")
        visiting.discard(key)
        done.add(key)

    def follow(parsed: ParsedFile, ref: str, owner: str) -> None:
        if ref in parsed.defs and ref != owner:
            need(parsed.path, ref)
        elif ref in parsed.local_refs and resolve_local:
            target = parsed.local_refs[ref]
            need(target.target, target.original)
            if target.original != ref:
                plan.aliases.setdefault(parsed.path, {})[ref] = target.original
        elif ref in parsed.unresolved_local and ref not in builtins.__dict__:
            plan.unresolved.append(
                f"{parsed.path.name}: `{ref}` comes from unresolvable project import `{parsed.unresolved_local[ref]}`"
            )

    if include_all_helpers:
        for name in main.defs:
            need(main.path, name)
    else:
        for name in sorted(root_names):
            need(main.path, name)
    return plan


def local_import_status(parsed: ParsedFile, node: ast.AST, resolve_local: bool) -> str:
    """'keep' for third-party imports, 'drop' for project-local ones."""
    if not resolve_local:
        return "keep"
    if isinstance(node, ast.ImportFrom):
        if node.level > 0:
            return "drop"
        if node.module and node.module.split(".")[0] == "__future__":
            return "keep"
        names = [alias.asname or alias.name for alias in node.names]
        if any(name in parsed.local_refs or name in parsed.unresolved_local for name in names):
            return "drop"
        return "keep"
    if isinstance(node, ast.Import):
        bound = {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        return "drop" if bound & set(parsed.unresolved_local) else "keep"
    if isinstance(node, ast.Try):
        return "keep"
    return "keep"


def render(
    workspace: Workspace,
    plan: Plan,
    main_path: Path,
    names: list[str],
    include_main: bool,
    include_all_helpers: bool,
    prune_imports: bool,
    resolve_local: bool,
    basis: str,
) -> tuple[str, list[str]]:
    main = workspace.load(main_path)
    notes: list[str] = []
    pieces: list[tuple[Path, str]] = []
    import_texts: list[str] = []
    seen_imports: set[str] = set()
    seen_defs: dict[str, tuple[Path, str]] = {}

    ordered_files = [path for path in plan.file_order if path != main.path] + [main.path]

    def imports_for(parsed: ParsedFile, used_refs: set[str], is_main_file: bool) -> list[str]:
        out: list[str] = []
        for node in parsed.tree.body:
            if not is_import(node):
                continue
            if local_import_status(parsed, node, resolve_local) == "drop":
                continue
            if (not is_main_file) or prune_imports:
                if not (import_bound_names(node) & used_refs):
                    continue
            out.append(source_segment(parsed.source, node))
        return out

    for path in ordered_files:
        parsed = workspace.load(path)
        is_main_file = path == main.path
        selected = plan.selected.get(path, set())
        file_nodes = [node for node in parsed.tree.body if node_name(node) in selected]
        used_refs: set[str] = set()
        for node in file_nodes:
            used_refs |= referenced_names(node)
        if is_main_file and include_main:
            for node in parsed.tree.body:
                if is_main_guard(node):
                    used_refs |= referenced_names(node)
        for text in imports_for(parsed, used_refs, is_main_file):
            if text not in seen_imports:
                seen_imports.add(text)
                import_texts.append(text)
        for alias, original in sorted(plan.aliases.get(path, {}).items()):
            if alias != original:
                pieces.append((path, f"{alias} = {original}"))
        for node in file_nodes:
            name = node_name(node)
            text = source_segment(parsed.source, node)
            if name in seen_defs and seen_defs[name][1] != text:
                notes.append(
                    f"name collision: `{name}` defined in both {seen_defs[name][0].name} and {path.name}; kept the first"
                )
                continue
            seen_defs[name] = (path, text)
            pieces.append((path, text))
        if is_main_file and include_main:
            for node in parsed.tree.body:
                if is_main_guard(node):
                    pieces.append((path, source_segment(parsed.source, node)))

    title_names = ", ".join(names) if names else "all top-level definitions"
    header = [
        '"""',
        f"Extracted plug-and-play module from: {_display(workspace, main_path)}",
        f"Selected symbols: {title_names}",
    ]
    inlined = sorted(path for path in plan.file_order if path != main.path)
    if inlined:
        header.append("Inlined project-local dependencies from: " + ", ".join(_display(workspace, p) for p in inlined))
    header += [
        f"Implementation basis: {basis}",
        "",
        "Review tensor shapes and dependencies against the source paper/code before reuse.",
        '"""',
        "",
    ]
    body = list(import_texts)
    if body:
        body += ["", ""]
    for _, text in pieces:
        body += [text, "", ""]
    for line in plan.unresolved:
        notes.append(f"unresolved: {line}")
    return "\n".join(header + body).rstrip() + "\n", notes


def _display(workspace: Workspace, path: Path) -> str:
    try:
        return path.relative_to(workspace.root).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------------------
# Generated smoke test (real repositories almost never ship one)
# --------------------------------------------------------------------------------------

DTYPE_SUFFIXES = {"float", "half", "long", "int", "bool"}


def parse_init_kwargs(init: str | None) -> dict[str, object]:
    """'dim=64,heads=(2,4),act="relu"' -> {'dim': 64, 'heads': (2, 4), 'act': 'relu'} (literals only)."""
    if not init:
        return {}
    call = ast.parse(f"dict({init})", mode="eval").body
    if not isinstance(call, ast.Call) or call.args:
        raise ValueError("--init must be keyword arguments such as 'channels=64,reduction=16'")
    return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}


def parse_input_specs(spec: str) -> list[tuple[tuple[int, ...], str]]:
    """'2,64,16,16;2,16:long' -> [((2, 64, 16, 16), 'float'), ((2, 16), 'long')]."""
    specs: list[tuple[tuple[int, ...], str]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        dtype = "float"
        if ":" in part:
            part, dtype = part.rsplit(":", 1)
            if dtype not in DTYPE_SUFFIXES:
                raise ValueError(f"unknown dtype '{dtype}' in --smoke-input (use one of {sorted(DTYPE_SUFFIXES)})")
        dims = tuple(int(dim) for dim in part.split(",") if dim.strip())
        if not dims or any(dim <= 0 for dim in dims):
            raise ValueError(f"bad shape '{part}' in --smoke-input")
        specs.append((dims, dtype))
    if not specs:
        raise ValueError("--smoke-input is empty")
    return specs


def build_smoke_block(symbol: str, is_class: bool, init: str | None, spec: str) -> str:
    kwargs = parse_init_kwargs(init)
    inputs = parse_input_specs(spec)
    kw_text = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
    makers = []
    for dims, dtype in inputs:
        shape = ", ".join(str(dim) for dim in dims)
        if dtype == "float":
            makers.append(f"torch.randn({shape}, device=device, requires_grad=True)")
        elif dtype == "half":
            makers.append(f"torch.randn({shape}, device=device).half()")
        elif dtype == "bool":
            makers.append(f"torch.rand({shape}, device=device) > 0.5")
        else:
            makers.append(f"torch.randint(0, 8, ({shape},), device=device)")
    lines = [
        'if __name__ == "__main__":',
        "    import torch",
        "",
        "    torch.manual_seed(0)",
        '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    ]
    if is_class:
        lines.append(f"    model = {symbol}({kw_text}).to(device)")
        call = "model(*inputs)"
    else:
        call = f"{symbol}(*inputs{', ' + kw_text if kw_text else ''})"
    lines.append("    inputs = [")
    lines += [f"        {maker}," for maker in makers]
    lines.append("    ]")
    lines += [
        f"    outputs = {call}",
        "    outputs = [o for o in (outputs if isinstance(outputs, (tuple, list)) else (outputs,)) if torch.is_tensor(o)]",
        '    print("input shapes :", [tuple(t.shape) for t in inputs])',
        '    print("output shapes:", [tuple(o.shape) for o in outputs])',
        "    differentiable = [o.float().mean() for o in outputs if o.requires_grad]",
        "    if differentiable:",
        "        sum(differentiable).backward()",
    ]
    if is_class:
        lines += [
            "        params = list(model.parameters())",
            "        with_grad = sum(p.grad is not None for p in params)",
            '        print(f"backward ok | parameters: {sum(p.numel() for p in params)} | with gradient: {with_grad}/{len(params)}")',
        ]
    else:
        lines.append('        print("backward ok")')
    lines += ["    else:", '        print("outputs are not differentiable; backward skipped")']
    return "\n".join(lines)


def parse_names(raw_names: list[str] | None) -> list[str]:
    if not raw_names:
        return []
    names: list[str] = []
    for raw in raw_names:
        names.extend(part.strip() for part in raw.split(",") if part.strip())
    return names


def extract(
    source_path: Path,
    names: list[str],
    *,
    root: Path | None = None,
    include_main: bool = False,
    include_all_helpers: bool = False,
    include_dependencies: bool = True,
    resolve_local: bool = True,
    prune_imports: bool = False,
    basis: str = "source-code-derived",
    smoke_input: str | None = None,
    init: str | None = None,
    force_smoke: bool = False,
) -> tuple[str, list[str]]:
    source_path = source_path.resolve()
    workspace = Workspace(root or source_path.parent)
    main = workspace.load(source_path)
    available = top_level_defs(main.tree)
    missing = sorted(set(names) - set(available))
    if missing:
        raise KeyError(f"symbols not found: {', '.join(missing)}")
    plan = build_plan(
        workspace,
        source_path,
        set(names),
        include_main,
        include_all_helpers,
        include_dependencies,
        resolve_local,
    )
    if not any(plan.selected.values()):
        raise ValueError("no nodes selected")
    text, notes = render(
        workspace,
        plan,
        source_path,
        names,
        include_main,
        include_all_helpers,
        prune_imports,
        resolve_local,
        basis,
    )
    has_main = any(is_main_guard(node) for node in ast.parse(text).body)
    if smoke_input and (force_smoke or not has_main):
        if has_main:  # replace the source's own smoke test
            tree = ast.parse(text)
            guard = next(node for node in tree.body if is_main_guard(node))
            lines = text.splitlines()
            text = "\n".join(lines[: guard.lineno - 1]).rstrip() + "\n"
        target = names[0]
        is_class = isinstance(top_level_node_map(main.tree).get(target), ast.ClassDef)
        text = text.rstrip() + "\n\n\n" + build_smoke_block(target, is_class, init, smoke_input) + "\n"
        notes.append(f"generated smoke test for `{target}` from --smoke-input/--init (check the shapes are valid for the module)")
    elif not has_main:
        notes.append(
            "the export has no __main__ smoke test (the source has none): rerun with --smoke-input \"B,C,H,W\" "
            "[--init \"arg=value,...\"] to generate one"
        )
    return text, notes


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source Python file.")
    parser.add_argument("--output", "-o", type=Path, help="Output Python module path (required unless --list).")
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
        "--root",
        type=Path,
        help="Repository root used to resolve project-local imports (default: the source file's directory).",
    )
    parser.add_argument(
        "--no-resolve-local",
        action="store_true",
        help="Do not inline helpers imported from other files of the same repository.",
    )
    parser.add_argument(
        "--prune-imports",
        action="store_true",
        help="Drop imports from the source file that the extracted code never references.",
    )
    parser.add_argument(
        "--smoke-input",
        help=(
            "Generate a CPU-safe __main__ smoke test (shapes, backward, parameter gradients). Input shapes, "
            "semicolon-separated, optional dtype suffix: \"2,64,16,16\" or \"2,16:long;2,16:bool\"."
        ),
    )
    parser.add_argument("--init", help="Constructor keyword arguments for the smoke test, e.g. \"channels=64,reduction=16\".")
    parser.add_argument(
        "--force-smoke",
        action="store_true",
        help="Replace the source file's own __main__ block with the generated smoke test.",
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

    names = parse_names(args.name)
    tree = ast.parse(source_path.read_text(encoding="utf-8", errors="replace"), filename=str(source_path))
    if args.list:
        for name in top_level_defs(tree):
            print(name)
        return 0

    if args.output is None:
        parser.error("--output is required unless --list is used")
    if not names and not args.include_all_helpers:
        parser.error("provide --name or use --include-all-helpers")

    try:
        output, notes = extract(
            source_path,
            names,
            root=args.root,
            include_main=args.include_main,
            include_all_helpers=args.include_all_helpers,
            include_dependencies=not args.no_dependencies,
            resolve_local=not args.no_resolve_local,
            prune_imports=args.prune_imports,
            basis=args.basis,
            smoke_input=args.smoke_input,
            init=args.init,
            force_smoke=args.force_smoke,
        )
    except KeyError as exc:
        parser.error(str(exc.args[0]))
    except (ValueError, SyntaxError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"Wrote extracted module: {args.output}")
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
