#!/usr/bin/env python3
"""Find and rank likely plug-and-play module candidates in a Python source tree.

Each candidate gets a ``plug_score`` (0-10), a ``tier`` (strong / possible / weak),
and a list of ``risks`` that would stop it from running standalone (project-local
imports, registries, config objects, custom CUDA ops, ...). Nothing is imported or
executed; the analysis is purely static.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass, field
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
    "node_modules",
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
    "pool",
    "embed",
    "gate",
    "mixer",
    "unit",
}

# Names that indicate pipeline scaffolding rather than a reusable component.
SCAFFOLD_NAME_KEYWORDS = {
    "trainer",
    "dataset",
    "dataloader",
    "datamodule",
    "sampler",
    "logger",
    "runner",
    "evaluator",
    "metric",
    "config",
    "args",
    "hook",
    "callback",
    "checkpoint",
    "launcher",
    "visualizer",
    "tokenizer",
    "transform",
    "pipeline",
}

# Suffixes that usually mean "complete model", which is a poor plug-and-play unit.
WHOLE_MODEL_SUFFIXES = (
    "net",
    "model",
    "detector",
    "classifier",
    "segmentor",
    "segmenter",
    "backbone",
    "encoder",
    "decoder",
    "transformer",
    "forecaster",
)

BASE_KEYWORDS = {
    "module",
    "nn.module",
    "torch.nn.module",
    "layer",
    "keras.layers.layer",
    "tf.keras.layers.layer",
    "tf.keras.model",
    "tf.module",
    "nn.sequential",
    "autograd.function",
    "torch.autograd.function",
}

SCAFFOLD_BASES = {"dataset", "torch.utils.data.dataset", "data.dataset", "pl.lightningmodule", "lightningmodule"}

FORWARD_METHODS = {"forward", "call"}

# Imports that are safe to rely on for a standalone file.
STDLIB_LIKE = {
    "__future__", "abc", "argparse", "collections", "contextlib", "copy", "dataclasses", "enum", "functools",
    "itertools", "math", "numbers", "operator", "os", "random", "re", "sys", "typing", "warnings", "typing_extensions",
}
COMMON_DEPS = {"torch", "torchvision", "numpy", "einops", "timm", "scipy", "tensorflow", "keras", "jax", "flax"}
HEAVY_DEP_HINTS = (
    "mmcv", "mmdet", "mmseg", "mmengine", "mmcls", "mmpretrain", "detectron2", "fvcore", "pytorch_lightning",
    "lightning", "hydra", "omegaconf", "yacs", "apex", "deepspeed", "accelerate",
)
CUDA_OP_HINTS = (
    "_ext", "cuda", "triton", "mamba_ssm", "selective_scan", "causal_conv1d", "deform", "flash_attn", "xformers",
    "pointnet2_ops", "torch_scatter", "torch_sparse", "torch_geometric", "spconv", "MinkowskiEngine",
)
CONFIG_NAMES = {"cfg", "config", "configs", "args", "opt", "opts", "params", "hparams"}


@dataclass
class Candidate:
    path: str
    kind: str
    name: str
    line: int
    reason: str
    bases: list[str]
    plug_score: int = 0
    tier: str = "weak"
    forward_args: list[str] = field(default_factory=list)
    n_lines: int = 0
    n_submodules: int = 0
    risks: list[str] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)


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
            prefix = "." * node.level
            for alias in node.names:
                names.append(f"{prefix}{module}.{alias.name}" if module else f"{prefix}{alias.name}")
    return sorted(set(names))


def has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.dump(node.test).replace("'", '"')
        if "__name__" in test and "__main__" in test:
            return True
    return False


def is_module_base(base: str) -> bool:
    lowered = base.lower()
    return lowered in BASE_KEYWORDS or lowered.endswith(".module") or lowered.endswith(".layer")


def name_keywords(name: str) -> list[str]:
    lowered = name.lower()
    compact = normalize(name)
    return sorted(word for word in NAME_KEYWORDS if word in lowered or word in compact)


def candidate_reason(name: str, bases: list[str], kind: str, local_module_classes: set[str] | None = None) -> str | None:
    matched_names = name_keywords(name)
    matched_bases = [base for base in bases if is_module_base(base)]
    if local_module_classes:
        matched_bases += [base for base in bases if base in local_module_classes]

    if matched_bases and matched_names:
        return f"{kind} inherits module-like base and name matches {', '.join(matched_names)}"
    if matched_bases:
        return f"{kind} inherits module-like base"
    if matched_names:
        return f"{kind} name matches {', '.join(matched_names)}"
    return None


def local_module_class_names(tree: ast.AST) -> set[str]:
    """Classes defined in this file that (transitively) inherit a module-like base."""
    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    result: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, node in classes.items():
            if name in result:
                continue
            bases = [dotted_name(base) for base in node.bases if dotted_name(base)]
            if any(is_module_base(base) or base in result for base in bases):
                result.add(name)
                changed = True
    return result


def forward_signature(node: ast.ClassDef) -> list[str] | None:
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name in FORWARD_METHODS:
            args = [arg.arg for arg in child.args.args if arg.arg != "self"]
            args += [arg.arg for arg in child.args.kwonlyargs]
            if child.args.vararg:
                args.append("*" + child.args.vararg.arg)
            return args
    return None


def collect_import_info(tree: ast.Module) -> dict[str, tuple[str, int]]:
    """Map imported local name -> (module path, relative level)."""
    info: dict[str, tuple[str, int]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                info[(alias.asname or alias.name).split(".")[0]] = (alias.name, 0)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                info[alias.asname or alias.name] = (module, node.level)
    return info


def classify_import(module: str, level: int) -> str:
    if level > 0:
        return "local"
    root = module.split(".")[0]
    if root in STDLIB_LIKE or root in COMMON_DEPS:
        return "ok"
    if any(hint in module for hint in HEAVY_DEP_HINTS):
        return "heavy"
    return "unknown"


def referenced_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}


def decorator_names(node: ast.AST) -> list[str]:
    return [dotted_name(dec) for dec in getattr(node, "decorator_list", []) if dotted_name(dec)]


def score_class(
    node: ast.ClassDef,
    relative: str,
    import_info: dict[str, tuple[str, int]],
    top_level_defs: set[str],
    local_module_classes: set[str],
) -> Candidate | None:
    bases = [dotted_name(base) for base in node.bases if dotted_name(base)]
    reason = candidate_reason(node.name, bases, "class", local_module_classes)
    if reason is None:
        return None

    positives: list[str] = []
    risks: list[str] = []
    score = 0

    is_module = any(is_module_base(base) or base in local_module_classes for base in bases)
    forward = forward_signature(node)
    n_lines = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
    lowered = node.name.lower()

    if is_module:
        score += 3
        positives.append("module-like base class")
    if forward is not None:
        score += 2
        positives.append(f"has forward/call({', '.join(forward)})")
        if 1 <= len(forward) <= 3:
            score += 1
            positives.append("small forward signature")
        elif len(forward) > 5:
            risks.append("forward takes many arguments (likely a full model or pipeline)")
    elif is_module:
        risks.append("no forward/call defined in this class (inherits it or is abstract)")
    if name_keywords(node.name):
        score += 1
        positives.append("name matches reusable-component vocabulary")
    if n_lines <= 120:
        score += 1
        positives.append("compact (<=120 lines)")
    elif n_lines > 400:
        score -= 2
        risks.append(f"large class ({n_lines} lines); may be a full model")

    # Scaffolding and whole-model signals.
    if any(word in lowered for word in SCAFFOLD_NAME_KEYWORDS) or any(base.lower() in SCAFFOLD_BASES for base in bases):
        score -= 4
        risks.append("name/base suggests training, data, or config scaffolding")
    stripped = lowered.rstrip("0123456789_")
    if stripped.endswith(WHOLE_MODEL_SUFFIXES) and not any(word in lowered for word in ("block", "module", "layer", "attention", "unit")):
        score -= 1
        risks.append("name suggests a complete model rather than a component")

    # Submodule composition (many distinct custom classes in __init__ => model, not block).
    constructed = {
        dotted_name(child.func)
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and dotted_name(child.func)
    }
    local_constructed = {name for name in constructed if name in top_level_defs and name != node.name}
    n_submodules = sum(1 for name in constructed if name.startswith(("nn.", "torch.nn.", "layers.")) or name in local_constructed)
    if len(local_constructed) >= 4:
        score -= 1
        risks.append(f"composes {len(local_constructed)} other local classes; inspect whether they are bundled")

    # Dependencies that break standalone use.
    referenced = referenced_names(node)
    for name in sorted(referenced):
        if name in import_info:
            module, level = import_info[name]
            status = classify_import(module, level)
            shown = f"{'.' * level}{module}" if module or level else name
            if status == "local":
                risks.append(f"uses project-local import `{name}` from `{shown}`")
                score -= 1
            elif status == "heavy":
                risks.append(f"depends on framework/registry package `{shown}`")
                score -= 2
            elif status == "unknown":
                risks.append(f"depends on non-standard package `{shown}`")
            if any(hint in shown for hint in CUDA_OP_HINTS):
                risks.append(f"likely needs compiled/CUDA op from `{shown}`")
                score -= 2
    decorators = decorator_names(node)
    if any("register" in dec.lower() for dec in decorators):
        risks.append("registered through a framework registry decorator (strip it when extracting)")
        score -= 1
    uses_config = {
        child.value.id
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id in CONFIG_NAMES
    }
    init_args = []
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name == "__init__":
            init_args = [arg.arg for arg in child.args.args if arg.arg != "self"]
    if any(arg in {"num_classes", "num_class", "n_classes", "nclass", "class_num"} for arg in init_args):
        risks.append("constructor takes a class count, which usually means a complete task model")
        score -= 2
    if uses_config or any(arg in CONFIG_NAMES for arg in init_args):
        risks.append("constructor depends on a config/args object (replace with explicit arguments)")
        score -= 1
    if has_hard_cuda(node):
        risks.append("hard-codes CUDA; make device-agnostic")
        score -= 1

    score = max(0, min(10, score))
    tier = "strong" if score >= 7 and is_module and forward is not None else "possible" if score >= 4 else "weak"
    return Candidate(
        path=relative,
        kind="class",
        name=node.name,
        line=node.lineno,
        reason=reason,
        bases=bases,
        plug_score=score,
        tier=tier,
        forward_args=forward or [],
        n_lines=n_lines,
        n_submodules=n_submodules,
        risks=risks,
        positives=positives,
    )


def has_hard_cuda(node: ast.AST) -> bool:
    """Unconditional ``.cuda()`` or ``.to("cuda")`` calls (guarded device strings are fine)."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        if child.func.attr == "cuda":
            return True
        if child.func.attr == "to" and any(
            isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("cuda")
            for arg in child.args
        ):
            return True
    return False


def score_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    relative: str,
    import_info: dict[str, tuple[str, int]],
) -> Candidate | None:
    reason = candidate_reason(node.name, [], "function")
    if reason is None or node.name.startswith("_"):
        return None
    lowered = node.name.lower()
    args = [arg.arg for arg in node.args.args if arg.arg != "self"]
    n_lines = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
    positives: list[str] = []
    risks: list[str] = []
    score = 1
    has_return = any(isinstance(child, ast.Return) and child.value is not None for child in ast.walk(node))
    if has_return:
        score += 2
        positives.append("returns a value")
    if "loss" in lowered or "attention" in lowered or "attn" in lowered or "norm" in lowered:
        score += 2
        positives.append("loss/attention/normalization-style name")
    if 1 <= len(args) <= 6:
        score += 1
        positives.append("bounded signature")
    if n_lines <= 80:
        score += 1
    if any(word in lowered for word in SCAFFOLD_NAME_KEYWORDS) or lowered.startswith(("train", "test", "eval", "load", "main", "parse", "get_args")):
        score -= 4
        risks.append("name suggests pipeline scaffolding")
    for name in sorted(referenced_names(node)):
        if name in import_info:
            module, level = import_info[name]
            status = classify_import(module, level)
            if status in {"local", "heavy"}:
                risks.append(f"depends on `{name}` from `{'.' * level}{module}`")
                score -= 1
    score = max(0, min(10, score))
    tier = "possible" if score >= 5 else "weak"
    return Candidate(
        path=relative,
        kind="function",
        name=node.name,
        line=node.lineno,
        reason=reason,
        bases=[],
        plug_score=score,
        tier=tier,
        forward_args=args,
        n_lines=n_lines,
        risks=risks,
        positives=positives,
    )


def scan_file(path: Path, root: Path) -> FileSummary:
    relative = path.relative_to(root).as_posix() if path != root else path.name
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

    import_info = collect_import_info(tree)
    top_level_defs = {
        node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    local_module_classes = local_module_class_names(tree)

    candidates: list[Candidate] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            candidate = score_class(node, relative, import_info, top_level_defs, local_module_classes)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.col_offset > 0:  # methods and nested helpers are not extractable units
                continue
            candidate = score_function(node, relative, import_info)
        else:
            continue
        if candidate:
            candidates.append(candidate)

    return FileSummary(
        path=relative,
        imports=import_names(tree),
        candidates=sorted(candidates, key=lambda item: (item.line, item.kind, item.name)),
        has_main=has_main_guard(tree),
    )


def iter_python_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


TIER_ORDER = {"strong": 0, "possible": 1, "weak": 2}


def overall_verdict(summaries: list[FileSummary]) -> str:
    tiers = {candidate.tier for summary in summaries for candidate in summary.candidates}
    if "strong" in tiers:
        return "strong"
    if "possible" in tiers:
        return "possible"
    return "none"


def ranked(summaries: list[FileSummary]) -> list[Candidate]:
    items = [candidate for summary in summaries for candidate in summary.candidates]
    return sorted(items, key=lambda item: (TIER_ORDER[item.tier], -item.plug_score, item.path, item.line))


def render_text(summaries: list[FileSummary], min_tier: str) -> str:
    threshold = TIER_ORDER[min_tier]
    lines: list[str] = []
    for summary in summaries:
        shown = [candidate for candidate in summary.candidates if TIER_ORDER[candidate.tier] <= threshold]
        if not shown and not summary.parse_error:
            continue
        lines.append(f"{summary.path}")
        if summary.parse_error:
            lines.append(f"  parse_error: {summary.parse_error}")
        if summary.imports:
            lines.append(f"  imports: {', '.join(summary.imports[:12])}")
        if summary.has_main:
            lines.append("  has_main: yes")
        for candidate in shown:
            base_text = f" bases={candidate.bases}" if candidate.bases else ""
            lines.append(
                f"  {candidate.kind} {candidate.name}:{candidate.line}{base_text} "
                f"[tier={candidate.tier} plug_score={candidate.plug_score}]"
            )
            lines.append(f"    reason: {candidate.reason}")
            if candidate.positives:
                lines.append(f"    positives: {'; '.join(candidate.positives)}")
            for risk in candidate.risks:
                lines.append(f"    risk: {risk}")
        lines.append("")
    return "\n".join(lines).strip()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Python source root (or single .py file) to scan.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--min-tier",
        choices=["strong", "possible", "weak"],
        default="weak",
        help="Hide candidates below this tier in text output (default: show all).",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.exists():
        parser.error(f"source root not found: {root}")
    scan_root = root.parent if root.is_file() else root

    summaries = [scan_file(path, scan_root) for path in sorted(iter_python_files(root))]
    if args.json:
        print(json.dumps([asdict(summary) for summary in summaries], ensure_ascii=False, indent=2))
    else:
        text = render_text(summaries, args.min_tier)
        verdict = overall_verdict(summaries)
        print(text if text else "No obvious Python module candidates found.")
        print(f"\ncode_candidates: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
