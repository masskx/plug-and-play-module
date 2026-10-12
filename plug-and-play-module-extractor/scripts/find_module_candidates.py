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
    is_module: bool = False
    whole_model: bool = False
    scaffolding: bool = False
    stateless_tiny: bool = False
    blocked_by_dependencies: bool = False
    composes: list[str] = field(default_factory=list)


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
    # ast.walk also reaches `try: from causal_conv1d import ... except ImportError:` guards and
    # `if TYPE_CHECKING:` blocks, which is where optional compiled-op dependencies usually hide.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                info.setdefault((alias.asname or alias.name).split(".")[0], (alias.name, 0))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                info.setdefault(alias.asname or alias.name, (module, node.level))
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
    if reason is None and forward_signature(node) is not None:
        # Framework bases (mmengine BaseModule, custom Layer subclasses) and unremarkable names such as
        # `FPN` or `Net` are still modules if they define forward()/call().
        reason = "class defines forward()/call()"
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
    scaffolding = any(word in lowered for word in SCAFFOLD_NAME_KEYWORDS) or any(
        base.lower() in SCAFFOLD_BASES for base in bases
    )
    whole_model = False
    if scaffolding:
        score -= 4
        risks.append("name/base suggests training, data, or config scaffolding")
    stripped = lowered.rstrip("0123456789_")
    if stripped.endswith(WHOLE_MODEL_SUFFIXES) and not any(word in lowered for word in ("block", "module", "layer", "attention", "unit")):
        whole_model = True
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
        whole_model = True
        score -= 2
    if uses_config or any(arg in CONFIG_NAMES for arg in init_args):
        risks.append("constructor depends on a config/args object (replace with explicit arguments)")
        score -= 1
    if has_hard_cuda(node):
        risks.append("hard-codes CUDA; make device-agnostic")
        score -= 1

    # Tiny stateless helpers (Flatten, ChannelPool, h_sigmoid) are real modules but rarely worth exporting.
    learnable_markers = ("Conv", "Linear", "Norm", "Embedding", "Parameter", "LSTM", "GRU", "MultiheadAttention", "Bilinear")
    has_state = bool(local_constructed) or any(any(marker in name for marker in learnable_markers) for name in constructed)
    forward_node = next((c for c in node.body if isinstance(c, ast.FunctionDef) and c.name in FORWARD_METHODS), None)
    forward_stmts = len(forward_node.body) if forward_node else 0
    stateless_tiny = bool(forward_node) and not has_state and forward_stmts <= 3 and n_lines <= 25
    if stateless_tiny:
        risks.append("tiny stateless helper (no learnable layers, <=3 statements); usually not worth exporting alone")
        score -= 2

    blocked = (
        forward is not None
        and not scaffolding
        and not whole_model
        and not stateless_tiny
        and score < 4
        and any(("package" in risk or "project-local" in risk or "registry" in risk or "CUDA" in risk or "config" in risk) for risk in risks)
    )
    if blocked:
        risks.append("module-like (defines forward) but blocked by dependencies; extractable only after replacing them")

    score = max(0, min(10, score))
    candidate_flags = dict(
        is_module=is_module,
        whole_model=whole_model,
        scaffolding=scaffolding,
        stateless_tiny=stateless_tiny,
        blocked_by_dependencies=blocked,
    )
    tier = assign_tier(score, forward is not None, candidate_flags)
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
        **candidate_flags,
    )


def assign_tier(score: int, has_forward: bool, flags: dict) -> str:
    """strong: self-contained module; possible: usable after inspection/dependency work; weak: probably not a unit."""
    if flags["whole_model"] or flags["scaffolding"]:
        return "weak"  # whole models and pipeline classes are listed as rejected, never as candidates
    if score >= 7 and flags["is_module"] and has_forward:
        return "strong"
    if score >= 4 or flags["blocked_by_dependencies"] or flags["stateless_tiny"] and score >= 2:
        return "possible"
    return "weak"


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


def boost_composite_roots(tree: ast.Module, candidates: list[Candidate]) -> None:
    """A class that composes other candidates of the file and is not used by any of them is the file's main unit."""
    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    names = {c.name for c in candidates if c.kind == "class"}
    uses: dict[str, set[str]] = {
        name: {ref for ref in referenced_names(classes[name]) if ref in names and ref != name}
        for name in names
        if name in classes
    }
    used_by_others = {ref for refs in uses.values() for ref in refs}
    for candidate in candidates:
        refs = uses.get(candidate.name)
        if not refs:
            continue
        candidate.composes = sorted(refs)
        if candidate.name not in used_by_others and candidate.tier != "weak":
            candidate.plug_score = min(10, candidate.plug_score + 1)
            candidate.positives.append(f"composes {', '.join(sorted(refs))} and is not used by them (main unit of the file)")
            if candidate.tier == "possible" and candidate.plug_score >= 7 and candidate.is_module:
                candidate.tier = "strong"


def read_source(path: Path) -> str:
    """Python source of a .py file, or the concatenated code cells of a notebook."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() != ".ipynb":
        return text
    try:
        cells = json.loads(text).get("cells", [])
    except json.JSONDecodeError:
        return ""
    chunks = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        body = cell.get("source", "")
        body = "".join(body) if isinstance(body, list) else body
        # Drop IPython magics and shell escapes so the cell parses as Python.
        chunks.append("\n".join(line for line in body.splitlines() if not line.lstrip().startswith(("%", "!"))))
    return "\n\n".join(chunks)


def scan_file(path: Path, root: Path) -> FileSummary:
    relative = path.relative_to(root).as_posix() if path != root else path.name
    source = read_source(path)

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

    boost_composite_roots(tree, candidates)
    whole_models = {c.name for c in candidates if c.kind == "class" and c.whole_model}
    for candidate in candidates:
        if candidate.kind != "function" or not whole_models:
            continue
        node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == candidate.name), None)
        if node is not None and referenced_names(node) & whole_models:
            candidate.tier = "weak"
            candidate.whole_model = True
            candidate.risks.append("factory function for a complete model, not a component")

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
    for pattern in ("*.py", "*.ipynb"):
        for path in root.rglob(pattern):
            if any(part in SKIP_DIRS or part == ".ipynb_checkpoints" for part in path.relative_to(root).parts):
                continue
            yield path


TIER_ORDER = {"strong": 0, "possible": 1, "weak": 2}


def scan_coverage(summaries: list[FileSummary]) -> dict:
    """How much of the code could actually be read. A verdict of 'none' is only meaningful with full coverage."""
    parsed = [s for s in summaries if not s.parse_error]
    return {
        "files": len(summaries),
        "parsed": len(parsed),
        "parse_errors": [f"{s.path}: {s.parse_error}" for s in summaries if s.parse_error],
        "notebooks": sum(1 for s in summaries if s.path.endswith(".ipynb")),
    }


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
