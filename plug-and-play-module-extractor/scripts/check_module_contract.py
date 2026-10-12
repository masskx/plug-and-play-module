#!/usr/bin/env python3
"""Statically check that an exported module is standalone and directly usable.

Checks (errors fail the run, warnings are advisory):

* compiles, has a module docstring and a ``__main__`` smoke test;
* no project-local / relative imports, no undefined names;
* no unconditional CUDA, network or download calls, no hard-coded absolute paths;
* leftover training scaffolding (argparse, DataLoader, optimizer steps);
* third-party dependencies are listed so the final report can state them.

Use ``--run`` to also execute the file (via ``smoke_test_module.py``) with the interpreter
that has the dependencies installed.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {
    "__future__", "typing_extensions",
}
KNOWN_THIRD_PARTY = {
    "torch", "torchvision", "torchaudio", "numpy", "einops", "timm", "scipy", "tensorflow", "keras", "jax", "flax",
    "pandas", "sklearn", "PIL", "cv2", "matplotlib", "mamba_ssm", "causal_conv1d", "pywt", "pytorch_wavelets",
    "torch_geometric", "fvcore", "thop",
}
NETWORK_PATTERNS = (
    (re.compile(r"\brequests\.(?:get|post|put|head)\b"), "network call via requests"),
    (re.compile(r"\burllib\.request\b|\burlopen\("), "network call via urllib"),
    (re.compile(r"torch\.hub\.(?:load|download_url_to_file)"), "torch.hub download"),
    (re.compile(r"\bfrom_pretrained\("), "from_pretrained() downloads weights"),
    (re.compile(r"\bpretrained\s*=\s*True"), "pretrained=True downloads weights"),
    (re.compile(r"\bdownload\s*=\s*True"), "download=True downloads data"),
    (re.compile(r"\bwget\b|\bcurl\b"), "shell download"),
)
ABS_PATH = re.compile(r"""["'](?:/home/|/Users/|/data/|/mnt/|[A-Za-z]:\\)""")
SCAFFOLD_PATTERNS = (
    (re.compile(r"\bargparse\b"), "argparse in a reusable module"),
    (re.compile(r"\bDataLoader\b"), "DataLoader leftover"),
    (re.compile(r"\boptimizer\.step\(|\boptim\.(?:SGD|Adam|AdamW)\b"), "optimizer/training loop leftover"),
    (re.compile(r"\bwandb\b|\btensorboard\b|\bSummaryWriter\b"), "logger leftover"),
    (re.compile(r"\btorch\.distributed\b|\bDistributedDataParallel\b"), "distributed-training leftover"),
    (re.compile(r"\btorch\.load\(|\bload_state_dict\("), "checkpoint loading in a reusable module"),
)


@dataclass
class Finding:
    severity: str  # error | warning
    code: str
    message: str
    line: int | None = None


@dataclass
class ContractReport:
    module: str
    status: str
    findings: list[Finding] = field(default_factory=list)
    third_party_dependencies: list[str] = field(default_factory=list)
    stdlib_dependencies: list[str] = field(default_factory=list)
    public_symbols: list[str] = field(default_factory=list)
    run: dict | None = None


class BindingCollector(ast.NodeVisitor):
    """Collect every name bound anywhere in the module (over-approximation of scope)."""

    def __init__(self) -> None:
        self.bound: set[str] = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__spec__", "__builtins__"}

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)
        self._args(node.args)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._args(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bound.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.bound.update(node.names)

    def visit_MatchAs(self, node: ast.AST) -> None:  # pragma: no cover - rare syntax
        name = getattr(node, "name", None)
        if name:
            self.bound.add(name)
        self.generic_visit(node)

    def _args(self, args: ast.arguments) -> None:
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            self.bound.add(arg.arg)
        if args.vararg:
            self.bound.add(args.vararg.arg)
        if args.kwarg:
            self.bound.add(args.kwarg.arg)


def undefined_names(tree: ast.Module) -> list[tuple[str, int]]:
    collector = BindingCollector()
    collector.visit(tree)
    missing: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in collector.bound:
            missing.setdefault(node.id, node.lineno)
    return sorted(missing.items(), key=lambda item: item[1])


def has_main_guard(tree: ast.Module) -> ast.If | None:
    for node in tree.body:
        if isinstance(node, ast.If):
            text = ast.dump(node.test).replace("'", '"')
            if "__name__" in text and "__main__" in text:
                return node
    return None


def line_of(source: str, pattern: re.Pattern[str]) -> int | None:
    for number, line in enumerate(source.splitlines(), start=1):
        if pattern.search(line) and not line.lstrip().startswith("#"):
            return number
    return None


def hard_cuda_lines(tree: ast.Module) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "cuda":
            lines.append(node.lineno)
        elif node.func.attr == "to" and any(
            isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("cuda")
            for arg in [*node.args, *[kw.value for kw in node.keywords]]
        ):
            lines.append(node.lineno)
        elif any(
            kw.arg == "device" and isinstance(kw.value, ast.Constant) and str(kw.value.value).startswith("cuda")
            for kw in node.keywords
        ):
            lines.append(node.lineno)
    return sorted(set(lines))


def check(path: Path, require_main: bool = True) -> ContractReport:
    findings: list[Finding] = []
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return ContractReport(
            module=str(path),
            status="failed",
            findings=[Finding("error", "syntax", f"{exc.msg}", exc.lineno)],
        )

    docstring = ast.get_docstring(tree) or ""
    if not docstring:
        findings.append(Finding("warning", "docstring", "missing module docstring (paper/module name, basis, shapes)"))
    else:
        lowered = docstring.lower()
        if "basis" not in lowered and "derived" not in lowered:
            findings.append(
                Finding("warning", "docstring", "docstring should state the implementation basis (source-code-derived / paper-derived / mixed)")
            )
        if not any(word in lowered for word in ("input", "output", "shape")):
            findings.append(Finding("warning", "docstring", "docstring should describe expected input/output shapes"))

    main_guard = has_main_guard(tree)
    if main_guard is None:
        findings.append(
            Finding("error" if require_main else "warning", "main", "no `if __name__ == \"__main__\":` smoke test")
        )
    else:
        main_source = ast.get_source_segment(source, main_guard) or ""
        if "shape" not in main_source:
            findings.append(Finding("warning", "main", "smoke test does not print input/output shapes", main_guard.lineno))
        if "backward" not in main_source and "requires_grad" not in main_source:
            findings.append(
                Finding("warning", "main", "smoke test never checks gradients (add a backward() call for trainable modules)", main_guard.lineno)
            )

    third_party: set[str] = set()
    stdlib: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                findings.append(
                    Finding("error", "local-import", f"relative import `{ast.unparse(node)}` is not standalone", node.lineno)
                )
                continue
            root = (node.module or "").split(".")[0]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                _classify_import(alias.name.split(".")[0], node.lineno, findings, third_party, stdlib)
            continue
        else:
            continue
        _classify_import(root, node.lineno, findings, third_party, stdlib)

    for name, line in undefined_names(tree):
        findings.append(Finding("error", "undefined-name", f"`{name}` is used but never defined or imported", line))

    for line in hard_cuda_lines(tree):
        findings.append(Finding("error", "hard-cuda", "unconditional CUDA use; select the device with torch.cuda.is_available()", line))

    code_only = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    for pattern, message in NETWORK_PATTERNS:
        if pattern.search(code_only):
            findings.append(Finding("error", "network", message, line_of(source, pattern)))
    if ABS_PATH.search(code_only):
        findings.append(Finding("warning", "absolute-path", "hard-coded absolute path", line_of(source, ABS_PATH)))
    for pattern, message in SCAFFOLD_PATTERNS:
        if pattern.search(code_only):
            findings.append(Finding("warning", "scaffolding", message, line_of(source, pattern)))

    public = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_")
    ]
    if not public:
        findings.append(Finding("error", "symbols", "no public class or function was exported"))

    has_error = any(f.severity == "error" for f in findings)
    return ContractReport(
        module=str(path),
        status="failed" if has_error else "passed",
        findings=findings,
        third_party_dependencies=sorted(third_party),
        stdlib_dependencies=sorted(stdlib),
        public_symbols=public,
    )


def _classify_import(root: str, line: int, findings: list[Finding], third_party: set[str], stdlib: set[str]) -> None:
    if not root:
        return
    if root in STDLIB:
        stdlib.add(root)
    elif root in KNOWN_THIRD_PARTY:
        third_party.add(root)
    else:
        third_party.add(root)
        findings.append(
            Finding(
                "warning",
                "dependency",
                f"`{root}` is not a standard dependency; confirm it is pip-installable or vendor the code it provides",
                line,
            )
        )


def run_module(path: Path, python: str, timeout: int) -> dict:
    smoke = Path(__file__).resolve().parent / "smoke_test_module.py"
    proc = subprocess.run(
        [sys.executable, str(smoke), str(path), "--python", python, "--timeout", str(timeout), "--json"],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"classification": "failed", "run": {"status": "failed", "detail": proc.stdout + proc.stderr}}


def render_text(report: ContractReport) -> str:
    lines = [
        f"module: {report.module}",
        f"contract: {report.status}",
        f"public_symbols: {', '.join(report.public_symbols) or 'none'}",
        f"third_party_dependencies: {', '.join(report.third_party_dependencies) or 'none'}",
        f"stdlib_dependencies: {', '.join(report.stdlib_dependencies) or 'none'}",
        "findings:",
    ]
    if not report.findings:
        lines.append("  none")
    for finding in report.findings:
        where = f":{finding.line}" if finding.line else ""
        lines.append(f"  - [{finding.severity}] {finding.code}{where}: {finding.message}")
    if report.run:
        lines.append(f"smoke_test: {report.run.get('classification')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("module", type=Path, help="Exported Python module to check.")
    parser.add_argument("--run", action="store_true", help="Also run the module through smoke_test_module.py.")
    parser.add_argument("--python", default=sys.executable, help="Interpreter with the module's dependencies.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-require-main", action="store_true", help="Downgrade a missing __main__ block to a warning.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.module.exists():
        parser.error(f"module not found: {args.module}")

    report = check(args.module.resolve(), require_main=not args.no_require_main)
    if args.run and report.status == "passed":
        report.run = run_module(args.module.resolve(), args.python, args.timeout)
        if report.run.get("classification") not in {"passed", "missing_dependency", "accelerator_unavailable"}:
            report.status = "failed"
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
