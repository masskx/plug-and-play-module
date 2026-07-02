#!/usr/bin/env python3
"""Compile, import, and run a generated Python module with clear diagnostics."""

from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class StepResult:
    status: str
    detail: str


@dataclass
class SmokeResult:
    module: str
    compile: StepResult
    import_check: StepResult
    run: StepResult
    classification: str


def classify_failure(text: str) -> str:
    lowered = text.lower()
    if "torch not compiled with cuda enabled" in lowered or "cuda is not available" in lowered:
        return "accelerator_unavailable"
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        return "missing_dependency"
    if "importerror" in lowered or "cannot import name" in lowered:
        return "import_error"
    if "syntaxerror" in lowered:
        return "syntax_error"
    if "runtimeerror" in lowered or "valueerror" in lowered or "assertionerror" in lowered:
        return "runtime_error"
    return "failed"


def run_subprocess(command: list[str], timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    output = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
    return proc.returncode, output


def compile_step(module_path: Path) -> StepResult:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            py_compile.compile(str(module_path), cfile=str(Path(temp_dir) / "module.pyc"), doraise=True)
        return StepResult("passed", "py_compile succeeded")
    except py_compile.PyCompileError as exc:
        return StepResult("failed", str(exc))


def import_step(module_path: Path, python: str, timeout: int) -> StepResult:
    code = (
        "import importlib.util, pathlib, sys\n"
        "path = pathlib.Path(sys.argv[1]).resolve()\n"
        "spec = importlib.util.spec_from_file_location('extracted_module_under_test', path)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "assert spec and spec.loader\n"
        "spec.loader.exec_module(module)\n"
        "print('import ok')\n"
    )
    try:
        returncode, output = run_subprocess([python, "-c", code, str(module_path)], timeout)
    except subprocess.TimeoutExpired:
        return StepResult("failed", f"import timed out after {timeout}s")
    return StepResult("passed" if returncode == 0 else "failed", output)


def run_step(module_path: Path, python: str, timeout: int, skip_run: bool) -> StepResult:
    if skip_run:
        return StepResult("not-run", "skipped by request")
    try:
        returncode, output = run_subprocess([python, str(module_path)], timeout)
    except subprocess.TimeoutExpired:
        return StepResult("failed", f"script run timed out after {timeout}s")
    return StepResult("passed" if returncode == 0 else "failed", output)


def summarize(module_path: Path, python: str, timeout: int, skip_run: bool) -> SmokeResult:
    compile_result = compile_step(module_path)
    if compile_result.status != "passed":
        return SmokeResult(
            str(module_path),
            compile_result,
            StepResult("not-run", "compile failed"),
            StepResult("not-run", "compile failed"),
            classify_failure(compile_result.detail),
        )

    import_result = import_step(module_path, python, timeout)
    if import_result.status != "passed":
        return SmokeResult(
            str(module_path),
            compile_result,
            import_result,
            StepResult("not-run", "import failed"),
            classify_failure(import_result.detail),
        )

    run_result = run_step(module_path, python, timeout, skip_run)
    if run_result.status == "passed":
        classification = "passed"
    elif run_result.status == "not-run":
        classification = "not_run"
    else:
        classification = classify_failure(run_result.detail)

    return SmokeResult(str(module_path), compile_result, import_result, run_result, classification)


def render_text(result: SmokeResult) -> str:
    return "\n".join(
        [
            f"module: {result.module}",
            f"classification: {result.classification}",
            f"compile: {result.compile.status} - {result.compile.detail}",
            f"import_check: {result.import_check.status} - {result.import_check.detail}",
            f"run: {result.run.status} - {result.run.detail}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=Path, help="Generated Python module to test.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per subprocess step.")
    parser.add_argument("--skip-run", action="store_true", help="Only compile and import the module.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    module_path = args.module.resolve()
    if not module_path.exists():
        parser.error(f"module not found: {module_path}")

    result = summarize(module_path, args.python, args.timeout, args.skip_run)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(render_text(result))

    return 0 if result.classification in {"passed", "not_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
