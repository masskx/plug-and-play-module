#!/usr/bin/env python3
"""Run the full regression suite for the plug-and-play module extractor skill."""

from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_paper_modules import analyze  # noqa: E402
from evaluate_sample_cases import choose_named_extract_target  # noqa: E402
from extract_pdf_text import extract_pages, render_pages  # noqa: E402


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run_command(command: list[str], timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )
    output = "\n".join(part for part in ((proc.stdout or "").strip(), (proc.stderr or "").strip()) if part)
    return proc.returncode, output


def summarize_status(returncode: int) -> str:
    return "passed" if returncode == 0 else "failed"


def compile_scripts() -> list[str]:
    failures: list[str] = []
    for script in sorted(SCRIPT_DIR.glob("*.py")):
        try:
            py_compile.compile(str(script), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{script.name}: {exc}")
    return failures


def build_simulated_generated_batch(samples_root: Path, output_root: Path, timeout: int) -> list[str]:
    failures: list[str] = []
    case_dirs = [path for path in samples_root.iterdir() if path.is_dir()]
    case_dirs.sort(key=lambda path: int(path.name) if path.name.isdigit() else path.name)

    for case_dir in case_dirs:
        pdf = next(iter(sorted(case_dir.glob("*.pdf"))), None)
        source = next(iter(sorted(case_dir.glob("*.py"))), None)
        if pdf is None or source is None:
            continue

        try:
            backend, pages = extract_pages(pdf, max_pages=4)
            text = render_pages(pdf, backend, pages)
            analysis = analyze(text)
            paper_candidates = [candidate.name for candidate in analysis.candidates]
            target = choose_named_extract_target(source, paper_candidates)
            if target == "none":
                failures.append(f"case {case_dir.name}: no named extraction target")
                continue
        except Exception as exc:
            failures.append(f"case {case_dir.name}: failed to choose target: {type(exc).__name__}: {exc}")
            continue

        output = output_root / f"case{case_dir.name}.py"
        code, command_output = run_command(
            [
                sys.executable,
                str(SCRIPT_DIR / "extract_python_module.py"),
                str(source),
                "--output",
                str(output),
                "--name",
                target,
                "--include-main",
            ],
            timeout,
        )
        if code != 0:
            failures.append(f"case {case_dir.name}: extraction failed for {target}: {command_output}")
    return failures


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_root", type=Path, help="Directory containing numbered sample folders.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for module smoke tests.")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout for subprocess steps.")
    parser.add_argument("--validator", type=Path, help="Optional path to skill-creator quick_validate.py.")
    parser.add_argument("--skip-generated-batch", action="store_true", help="Skip simulated generated batch evaluation.")
    parser.add_argument("--report-json", type=Path, help="Write a JSON report for this regression run.")
    args = parser.parse_args(argv)

    samples_root = args.samples_root.resolve()
    if not samples_root.exists():
        parser.error(f"samples root not found: {samples_root}")

    failures: list[str] = []
    report: dict[str, object] = {
        "skill_root": str(SKILL_ROOT),
        "samples_root": str(samples_root),
        "python": args.python,
        "stages": {},
        "failures": failures,
    }

    compile_failures = compile_scripts()
    if compile_failures:
        failures.extend(f"compile: {failure}" for failure in compile_failures)
    print(f"compile_scripts: {'passed' if not compile_failures else 'failed'}")
    report["stages"]["compile_scripts"] = {
        "status": "passed" if not compile_failures else "failed",
        "failures": compile_failures,
    }

    if args.validator:
        code, output = run_command([sys.executable, str(args.validator), str(SKILL_ROOT)], args.timeout)
        print(output)
        if code != 0:
            failures.append(f"quick_validate failed: {output}")
        report["stages"]["quick_validate"] = {
            "status": summarize_status(code),
            "output": output,
        }
    else:
        print("quick_validate: skipped")
        report["stages"]["quick_validate"] = {"status": "skipped"}

    code, output = run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "evaluate_sample_cases.py"),
            str(samples_root),
            "--python",
            args.python,
            "--timeout",
            str(args.timeout),
        ],
        max(args.timeout * 20, 300),
    )
    print(output)
    if code != 0:
        failures.append("evaluate_sample_cases failed")
    report["stages"]["evaluate_sample_cases"] = {
        "status": summarize_status(code),
        "output": output,
    }

    if not args.skip_generated_batch:
        with tempfile.TemporaryDirectory() as temp_dir:
            generated_root = Path(temp_dir) / "generated"
            generated_root.mkdir(parents=True, exist_ok=True)
            generation_failures = build_simulated_generated_batch(samples_root, generated_root, args.timeout)
            failures.extend(generation_failures)
            if generation_failures:
                print("simulated_generated_batch: failed")
                for failure in generation_failures:
                    print(failure)
            else:
                print("simulated_generated_batch: generated")
            report["stages"]["simulated_generated_batch"] = {
                "status": "failed" if generation_failures else "passed",
                "failures": generation_failures,
            }

            code, output = run_command(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "evaluate_generated_batch.py"),
                    str(samples_root),
                    str(generated_root),
                    "--python",
                    args.python,
                    "--timeout",
                    str(args.timeout),
                ],
                max(args.timeout * 20, 300),
            )
            print(output)
            if code != 0:
                failures.append("evaluate_generated_batch failed")
            report["stages"]["evaluate_generated_batch"] = {
                "status": summarize_status(code),
                "output": output,
            }
    else:
        report["stages"]["simulated_generated_batch"] = {"status": "skipped"}
        report["stages"]["evaluate_generated_batch"] = {"status": "skipped"}

    if failures:
        print("regression_suite: failed")
        for failure in failures:
            print(f"- {failure}")
        report["status"] = "failed"
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    print("regression_suite: passed")
    report["status"] = "passed"
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
