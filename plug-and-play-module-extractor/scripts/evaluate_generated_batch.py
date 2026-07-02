#!/usr/bin/env python3
"""Evaluate a directory of generated modules against numbered sample cases."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_generated_module import evaluate  # noqa: E402


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def candidate_generated_paths(generated_root: Path, case_name: str) -> list[Path]:
    return [
        generated_root / f"case{case_name}.py",
        generated_root / f"case_{case_name}.py",
        generated_root / f"{case_name}.py",
        generated_root / case_name / "module.py",
        generated_root / case_name / f"case{case_name}.py",
    ]


def find_generated_module(generated_root: Path, case_name: str) -> Path | None:
    for path in candidate_generated_paths(generated_root, case_name):
        if path.exists():
            return path
    return None


def render_text(results: list[dict]) -> str:
    ok = sum(1 for result in results if result["status"] == "ok")
    skipped = sum(1 for result in results if result["status"] == "skipped")
    needs_attention = sum(1 for result in results if result["status"] == "needs_attention")
    lines = [f"summary: ok={ok} skipped={skipped} needs_attention={needs_attention}", ""]
    for result in results:
        if result["status"] == "skipped":
            lines.append(f"case {result['case']}: status=skipped notes={result['notes']}")
        else:
            lines.append(
                f"case {result['case']}: status={result['status']} "
                f"paper_alignment={result['paper_alignment']} "
                f"reference_alignment={result['reference_alignment']} "
                f"smoke={result['smoke_classification']} "
                f"generated={result['generated_module']}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_root", type=Path, help="Directory containing numbered sample case folders.")
    parser.add_argument("generated_root", type=Path, help="Directory containing generated modules.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for smoke tests.")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout for subprocess steps.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip cases without generated modules.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    samples_root = args.samples_root.resolve()
    generated_root = args.generated_root.resolve()
    if not samples_root.exists():
        parser.error(f"samples root not found: {samples_root}")
    if not generated_root.exists():
        parser.error(f"generated root not found: {generated_root}")

    case_dirs = [path for path in samples_root.iterdir() if path.is_dir()]
    case_dirs.sort(key=lambda path: int(path.name) if path.name.isdigit() else path.name)

    results: list[dict] = []
    for case_dir in case_dirs:
        has_pdf = bool(list(case_dir.glob("*.pdf")))
        has_reference = bool(list(case_dir.glob("*.py")))
        generated_module = find_generated_module(generated_root, case_dir.name)
        if not has_pdf or not has_reference:
            results.append(
                {
                    "case": case_dir.name,
                    "status": "skipped",
                    "notes": "missing sample PDF or reference module",
                }
            )
            continue
        if generated_module is None:
            if args.allow_missing:
                results.append(
                    {
                        "case": case_dir.name,
                        "status": "skipped",
                        "notes": "generated module not found",
                    }
                )
                continue
            results.append(
                {
                    "case": case_dir.name,
                    "status": "needs_attention",
                    "paper_alignment": "not-run",
                    "reference_alignment": "not-run",
                    "smoke_classification": "not-run",
                    "generated_module": "missing",
                    "notes": "generated module not found",
                }
            )
            continue
        evaluation = evaluate(case_dir, generated_module, args.python, args.timeout)
        data = asdict(evaluation)
        data["case"] = case_dir.name
        results.append(data)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(render_text(results))

    return 1 if any(result["status"] == "needs_attention" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
