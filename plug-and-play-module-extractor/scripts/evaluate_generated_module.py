#!/usr/bin/env python3
"""Evaluate a generated PDF-only module against one sample case."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_paper_modules import analyze  # noqa: E402


ACCEPTABLE_SMOKE = {"passed", "missing_dependency", "accelerator_unavailable", "not_run"}
GENERIC_TOKENS = {
    "attention",
    "block",
    "cell",
    "class",
    "conv",
    "decomp",
    "filter",
    "for",
    "function",
    "layer",
    "learning",
    "loss",
    "method",
    "mixing",
    "module",
    "net",
    "network",
    "the",
}
SPECIAL_TOKENS = {"attention", "conv", "decomposition", "filter", "fusion", "loss", "mixing", "selector", "token"}


@dataclass
class GeneratedEvaluation:
    case_dir: str
    generated_module: str
    found_module: str
    paper_candidates: list[str]
    reference_symbols: list[str]
    generated_symbols: list[str]
    paper_alignment: str
    reference_alignment: str
    smoke_classification: str
    status: str
    notes: list[str]


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def raw_tokens(value: str) -> set[str]:
    split_camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return {part.lower() for part in re.split(r"[^A-Za-z0-9]+", split_camel) if len(part) >= 3}


def meaningful_tokens(value: str) -> set[str]:
    return {token for token in raw_tokens(value) if token not in GENERIC_TOKENS}


def aligned(left_values: list[str], right_values: list[str]) -> list[str]:
    matches: list[str] = []
    right_index = [(value, compact(value), meaningful_tokens(value), raw_tokens(value)) for value in right_values]
    for left in left_values:
        left_compact = compact(left)
        left_tokens = meaningful_tokens(left)
        left_raw = raw_tokens(left)
        if len(left_compact) < 3:
            continue
        for right, right_compact, right_tokens, right_raw in right_index:
            direct = left_compact in right_compact or right_compact in left_compact
            token = bool(left_tokens & right_tokens) and (
                len(left_tokens & right_tokens) >= 2 or len(left_compact) >= 5 or len(right_compact) <= 8
            )
            special = bool(left_raw & right_raw & SPECIAL_TOKENS) and (len(left_compact) >= 8 or len(right_compact) >= 8)
            if direct or token or special:
                matches.append(f"{left}->{right}")
                break
    return matches


def top_level_symbols(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(path))
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(node.name)
    return symbols


def run_json(command: list[str], timeout: int) -> tuple[int, dict | None, str]:
    proc = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )
    output = "\n".join(part for part in ((proc.stdout or "").strip(), (proc.stderr or "").strip()) if part)
    try:
        data = json.loads(proc.stdout or "")
        return proc.returncode, data if isinstance(data, dict) else None, output
    except json.JSONDecodeError:
        return proc.returncode, None, output


def evaluate(case_dir: Path, generated_module: Path, python: str, timeout: int) -> GeneratedEvaluation:
    notes: list[str] = []
    pdfs = sorted(case_dir.glob("*.pdf"))
    refs = sorted(case_dir.glob("*.py"))
    if not pdfs:
        raise FileNotFoundError(f"no PDF found in {case_dir}")
    if not refs:
        raise FileNotFoundError(f"no reference Python module found in {case_dir}")

    with tempfile.TemporaryDirectory() as temp_dir:
        text_path = Path(temp_dir) / "paper.txt"
        extract_code, _, extract_output = run_json(
            [
                sys.executable,
                str(SCRIPT_DIR / "extract_pdf_text.py"),
                str(pdfs[0]),
                "--max-pages",
                "4",
                "--output",
                str(text_path),
            ],
            timeout,
        )
        if extract_code != 0 or not text_path.exists():
            raise RuntimeError(f"PDF text extraction failed: {extract_output}")

        analysis = analyze(text_path.read_text(encoding="utf-8", errors="replace"))

    paper_candidates = [candidate.name for candidate in analysis.candidates]
    reference_symbols = top_level_symbols(refs[0])
    generated_symbols = top_level_symbols(generated_module)

    paper_matches = aligned(paper_candidates, generated_symbols + [generated_module.stem])
    paper_reference_matches = aligned(paper_candidates, reference_symbols + [refs[0].stem])
    reference_matches = aligned(generated_symbols, reference_symbols) or aligned(reference_symbols, generated_symbols)
    paper_alignment = "passed" if paper_matches or (paper_reference_matches and reference_matches) else "failed"
    reference_alignment = "passed" if reference_matches else "failed"
    if paper_alignment == "failed":
        notes.append("generated symbols do not align with paper candidates")
    if reference_alignment == "failed":
        notes.append("generated symbols do not align with reference sample symbols")

    smoke_code, smoke_data, smoke_output = run_json(
        [
            sys.executable,
            str(SCRIPT_DIR / "smoke_test_module.py"),
            str(generated_module),
            "--python",
            python,
            "--timeout",
            str(timeout),
            "--json",
        ],
        timeout + 15,
    )
    if smoke_data:
        smoke_classification = str(smoke_data.get("classification", "unknown"))
    else:
        smoke_classification = "tester_failed"
        notes.append(f"smoke tester failed: {smoke_output}")

    status = (
        "ok"
        if analysis.found_module in {"yes", "uncertain"}
        and paper_alignment == "passed"
        and reference_alignment == "passed"
        and smoke_classification in ACCEPTABLE_SMOKE
        else "needs_attention"
    )

    return GeneratedEvaluation(
        case_dir=str(case_dir),
        generated_module=str(generated_module),
        found_module=analysis.found_module,
        paper_candidates=paper_candidates[:8],
        reference_symbols=reference_symbols,
        generated_symbols=generated_symbols,
        paper_alignment=paper_alignment,
        reference_alignment=reference_alignment,
        smoke_classification=smoke_classification,
        status=status,
        notes=notes,
    )


def render_text(result: GeneratedEvaluation) -> str:
    return "\n".join(
        [
            f"status: {result.status}",
            f"case_dir: {result.case_dir}",
            f"generated_module: {result.generated_module}",
            f"found_module: {result.found_module}",
            f"paper_alignment: {result.paper_alignment}",
            f"reference_alignment: {result.reference_alignment}",
            f"smoke_classification: {result.smoke_classification}",
            f"paper_candidates: {', '.join(result.paper_candidates) if result.paper_candidates else 'none'}",
            f"reference_symbols: {', '.join(result.reference_symbols) if result.reference_symbols else 'none'}",
            f"generated_symbols: {', '.join(result.generated_symbols) if result.generated_symbols else 'none'}",
            f"notes: {'; '.join(result.notes) if result.notes else 'none'}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, help="Sample case directory containing PDF and reference .py.")
    parser.add_argument("generated_module", type=Path, help="Generated module to evaluate.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for smoke test.")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout for subprocess steps.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    result = evaluate(args.case_dir.resolve(), args.generated_module.resolve(), args.python, args.timeout)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(render_text(result))
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
