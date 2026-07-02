#!/usr/bin/env python3
"""Run repeatable regression checks over plug-and-play module sample cases."""

from __future__ import annotations

import argparse
import json
import re
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
class CaseResult:
    case: str
    pdf_count: int
    pdf_text_status: str
    pdf_text_chars: int
    paper_found_module: str
    paper_candidate_count: int
    paper_alignment: str
    paper_alignment_matches: list[str]
    brief_status: str
    py_count: int
    candidate_count: int
    has_main: bool
    smoke_classification: str
    extracted_smoke_classification: str
    named_extract_target: str
    named_extract_smoke_classification: str
    interface_status: str
    status: str
    notes: list[str]


def run_json(command: list[str], timeout: int) -> tuple[int, object | None, str]:
    proc = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    output = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
    try:
        return proc.returncode, json.loads(stdout), output
    except json.JSONDecodeError:
        if proc.returncode != 0:
            return proc.returncode, None, output
        return proc.returncode, None, output


def evaluate_case(
    case_dir: Path,
    scanner: Path,
    smoke_tester: Path,
    pdf_extractor: Path,
    module_extractor: Path,
    paper_analyzer: Path,
    brief_builder: Path,
    interface_comparator: Path,
    smoke_python: str,
    timeout: int,
    check_pdf_text: bool,
    check_extraction: bool,
) -> CaseResult:
    notes: list[str] = []
    pdfs = sorted(case_dir.glob("*.pdf"))
    py_files = sorted(case_dir.glob("*.py"))
    candidate_count = 0
    has_main = False
    pdf_text_status = "not-run"
    pdf_text_chars = 0
    paper_found_module = "not-run"
    paper_candidate_count = 0
    paper_candidate_names: list[str] = []
    paper_alignment = "not-run"
    paper_alignment_matches: list[str] = []
    brief_status = "not-run"
    smoke_classification = "not-run"
    extracted_smoke_classification = "not-run"
    named_extract_target = "none"
    named_extract_smoke_classification = "not-run"
    interface_status = "not-run"

    if not pdfs:
        notes.append("missing PDF")
    if not py_files:
        notes.append("missing Python module")

    if check_pdf_text and pdfs:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "paper.txt"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(pdf_extractor),
                    str(pdfs[0]),
                    "--max-pages",
                    "4",
                    "--output",
                    str(output_path),
                ],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
            )
            if proc.returncode == 0 and output_path.exists():
                text = output_path.read_text(encoding="utf-8", errors="replace")
                pdf_text_chars = len(text.strip())
                pdf_text_status = "passed" if pdf_text_chars > 0 else "empty"
                if pdf_text_status == "empty":
                    notes.append("PDF text extraction returned empty text")
                else:
                    analysis_code, analysis_data, analysis_output = run_json(
                        [
                            sys.executable,
                            str(paper_analyzer),
                            str(output_path),
                            "--json",
                        ],
                        timeout,
                    )
                    if analysis_code != 0 and analysis_data is None:
                        paper_found_module = "analysis_failed"
                        notes.append(f"paper analysis failed: {analysis_output}")
                    elif isinstance(analysis_data, dict):
                        paper_found_module = str(analysis_data.get("found_module", "unknown"))
                        candidates = analysis_data.get("candidates", [])
                        if isinstance(candidates, list):
                            paper_candidate_count = len(candidates)
                            for candidate in candidates:
                                if isinstance(candidate, dict) and isinstance(candidate.get("name"), str):
                                    paper_candidate_names.append(candidate["name"])

                    brief_path = Path(temp_dir) / "brief.md"
                    brief_proc = subprocess.run(
                        [
                            sys.executable,
                            str(brief_builder),
                            str(output_path),
                            "--output",
                            str(brief_path),
                        ],
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        capture_output=True,
                        timeout=timeout,
                    )
                    if brief_proc.returncode == 0 and brief_path.exists():
                        brief_text = brief_path.read_text(encoding="utf-8", errors="replace")
                        required_markers = [
                            "found_module:",
                            "## Candidate Modules",
                            "## Evidence Sentences",
                            "## Implementation Hints",
                            "## Implementation Notes To Fill Before Coding",
                            "smoke_test:",
                        ]
                        evidence_section = brief_text.split("## Evidence Sentences", 1)[-1].split("##", 1)[0]
                        hints_section = brief_text.split("## Implementation Hints", 1)[-1].split(
                            "## Implementation Notes To Fill Before Coding", 1
                        )[0]
                        brief_status = (
                            "passed"
                            if all(marker in brief_text for marker in required_markers)
                            and "- none" not in evidence_section
                            and hints_section.count("- none") < 3
                            else "incomplete"
                        )
                        if brief_status != "passed":
                            notes.append("reconstruction brief missing required content")
                    else:
                        brief_status = "failed"
                        detail = "\n".join(
                            part for part in (brief_proc.stdout.strip(), brief_proc.stderr.strip()) if part
                        )
                        notes.append(f"reconstruction brief failed: {detail}")
            else:
                pdf_text_status = "failed"
                detail = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
                notes.append(f"PDF text extraction failed: {detail}")
    elif check_pdf_text:
        pdf_text_status = "missing_pdf"
        paper_found_module = "missing_pdf"

    scan_code, scan_data, scan_output = run_json(
        [sys.executable, str(scanner), str(case_dir), "--json"],
        timeout,
    )
    if scan_code != 0:
        notes.append(f"candidate scan failed: {scan_output}")
    elif isinstance(scan_data, list):
        for summary in scan_data:
            if isinstance(summary, dict):
                candidate_count += len(summary.get("candidates", []))
                has_main = has_main or bool(summary.get("has_main"))

    if py_files:
        smoke_code, smoke_data, smoke_output = run_json(
            [
                sys.executable,
                str(smoke_tester),
                str(py_files[0]),
                "--python",
                smoke_python,
                "--timeout",
                str(timeout),
                "--json",
            ],
            timeout + 15,
        )
        if smoke_code != 0 and smoke_data is None:
            notes.append(f"smoke tester failed: {smoke_output}")
            smoke_classification = "tester_failed"
        elif isinstance(smoke_data, dict):
            smoke_classification = str(smoke_data.get("classification", "unknown"))

    if check_extraction and py_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_path = Path(temp_dir) / f"case_{case_dir.name}_extracted.py"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(module_extractor),
                    str(py_files[0]),
                    "--output",
                    str(extracted_path),
                    "--include-all-helpers",
                    "--include-main",
                ],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
            )
            if proc.returncode != 0 or not extracted_path.exists():
                extracted_smoke_classification = "extract_failed"
                detail = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
                notes.append(f"module extraction failed: {detail}")
            else:
                extract_smoke_code, extract_smoke_data, extract_smoke_output = run_json(
                    [
                        sys.executable,
                        str(smoke_tester),
                        str(extracted_path),
                        "--python",
                        smoke_python,
                        "--timeout",
                        str(timeout),
                        "--json",
                    ],
                    timeout + 15,
                )
                if extract_smoke_code != 0 and extract_smoke_data is None:
                    extracted_smoke_classification = "tester_failed"
                    notes.append(f"extracted smoke tester failed: {extract_smoke_output}")
                elif isinstance(extract_smoke_data, dict):
                    extracted_smoke_classification = str(extract_smoke_data.get("classification", "unknown"))

                compare_code, compare_data, compare_output = run_json(
                    [
                        sys.executable,
                        str(interface_comparator),
                        str(py_files[0]),
                        str(extracted_path),
                        "--json",
                    ],
                    timeout,
                )
                if compare_code != 0 and compare_data is None:
                    interface_status = "compare_failed"
                    notes.append(f"interface comparison failed: {compare_output}")
                elif isinstance(compare_data, dict):
                    interface_status = str(compare_data.get("status", "unknown"))

            named_extract_target = choose_named_extract_target(py_files[0], paper_candidate_names)
            if named_extract_target != "none":
                named_path = Path(temp_dir) / f"case_{case_dir.name}_{named_extract_target}_named.py"
                named_proc = subprocess.run(
                    [
                        sys.executable,
                        str(module_extractor),
                        str(py_files[0]),
                        "--output",
                        str(named_path),
                        "--name",
                        named_extract_target,
                        "--include-main",
                    ],
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    timeout=timeout,
                )
                if named_proc.returncode != 0 or not named_path.exists():
                    named_extract_smoke_classification = "extract_failed"
                    detail = "\n".join(part for part in (named_proc.stdout.strip(), named_proc.stderr.strip()) if part)
                    notes.append(f"named extraction failed for {named_extract_target}: {detail}")
                else:
                    named_smoke_code, named_smoke_data, named_smoke_output = run_json(
                        [
                            sys.executable,
                            str(smoke_tester),
                            str(named_path),
                            "--python",
                            smoke_python,
                            "--timeout",
                            str(timeout),
                            "--json",
                        ],
                        timeout + 15,
                    )
                    if named_smoke_code != 0 and named_smoke_data is None:
                        named_extract_smoke_classification = "tester_failed"
                        notes.append(f"named extraction smoke tester failed: {named_smoke_output}")
                    elif isinstance(named_smoke_data, dict):
                        named_extract_smoke_classification = str(named_smoke_data.get("classification", "unknown"))

    if py_files and candidate_count == 0:
        notes.append("no module candidates found")
    if py_files and not has_main:
        notes.append("module lacks __main__ smoke test")

    if py_files and paper_candidate_names:
        alignment_targets = source_alignment_targets(py_files[0])
        paper_alignment_matches = aligned_candidates(paper_candidate_names, alignment_targets)
        paper_alignment = "passed" if paper_alignment_matches else "failed"
        if paper_alignment == "failed":
            notes.append("paper candidates do not align with source symbols or filename")
    elif py_files and check_pdf_text:
        paper_alignment = "failed"
        notes.append("no paper candidates available for alignment")

    acceptable_smoke = {
        "passed",
        "missing_dependency",
        "accelerator_unavailable",
        "not_run",
    }
    if not py_files:
        status = "skipped"
    elif (
        candidate_count > 0
        and has_main
        and smoke_classification in acceptable_smoke
        and (not check_extraction or extracted_smoke_classification in acceptable_smoke)
        and (not check_extraction or named_extract_smoke_classification in acceptable_smoke)
        and (not check_extraction or interface_status == "passed")
        and (not check_pdf_text or pdf_text_status == "passed")
        and (not check_pdf_text or paper_found_module in {"yes", "uncertain"})
        and (not check_pdf_text or paper_alignment == "passed")
        and (not check_pdf_text or brief_status == "passed")
    ):
        status = "ok"
    else:
        status = "needs_attention"

    return CaseResult(
        case=case_dir.name,
        pdf_count=len(pdfs),
        pdf_text_status=pdf_text_status,
        pdf_text_chars=pdf_text_chars,
        paper_found_module=paper_found_module,
        paper_candidate_count=paper_candidate_count,
        paper_alignment=paper_alignment,
        paper_alignment_matches=paper_alignment_matches[:5],
        brief_status=brief_status,
        py_count=len(py_files),
        candidate_count=candidate_count,
        has_main=has_main,
        smoke_classification=smoke_classification,
        extracted_smoke_classification=extracted_smoke_classification,
        named_extract_target=named_extract_target,
        named_extract_smoke_classification=named_extract_smoke_classification,
        interface_status=interface_status,
        status=status,
        notes=notes,
    )


def render_text(results: list[CaseResult]) -> str:
    lines: list[str] = []
    totals = {
        "ok": sum(1 for result in results if result.status == "ok"),
        "skipped": sum(1 for result in results if result.status == "skipped"),
        "needs_attention": sum(1 for result in results if result.status == "needs_attention"),
    }
    lines.append(
        "summary: "
        f"ok={totals['ok']} skipped={totals['skipped']} needs_attention={totals['needs_attention']}"
    )
    lines.append("")
    for result in results:
        note_text = "; ".join(result.notes) if result.notes else "none"
        lines.append(
            f"case {result.case}: status={result.status} pdfs={result.pdf_count} "
            f"pdf_text={result.pdf_text_status} chars={result.pdf_text_chars} "
            f"paper_found={result.paper_found_module} paper_candidates={result.paper_candidate_count} "
            f"paper_alignment={result.paper_alignment} brief={result.brief_status} "
            f"py={result.py_count} candidates={result.candidate_count} "
            f"main={str(result.has_main).lower()} smoke={result.smoke_classification} "
            f"extracted_smoke={result.extracted_smoke_classification} "
            f"named_target={result.named_extract_target} named_smoke={result.named_extract_smoke_classification} "
            f"interface={result.interface_status} "
            f"notes={note_text}"
        )
    return "\n".join(lines)


GENERIC_ALIGNMENT_TOKENS = {
    "a",
    "an",
    "and",
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
    "of",
    "the",
    "time",
    "to",
}

SPECIAL_MODULE_TOKENS = {
    "attention",
    "conv",
    "decomposition",
    "filter",
    "fusion",
    "loss",
    "mixing",
    "selector",
    "token",
}


def compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def tokens(value: str) -> set[str]:
    split_camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    parts = re.split(r"[^A-Za-z0-9]+", split_camel)
    return {
        part.lower()
        for part in parts
        if len(part) >= 3 and part.lower() not in GENERIC_ALIGNMENT_TOKENS
    }


def raw_tokens(value: str) -> set[str]:
    split_camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    parts = re.split(r"[^A-Za-z0-9]+", split_camel)
    return {part.lower() for part in parts if len(part) >= 3}


def source_alignment_targets(py_file: Path) -> list[str]:
    targets = [py_file.stem]
    try:
        source = py_file.read_text(encoding="utf-8", errors="replace")
        tree = __import__("ast").parse(source, filename=str(py_file))
        for node in tree.body:
            if hasattr(node, "name"):
                targets.append(str(node.name))
    except Exception:
        pass
    return targets


def aligned_candidates(candidates: list[str], targets: list[str]) -> list[str]:
    matches: list[str] = []
    target_compacts = [(target, compact(target), tokens(target)) for target in targets]
    for candidate in candidates:
        candidate_compact = compact(candidate)
        candidate_tokens = tokens(candidate)
        candidate_raw_tokens = raw_tokens(candidate)
        if len(candidate_compact) < 3:
            continue
        for target, target_compact, target_tokens in target_compacts:
            if not target_compact:
                continue
            target_raw_tokens = raw_tokens(target)
            direct_match = candidate_compact in target_compact or target_compact in candidate_compact
            token_match = len(candidate_tokens & target_tokens) >= 1 and (
                len(candidate_tokens & target_tokens) >= 2
                or len(candidate_compact) >= 5
                or len(target_compact) <= 8
            )
            special_match = bool(candidate_raw_tokens & target_raw_tokens & SPECIAL_MODULE_TOKENS) and (
                len(candidate_compact) >= 8 or len(target_compact) >= 8
            )
            if direct_match or token_match or special_match:
                matches.append(f"{candidate}->{target}")
                break
    return matches


def choose_named_extract_target(py_file: Path, paper_candidates: list[str]) -> str:
    targets = source_alignment_targets(py_file)
    if not targets:
        return "none"

    symbol_targets = targets[1:] if len(targets) > 1 else targets
    for candidate in paper_candidates:
        for symbol in symbol_targets:
            if aligned_candidates([candidate], [symbol]):
                return symbol

    for symbol in reversed(symbol_targets):
        if not symbol.startswith("_") and symbol.lower() not in {"configs", "mockconfigs"}:
            return symbol
    return symbol_targets[-1] if symbol_targets else "none"


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    skill_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_root", type=Path, help="Directory containing numbered sample case folders.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run sample modules.")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout for each scanner/smoke subprocess.")
    parser.add_argument("--skip-pdf-text", action="store_true", help="Do not check first-page PDF text extraction.")
    parser.add_argument("--skip-extraction", action="store_true", help="Do not extract source modules and smoke-test the extracted copy.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    samples_root = args.samples_root.resolve()
    if not samples_root.exists():
        parser.error(f"samples root not found: {samples_root}")

    case_dirs = [path for path in samples_root.iterdir() if path.is_dir()]
    case_dirs.sort(key=lambda path: int(path.name) if path.name.isdigit() else path.name)

    scanner = skill_root / "scripts" / "find_module_candidates.py"
    smoke_tester = skill_root / "scripts" / "smoke_test_module.py"
    pdf_extractor = skill_root / "scripts" / "extract_pdf_text.py"
    module_extractor = skill_root / "scripts" / "extract_python_module.py"
    paper_analyzer = skill_root / "scripts" / "analyze_paper_modules.py"
    brief_builder = skill_root / "scripts" / "build_reconstruction_brief.py"
    interface_comparator = skill_root / "scripts" / "compare_module_interface.py"
    results = [
        evaluate_case(
            case_dir,
            scanner,
            smoke_tester,
            pdf_extractor,
            module_extractor,
            paper_analyzer,
            brief_builder,
            interface_comparator,
            args.python,
            args.timeout,
            not args.skip_pdf_text,
            not args.skip_extraction,
        )
        for case_dir in case_dirs
    ]

    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        print(render_text(results))

    return 1 if any(result.status == "needs_attention" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
