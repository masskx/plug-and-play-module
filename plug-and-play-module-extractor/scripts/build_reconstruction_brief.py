#!/usr/bin/env python3
"""Build a paper-only implementation brief for reconstructing a module."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_paper_modules import analyze, strip_extractor_metadata  # noqa: E402


IMPLEMENTATION_TERMS = (
    "input",
    "output",
    "dimension",
    "shape",
    "feature",
    "token",
    "channel",
    "sequence",
    "embedding",
    "hidden",
    "kernel",
    "stride",
    "padding",
    "softmax",
    "sigmoid",
    "normalization",
    "convolution",
    "pooling",
)


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def split_candidate_sentences(text: str) -> list[str]:
    normalized = " ".join(text.replace("-\n", "").split())
    parts: list[str] = []
    for piece in normalized.split(". "):
        piece = piece.strip()
        if len(piece) >= 40:
            parts.append(piece if piece.endswith(".") else piece + ".")
    return parts


def extract_implementation_hints(text: str) -> dict[str, list[str]]:
    sentences = split_candidate_sentences(text)
    shape_hints: list[str] = []
    equation_hints: list[str] = []
    structure_hints: list[str] = []
    equation_markers = ("+", "-", "*", "/", "softmax", "sum", "sigma", "log", "exp")

    for sentence in sentences:
        lowered = sentence.lower()
        if any(term in lowered for term in IMPLEMENTATION_TERMS) and len(shape_hints) < 8:
            shape_hints.append(sentence[:500])
        if (("=" in sentence and any(symbol in lowered for symbol in equation_markers)) or "equation" in lowered) and len(
            equation_hints
        ) < 8:
            equation_hints.append(sentence[:500])
        if any(term in lowered for term in ("figure", "algorithm", "ablation", "module", "block")) and len(
            structure_hints
        ) < 8:
            structure_hints.append(sentence[:500])

    return {
        "shape_or_operation_hints": shape_hints,
        "equation_hints": equation_hints,
        "structure_hints": structure_hints,
    }


def render_hint_list(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item.replace(chr(10), ' ')}" for item in items)


def render_markdown(text_path: Path, max_candidates: int = 8) -> str:
    raw_text = text_path.read_text(encoding="utf-8", errors="replace")
    analysis = analyze(raw_text)
    clean_text = strip_extractor_metadata(raw_text)
    hints = extract_implementation_hints(clean_text)
    candidate_lines = []
    for candidate in analysis.candidates[:max_candidates]:
        evidence = " | ".join(candidate.evidence[:2]).replace("\n", " ")
        candidate_lines.append(f"- `{candidate.name}` score={candidate.score}: {evidence}")
    if not candidate_lines:
        candidate_lines.append("- none")

    evidence_lines = [f"- {sentence.replace(chr(10), ' ')}" for sentence in analysis.evidence[:8]]
    if not evidence_lines:
        evidence_lines.append("- none")

    section_hints = []
    lowered = clean_text.lower()
    for marker in ("abstract", "introduction", "method", "approach", "module", "ablation"):
        index = lowered.find(marker)
        if index >= 0:
            snippet = clean_text[max(0, index - 120) : index + 500]
            section_hints.append(f"### Around `{marker}`\n\n{snippet.strip()}")
    if not section_hints:
        section_hints.append("No method-section hints found in extracted text.")

    return "\n\n".join(
        [
            "# Paper-Only Reconstruction Brief",
            f"source_text: `{text_path}`",
            f"found_module: `{analysis.found_module}`",
            f"title: {analysis.title or 'unknown'}",
            f"score: {analysis.score}",
            "## Candidate Modules",
            "\n".join(candidate_lines),
            "## Evidence Sentences",
            "\n".join(evidence_lines),
            "## Implementation Hints",
            "### Shape Or Operation Hints",
            render_hint_list(hints["shape_or_operation_hints"]),
            "### Equation Hints",
            render_hint_list(hints["equation_hints"]),
            "### Structure Hints",
            render_hint_list(hints["structure_hints"]),
            "## Implementation Notes To Fill Before Coding",
            "\n".join(
                [
                    "- selected_module: choose the highest-evidence candidate or the user-requested module.",
                    "- implementation_basis: paper-derived.",
                    "- expected_inputs: infer tensor layout from figures, equations, or domain conventions.",
                    "- expected_outputs: preserve module-level transformation shape when possible.",
                    "- unresolved_assumptions: list missing equation, hyperparameter, or tensor-shape details.",
                    "- smoke_test: create CPU random tensors and print input/output shapes.",
                ]
            ),
            "## Section Hints",
            "\n\n".join(section_hints[:4]),
        ]
    ) + "\n"


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=Path, help="Extracted paper text file.")
    parser.add_argument("--output", "-o", type=Path, help="Write Markdown brief to this file.")
    parser.add_argument("--json", action="store_true", help="Emit the underlying analysis JSON.")
    args = parser.parse_args(argv)

    if not args.text.exists():
        parser.error(f"text file not found: {args.text}")

    if args.json:
        result = analyze(args.text.read_text(encoding="utf-8", errors="replace"))
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0 if result.found_module != "no" else 1

    markdown = render_markdown(args.text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote reconstruction brief: {args.output}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
