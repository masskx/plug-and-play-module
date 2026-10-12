#!/usr/bin/env python3
"""Review gate: decide whether any usable plug-and-play module exists *before* writing code.

Combines the paper analysis, the code scan, and the paper/code alignment into one verdict
for the four supported situations (pdf-plus-code, pdf-only, code-only, title-only).

The verdict is advisory evidence for the agent, which must still read the method section /
the candidate source (see references/module_detection.md). When the verdict is ``no`` the
script prints the standard no-usable-module report, and nothing else should be produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_paper_modules import analyze, strip_extractor_metadata  # noqa: E402
from find_module_candidates import iter_python_files, ranked, scan_file  # noqa: E402
from match_paper_to_code import alignment_label, match  # noqa: E402

NO_MODULE_EN = "After review, no usable plug-and-play modules were found in {subject}."
NO_MODULE_ZH = "经审查，{subject}中未发现可用的即插即用模块。"


@dataclass
class Gate:
    source_mode: str
    found_module: str
    implementation_basis: str
    subject: str
    reasons: list[str] = field(default_factory=list)
    paper: dict | None = None
    code: dict | None = None
    alignment: dict | None = None
    targets: list[dict] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    manual_review: list[str] = field(default_factory=list)
    report: str = ""


MANUAL_REVIEW_CHECKLIST = [
    "Read the method section (and appendix/supplement) end to end; the script only sees keywords and structure.",
    "For each candidate: write down its input tensor(s), output tensor(s), and the operations in between.",
    "Reject candidates that are standard layers, whole models, losses tied to one dataset, or need undisclosed details.",
    "Check that the module can be inserted into another model without the rest of the paper's pipeline.",
]


def subject_of(paper_text: str | None, code_root: Path | None, title: str | None) -> str:
    if paper_text:
        analysis_title = analyze(paper_text).title
        if analysis_title:
            return f"“{analysis_title}”"
    if title:
        return f"“{title}”"
    if code_root:
        return f"`{code_root.name}`"
    return "the provided material"


def scan_code(code_root: Path):
    scan_root = code_root if code_root.is_dir() else code_root.parent
    summaries = [scan_file(path, scan_root) for path in sorted(iter_python_files(code_root))]
    return summaries, ranked(summaries)


def target_dict(candidate, score: float | None = None) -> dict:
    data = {
        "name": candidate.name,
        "kind": candidate.kind,
        "path": candidate.path,
        "line": candidate.line,
        "tier": candidate.tier,
        "plug_score": candidate.plug_score,
        "risks": candidate.risks,
    }
    if score is not None:
        data["alignment_score"] = score
    return data


def no_module_report(subject: str, reasons: list[str], lang: str, rejected: list[str]) -> str:
    sentence = (NO_MODULE_ZH if lang == "zh" else NO_MODULE_EN).format(subject=subject)
    lines = [sentence]
    if reasons:
        lines.append("")
        lines.append("依据：" if lang == "zh" else "Basis:")
        lines += [f"- {reason}" for reason in reasons[:4]]
    if rejected:
        lines.append("")
        lines.append("已排除的候选：" if lang == "zh" else "Candidates considered and rejected:")
        lines += [f"- {item}" for item in rejected[:5]]
    return "\n".join(lines)


def decide_with_code_and_paper(paper_verdict: str, code_verdict: str, align: str) -> tuple[str, str, str]:
    """Returns (found_module, basis, reason) for the pdf-plus-code mode."""
    aligned = align in {"matched", "weak"} and code_verdict != "none"
    if paper_verdict == "yes":
        if aligned and align == "matched":
            return "yes", "source-code-derived", "paper module found and the code implements a matching symbol"
        if code_verdict != "none":
            return (
                "yes",
                "mixed",
                "paper describes a module but no code symbol clearly matches its name; inspect the code, "
                "otherwise reconstruct from the paper and label the result mixed/paper-derived",
            )
        return "yes", "paper-derived", "paper describes a module; the supplied code has no component candidates"
    if paper_verdict == "uncertain":
        if code_verdict == "strong" and aligned:
            return "yes", "source-code-derived", "code resolves the paper's under-specified module"
        if code_verdict != "none":
            return "uncertain", "mixed", "paper detail is thin; decide from the code after manual inspection"
        return "uncertain", "none", "paper detail is thin and the code offers no candidate"
    # paper_verdict == "no"
    if code_verdict == "strong":
        return (
            "uncertain",
            "source-code-derived",
            "paper does not present a separable module, but the code contains strong reusable components; "
            "decide after manual review whether they are meaningful on their own",
        )
    return "no", "none", "neither the paper nor the code presents a separable reusable component"


def run_gate(
    paper_text: str | None,
    code_root: Path | None,
    title: str | None,
    lang: str = "en",
    mode_override: str | None = None,
) -> Gate:
    if paper_text is not None and not paper_text.strip():
        paper_text = None
    has_paper = paper_text is not None
    has_code = code_root is not None

    if mode_override:
        mode = mode_override
    elif has_paper and has_code:
        mode = "pdf-plus-code"
    elif has_paper:
        mode = "pdf-only"
    elif has_code:
        mode = "code-only"
    elif title:
        mode = "title-only"
    else:
        mode = "none"

    subject = subject_of(paper_text, code_root, title)
    gate = Gate(source_mode=mode, found_module="uncertain", implementation_basis="none", subject=subject)

    if mode == "none":
        gate.reasons = ["no paper, code, or title was provided"]
        gate.next_steps = ["Ask the user for a paper (PDF / arXiv link / title) and/or code."]
        return gate

    if mode == "title-only":
        gate.reasons = [
            "only a title was provided and the paper text has not been retrieved; a 'no module' verdict is not possible without reviewing the paper"
        ]
        gate.next_steps = [
            f'python scripts/resolve_paper_by_title.py --title "{title}" --download-dir ./paper --github',
            "If the network is blocked, find the arXiv page / official repo with WebSearch+WebFetch.",
            "Then rerun: python scripts/review_gate.py --paper-text paper.txt [--code-root ./code]",
            "If the paper cannot be verified, stop and ask the user for the PDF; never reconstruct from the title alone.",
        ]
        gate.manual_review = ["Title resolution must reach status=resolved before any module is extracted."]
        return gate

    paper_info = None
    code_info = None
    rejected: list[str] = []

    paper_verdict = "none"
    if has_paper:
        analysis = analyze(strip_extractor_metadata(paper_text))
        paper_verdict = analysis.found_module
        paper_info = {
            "found_module": analysis.found_module,
            "title": analysis.title,
            "contribution_type": analysis.contribution_type,
            "structural_score": analysis.structural.score if analysis.structural else None,
            "candidates": [c.name for c in analysis.candidates[:8]],
            "reasons": analysis.reasons,
            "plug_claims": analysis.structural.plug_claims if analysis.structural else [],
        }
        gate.paper = paper_info

    code_verdict = "none"
    code_candidates = []
    if has_code:
        _, code_candidates = scan_code(code_root)
        tiers = {c.tier for c in code_candidates}
        code_verdict = "strong" if "strong" in tiers else "possible" if "possible" in tiers else "none"
        code_info = {
            "verdict": code_verdict,
            "top": [target_dict(c) for c in code_candidates[:8]],
        }
        gate.code = code_info
        rejected = [
            f"{c.name} ({c.path}): {c.risks[0]}" for c in code_candidates if c.tier == "weak" and c.risks
        ]

    if mode == "pdf-only":
        found = paper_verdict
        gate.found_module = found
        gate.implementation_basis = "paper-derived" if found != "no" else "none"
        gate.reasons = paper_info["reasons"] if paper_info else []
        gate.targets = [{"name": name, "kind": "paper-candidate"} for name in (paper_info or {}).get("candidates", [])[:4]]
    elif mode == "code-only":
        gate.found_module = {"strong": "yes", "possible": "uncertain", "none": "no"}[code_verdict]
        gate.implementation_basis = "source-code-derived" if gate.found_module != "no" else "none"
        gate.reasons = {
            "strong": ["code contains self-contained tensor-in/tensor-out components (tier=strong)"],
            "possible": ["only borderline components found (tier=possible); inspect the risks before extracting"],
            "none": ["no class or function in the code looks like a reusable module"],
        }[code_verdict]
        gate.targets = [target_dict(c) for c in code_candidates if c.tier in {"strong", "possible"}][:6]
    else:  # pdf-plus-code
        names = list((paper_info or {}).get("candidates", []))
        if paper_info and paper_info.get("title"):
            names.append(str(paper_info["title"]).split(":")[0].strip())
        matches = match(names, code_candidates) if names else []
        align = alignment_label(matches)
        gate.alignment = {"label": align, "matches": [asdict(m) for m in matches[:6]]}
        found, basis, reason = decide_with_code_and_paper(paper_verdict, code_verdict, align)
        gate.found_module = found
        gate.implementation_basis = basis
        gate.reasons = [reason] + (paper_info["reasons"] if paper_info else [])
        by_key = {(c.path, c.name): c for c in code_candidates}
        for m in matches[:3]:
            if m.score >= 0.6 and (m.code_path, m.code_symbol) in by_key:
                gate.targets.append(target_dict(by_key[(m.code_path, m.code_symbol)], m.score))
        if not gate.targets:
            gate.targets = [target_dict(c) for c in code_candidates if c.tier == "strong"][:4]

    # Next steps and report.
    if gate.found_module == "no":
        if paper_info:
            kind = paper_info["contribution_type"]
            why = f"it is a {kind}, not a reusable component" if kind != "unknown" else "no concrete operations, shapes, or plug-in claim"
            rejected = [f"{name} (named in the paper; {why})" for name in paper_info["candidates"][:3]] + rejected
        gate.targets = []
        gate.report = no_module_report(subject, gate.reasons, lang, rejected)
        gate.next_steps = ["Stop. Output the report above verbatim; create no code and no files."]
    else:
        gate.manual_review = MANUAL_REVIEW_CHECKLIST
        unaligned = mode == "pdf-plus-code" and (gate.alignment or {}).get("label") == "none"
        if unaligned:
            gate.next_steps.append(
                "Code candidates do not match the paper's module name: confirm by reading the code. If the paper's "
                "module is not implemented there, reconstruct it from the paper (basis: mixed/paper-derived)."
            )
        if mode in {"code-only", "pdf-plus-code"} and gate.targets and not unaligned:
            top = gate.targets[0]
            if top.get("kind") != "paper-candidate":
                gate.next_steps.append(
                    f"python scripts/extract_python_module.py {top['path']} --name {top['name']} --include-main "
                    "--root <code-root> --prune-imports --output <module.py>"
                )
        if gate.implementation_basis in {"paper-derived", "mixed"}:
            gate.next_steps.append("python scripts/build_reconstruction_brief.py paper.txt --output brief.md")
        gate.next_steps.append("python scripts/check_module_contract.py <module.py> --run --python <python-with-torch>")
        if gate.found_module == "uncertain":
            gate.next_steps.insert(0, "Resolve the uncertainty by manual review first; label any output as uncertain.")
    return gate


def render_text(gate: Gate) -> str:
    lines = [
        f"found_module: {gate.found_module}",
        f"source_mode: {gate.source_mode}",
        f"implementation_basis: {gate.implementation_basis}",
        f"subject: {gate.subject}",
        "reasons:",
    ]
    lines += [f"  - {reason}" for reason in gate.reasons] or ["  - none"]
    if gate.paper:
        lines.append(
            f"paper: verdict={gate.paper['found_module']} contribution={gate.paper['contribution_type']} "
            f"structural_score={gate.paper['structural_score']} candidates={', '.join(gate.paper['candidates'][:5]) or 'none'}"
        )
    if gate.code:
        lines.append(f"code: verdict={gate.code['verdict']}")
    if gate.alignment:
        lines.append(f"paper_code_alignment: {gate.alignment['label']}")
    if gate.targets:
        lines.append("targets:")
        for target in gate.targets:
            where = f" ({target['path']}:{target['line']}, tier={target['tier']}, plug_score={target['plug_score']})" if "path" in target else ""
            lines.append(f"  - {target['name']}{where}")
            for risk in target.get("risks", [])[:3]:
                lines.append(f"      risk: {risk}")
    if gate.manual_review:
        lines.append("manual_review:")
        lines += [f"  - {item}" for item in gate.manual_review]
    lines.append("next_steps:")
    lines += [f"  - {step}" for step in gate.next_steps] or ["  - none"]
    if gate.report:
        lines += ["", "--- report ---", gate.report]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-text", type=Path, help="Extracted paper text (see extract_pdf_text.py).")
    parser.add_argument("--code-root", type=Path, help="Code repository root or single .py file.")
    parser.add_argument("--title", help="Paper title when nothing else is available.")
    parser.add_argument("--lang", choices=["en", "zh"], default="en", help="Language of the no-module report.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    paper_text = None
    if args.paper_text:
        if not args.paper_text.exists():
            parser.error(f"paper text not found: {args.paper_text}")
        paper_text = args.paper_text.read_text(encoding="utf-8", errors="replace")
    if args.code_root and not args.code_root.exists():
        parser.error(f"code root not found: {args.code_root}")

    gate = run_gate(paper_text, args.code_root.resolve() if args.code_root else None, args.title, args.lang)
    if args.json:
        print(json.dumps(asdict(gate), ensure_ascii=False, indent=2))
    else:
        print(render_text(gate))
    if gate.source_mode in {"title-only", "none"}:
        return 2
    return {"yes": 0, "uncertain": 0, "no": 1}[gate.found_module]


if __name__ == "__main__":
    raise SystemExit(main())
