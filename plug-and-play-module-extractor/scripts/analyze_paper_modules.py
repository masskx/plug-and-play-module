#!/usr/bin/env python3
"""Analyze extracted paper text for plug-and-play module candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


POSITIVE_TERMS = {
    "module",
    "block",
    "layer",
    "attention",
    "adapter",
    "filter",
    "loss",
    "fusion",
    "selector",
    "selection",
    "token",
    "normalization",
    "decomposition",
    "mixing",
    "mixer",
    "plug-and-play",
    "plug and play",
    "can be inserted",
    "inserted into",
    "lightweight",
    "ablation",
}

NEGATIVE_TERMS = {
    "dataset",
    "benchmark",
    "survey",
    "annotation",
    "challenge",
    "leaderboard",
    "evaluation protocol",
}

MODULE_NOUNS = (
    "module",
    "block",
    "layer",
    "attention",
    "adapter",
    "filter",
    "loss",
    "fusion",
    "selector",
    "selection",
    "mixing",
    "mixer",
    "decomposition",
    "network",
)

STOP_CANDIDATES = {
    "ABSTRACT",
    "ADAM",
    "ASPP",
    "CNN",
    "CV",
    "DETAILS",
    "ELU",
    "EMD",
    "FFT",
    "FMCW",
    "FOR",
    "GPT",
    "ICLR",
    "IEEE",
    "IFFT",
    "IMPLEMENTATION",
    "LINEAR",
    "LSTM",
    "MLP",
    "MSE",
    "PDF",
    "RNN",
    "SOTA",
    "TRANSFORMERS",
    "VISION",
    "A",
    "AN",
    "AND",
    "AS",
    "BY",
    "CAN",
    "DURING",
    "FOR",
    "FUNCTION",
    "IN",
    "IS",
    "IT",
    "MECHANISM",
    "MECHANISMS",
    "MODULE",
    "OF",
    "ON",
    "THE",
    "TO",
    "TWO",
    "UNDER",
    "WHICH",
}


# Section headings and affiliation words that leak into capitalized-phrase candidates.
NOISE_TOKENS = {"abstract", "introduction", "conclusion", "university", "department", "science", "institute", "section", "figure", "table"}


@dataclass
class ModuleCandidate:
    name: str
    score: int
    evidence: list[str]


@dataclass
class StructuralEvidence:
    """Signals that a described component is concrete enough to implement."""

    plug_claims: list[str]
    equation_lines: int
    shape_mentions: int
    algorithm_mentions: int
    figure_module_mentions: int
    module_ablation: bool
    operation_terms: list[str]
    score: int


@dataclass
class PaperAnalysis:
    found_module: str
    title: str
    score: int
    positive_hits: list[str]
    negative_hits: list[str]
    candidates: list[ModuleCandidate]
    evidence: list[str]
    contribution_type: str = "unknown"
    structural: StructuralEvidence | None = None
    reasons: list[str] | None = None


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def normalize_space(text: str) -> str:
    text = text.replace("-\n", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_extractor_metadata(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            kept.append(line)
            continue
        if lowered in {"# extracted pdf text"}:
            continue
        if lowered.startswith(("source:", "backend:", "pages:", "## page")):
            continue
        kept.append(line)
    return "\n".join(kept)


def split_sentences(text: str) -> list[str]:
    normalized = normalize_space(text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [part.strip() for part in parts if len(part.strip()) > 20]


TITLE_WORDS = {"a", "an", "the", "of", "for", "with", "via", "in", "on", "to", "and", "by", "from", "towards", "toward"}


def looks_like_author_line(line: str) -> bool:
    """'Jane Doe, John Roe' or 'A. Smith and B. Jones' -- names only, no title vocabulary."""
    chunks = [chunk.strip(" *†‡§¶0123456789") for chunk in re.split(r",|\band\b|&", line) if chunk.strip(" *†‡§¶0123456789")]
    if len(chunks) < 2 and not re.fullmatch(r"[A-Z][\w.'-]*(?: [A-Z][\w.'-]*){1,3}", line.strip(" *†‡§¶0123456789")):
        return False
    if ":" in line:
        return False
    for chunk in chunks:
        words = chunk.split()
        if not 2 <= len(words) <= 4:
            return False
        if any(word.lower() in TITLE_WORDS or not word[0].isupper() for word in words):
            return False
    return True


def extract_title(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [
        line
        for line in lines
        if not line.lower().startswith(("source:", "backend:", "pages:", "## page", "# extracted"))
    ]
    if not lines:
        return ""
    title_parts: list[str] = []
    for index, line in enumerate(lines[:8]):
        if re.search(r"\b(abstract|introduction)\b", line, flags=re.I):
            break
        if re.search(r"[@{}]|\b(university|institute|school|department)\b", line, flags=re.I):
            break
        if index > 0 and looks_like_author_line(line):
            break
        title_parts.append(line)
        if len(" ".join(title_parts)) > 140:
            break
    return normalize_space(" ".join(title_parts))


def candidate_names_from_sentence(sentence: str) -> list[str]:
    names: list[str] = []

    propose_pattern = re.compile(
        r"\b(?:we\s+)?(?:propose|present|introduce|develop)\s+"
        r"(?:(?:a|an|the|one|our|novel|simple|effective|lightweight|new|deep|learning|method|framework|"
        r"model|network|module|architecture|plug-and-play|called|namely|,)\s+)*"
        r"([A-Z][A-Za-z0-9+_.-]{1,30}(?:\s+[A-Z][A-Za-z0-9+_.-]{1,30}){0,5})"
    )
    for match in propose_pattern.finditer(sentence):
        phrase = match.group(1)
        phrase = re.split(r"\s+(?:to|for|that|which|with|by|and|as|built|based)\b", phrase, maxsplit=1)[0]
        names.append(phrase)

    named_pattern = re.compile(
        r"\b(?:namely|called|termed)\s+([A-Z][A-Za-z0-9+_.-]{1,30}(?:\s+[A-Z][A-Za-z0-9+_.-]{1,30}){0,5})"
    )
    for match in named_pattern.finditer(sentence):
        names.append(match.group(1))

    module_after_pattern = re.compile(
        r"\b(?:module|block|layer|loss|attention|filter|network|mechanism)\s*,?\s+([A-Z][A-Za-z0-9+_.-]{1,30})\b"
    )
    for match in module_after_pattern.finditer(sentence):
        names.append(match.group(1))

    name_before_module_pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9+_.-]{1,30}(?:\s+[A-Z][A-Za-z0-9+_.-]{1,30}){0,4})\s+"
        r"(?:module|block|layer|loss|attention|filter|network|mechanism)\b"
    )
    for match in name_before_module_pattern.finditer(sentence):
        names.append(match.group(1))

    paren_pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9+_.-]*(?:\s+[A-Z][A-Za-z0-9+_.-]*){0,4})\s*\(([A-Z][A-Z0-9+_.-]{1,12})\)"
    )
    for match in paren_pattern.finditer(sentence):
        long_name, acronym = match.groups()
        if any(noun in long_name.lower() for noun in MODULE_NOUNS):
            names.append(acronym)
            names.append(long_name)

    noun_pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9+_.-]*(?:\s+[A-Z][A-Za-z0-9+_.-]*){0,5}\s+"
        r"(?:Module|Block|Layer|Attention|Adapter|Filter|Loss|Fusion|Selector|Selection|Mixing|Mixer|Decomposition|Network))\b"
    )
    for match in noun_pattern.finditer(sentence):
        names.append(match.group(1))

    acronym_pattern = re.compile(r"\b([A-Z][A-Z0-9+_.-]{2,15})\b")
    for match in acronym_pattern.finditer(sentence):
        token = match.group(1)
        if any(noun in sentence.lower() for noun in MODULE_NOUNS):
            names.append(token)

    cleaned: list[str] = []
    for name in names:
        value = normalize_space(name).strip(" ,.;:()[]")
        value = re.sub(r"^(a|an|the|our|novel|simple|effective|new)\s+", "", value, flags=re.I)
        value = value.strip(" ,.;:()[]")
        compact = value.replace(" ", "").upper()
        if {token.lower() for token in value.split()} & NOISE_TOKENS:
            continue
        if (
            len(value) >= 2
            and (len(value) > 2 or value.isupper())
            and not value.islower()
            and value.lower() not in {"the", "this", "our", "figure", "table"}
            and compact not in STOP_CANDIDATES
            and not re.fullmatch(r"[A-Z]\.?\d+", compact)
        ):
            cleaned.append(value)
    return cleaned


def candidate_names_from_title(title: str) -> list[str]:
    names: list[str] = []
    if ":" in title:
        before_colon, after_colon = title.split(":", 1)
        method = normalize_space(before_colon).strip("0123456789 .,-")
        if 2 <= len(method) <= 80:
            names.append(method)
        names.extend(candidate_names_from_sentence(after_colon))
    names.extend(candidate_names_from_sentence(title))
    return names


PLUG_CLAIM_PATTERNS = (
    r"plug[- ]and[- ]play",
    r"plug[- ]in\b",
    r"drop[- ]in\b",
    r"can be (?:easily |readily |seamlessly |directly )?(?:inserted|integrated|embedded|incorporated|plugged|applied|adopted)",
    r"(?:inserted|integrated|embedded|plugged) (?:into|in) (?:any|existing|other|various|different|arbitrary)",
    r"(?:replac\w+|substitut\w+) (?:the )?(?:standard|vanilla|original|conventional) ",
    r"architecture[- ]agnostic",
    r"model[- ]agnostic",
    r"generic (?:module|block|layer|component|loss|operator)",
)

OPERATION_TERMS = (
    "convolution",
    "conv",
    "softmax",
    "sigmoid",
    "attention",
    "pooling",
    "linear projection",
    "fully connected",
    "layer norm",
    "batch norm",
    "normalization",
    "concatenat",
    "element-wise",
    "hadamard",
    "matrix multiplication",
    "residual",
    "gating",
    "fourier",
    "fft",
    "reshape",
    "upsampl",
    "downsampl",
    "dilat",
    "mlp",
)

SHAPE_PATTERNS = (
    r"\bR\s*\^?\s*\{?\s*[A-Za-z0-9]+\s*[×x\*]\s*[A-Za-z0-9]+",
    r"ℝ\s*\^?\s*\{?\s*[A-Za-z0-9]+\s*[×x]",
    r"\b[BNCHWLTD]\s*[×x]\s*[BNCHWLTD]\s*(?:[×x]\s*[BNCHWLTD])?\b",
    r"\(\s*[BNCHWLTD]\s*,\s*[BNCHWLTD]\s*(?:,\s*[BNCHWLTD]\s*){0,2}\)",
    r"\b(?:feature map|tensor|input|output)s? (?:of|with) (?:size|shape|dimension)",
    r"\bof (?:size|shape) [A-Za-z0-9]+\s*[×x]\s*[A-Za-z0-9]+",
)

MATH_CHARS = set("∑∏σ⊙⊗⊕∈ℝ∥‖→←λμαβγθϕφψΩΔ∇√≤≥≈∂")

CONTRIBUTION_PATTERNS = {
    "dataset": (
        r"we (?:introduce|present|release|construct|collect|build|curate|contribute|propose) (?:a |an |the )?(?:new |novel |large[- ]scale |first |comprehensive |high[- ]quality )*(?:\w+ ){0,3}(?:dataset|corpus)",
        r"(?:new|novel|large[- ]scale) (?:\w+ ){0,2}dataset",
    ),
    "benchmark": (
        r"we (?:introduce|present|release|construct|propose|establish) (?:a |an |the )?(?:new |novel |comprehensive |unified |large[- ]scale )*(?:\w+ ){0,3}(?:benchmark|leaderboard|testbed|evaluation suite)",
        r"\bbenchmark(?:ing)? (?:for|of|on)\b",
    ),
    "survey": (
        r"\b(?:a )?(?:comprehensive |systematic |brief )?(?:survey|literature review|review of|overview of|taxonomy of)\b",
        r"this (?:survey|review|paper reviews|article reviews)",
    ),
    "theory": (
        r"we (?:prove|show theoretically|derive (?:a )?(?:generalization|regret|convergence)|establish (?:a )?(?:convergence|generalization|lower bound|upper bound))",
        r"\b(?:regret bound|convergence rate|generalization bound|minimax)\b",
    ),
    "position": (
        r"\b(?:position paper|we argue that|perspective paper|this position)\b",
    ),
    "system": (
        r"we (?:present|build|develop|deploy|design) (?:[a-z0-9-]+, )?(?:an? )?(?:end[- ]to[- ]end |full |complete |large[- ]scale |open[- ]source )*(?:system|pipeline|platform|framework for deployment|toolkit|library)",
    ),
    "training-recipe": (
        r"we (?:propose|present|introduce) (?:a |an |the )?(?:new |novel |simple |effective )*(?:\w+ ){0,2}(?:training (?:recipe|strategy|scheme|paradigm)|pre-?training (?:strategy|scheme|objective)|fine-?tuning (?:strategy|recipe)|curriculum|data (?:selection|mixture) strategy)",
    ),
    "module": (
        r"we (?:propose|present|introduce|design|develop) (?:a |an |the )?(?:new |novel |simple |effective |lightweight |efficient |generic |general |plug-and-play )*(?:\w+[- ]){0,4}(?:module|block|layer|attention|adapter|loss|unit|operator|filter|mixer|encoding|embedding|normalization|pooling|convolution|activation|regularizer|regulariser)\b",
        r"plug[- ]and[- ]play",
    ),
}


def abstract_window(text: str, limit: int = 3500) -> str:
    """Return the title/abstract/introduction area where contributions are stated."""
    lowered = text.lower()
    start = lowered.find("abstract")
    start = 0 if start < 0 or start > 4000 else start
    return text[start : start + limit]


def detect_contribution_type(title: str, text: str) -> tuple[str, dict[str, int]]:
    window = normalize_space(f"{title}. {abstract_window(text)}").lower()
    counts: dict[str, int] = {}
    for kind, patterns in CONTRIBUTION_PATTERNS.items():
        counts[kind] = sum(len(re.findall(pattern, window)) for pattern in patterns)
    non_module = {key: value for key, value in counts.items() if key != "module" and value}
    if counts.get("module", 0) and counts["module"] >= max(non_module.values(), default=0):
        return "module", counts
    if non_module:
        return max(non_module, key=lambda key: non_module[key]), counts
    return "unknown", counts


def is_equation_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 4 or len(stripped) > 160:
        return False
    numbered = re.search(r"\(\s*\d{1,2}\s*\)\s*$", stripped) is not None
    has_math = "=" in stripped or any(char in MATH_CHARS for char in stripped)
    if numbered and has_math:
        return True
    words = re.findall(r"[A-Za-z]{4,}", stripped)
    symbols = sum(1 for char in stripped if char in "=+-*/^_(){}[]|,")
    return "=" in stripped and len(words) <= 4 and symbols >= 3


def collect_structural_evidence(text: str, lowered: str, candidates: list[ModuleCandidate]) -> StructuralEvidence:
    plug_claims: list[str] = []
    for sentence in split_sentences(text):
        sentence_lower = sentence.lower()
        if any(re.search(pattern, sentence_lower) for pattern in PLUG_CLAIM_PATTERNS):
            plug_claims.append(sentence[:300])
            if len(plug_claims) >= 4:
                break

    equation_lines = sum(1 for line in text.splitlines() if is_equation_line(line))
    shape_mentions = sum(len(re.findall(pattern, text)) for pattern in SHAPE_PATTERNS)
    algorithm_mentions = len(re.findall(r"\balgorithm\s*\d+|\bpseudo-?code\b|\bpytorch[- ]style\b", lowered))
    figure_module_mentions = len(
        re.findall(
            r"(?:fig(?:ure)?\.?\s*\d+)[^.]{0,120}(?:module|block|layer|unit|attention|architecture of|structure of|illustrat\w+ (?:the )?(?:proposed|our))",
            lowered,
        )
    )

    names = [candidate.name.lower() for candidate in candidates[:6]]
    module_ablation = False
    for sentence in split_sentences(text):
        sentence_lower = sentence.lower()
        if "ablat" not in sentence_lower and "w/o" not in sentence_lower and "without" not in sentence_lower:
            continue
        if any(name in sentence_lower for name in names) or re.search(r"\b(?:module|block|component)s?\b", sentence_lower):
            module_ablation = True
            break

    operations = sorted({term for term in OPERATION_TERMS if term in lowered})

    score = 0
    if plug_claims:
        score += 3
    score += min(equation_lines, 4)
    score += min(shape_mentions, 3)
    score += 2 if algorithm_mentions else 0
    score += min(figure_module_mentions, 2)
    score += 2 if module_ablation else 0
    score += min(len(operations), 3)
    return StructuralEvidence(
        plug_claims=plug_claims,
        equation_lines=equation_lines,
        shape_mentions=shape_mentions,
        algorithm_mentions=algorithm_mentions,
        figure_module_mentions=figure_module_mentions,
        module_ablation=module_ablation,
        operation_terms=operations,
        score=score,
    )


MIN_RELIABLE_TEXT_CHARS = 1500
NON_MODULE_TYPES = {"dataset", "benchmark", "survey", "theory", "position", "system", "training-recipe"}


def _decide(
    legacy_score: int,
    positive_hits: list[str],
    candidates: list[ModuleCandidate],
    structural: StructuralEvidence,
    contribution_type: str,

) -> tuple[str, list[str]]:
    reasons: list[str] = []
    strong_claim = bool(structural.plug_claims)

    if contribution_type in NON_MODULE_TYPES and not strong_claim:
        reasons.append(f"primary contribution looks like a {contribution_type}, not a separable component")
        if structural.score >= 9 and candidates:
            reasons.append("but the text has substantial structural detail; manual review required")
            return "uncertain", reasons
        return "no", reasons

    if not candidates and not strong_claim:
        reasons.append("no named component or plug-in claim found")
        return "no", reasons

    if structural.score < 3 and not strong_claim:
        if contribution_type == "module" and candidates:
            reasons.append(
                "the paper claims a named module but the extracted text has no equations, shapes, or algorithm; "
                "check the full PDF/supplement before concluding"
            )
            return "uncertain", reasons
        reasons.append("generic vocabulary only: no equations, shapes, algorithm, or plug-in claim")
        return "no", reasons

    if candidates and legacy_score >= 8 and structural.score >= 6:
        reasons.append(f"named candidate(s) with structural detail (structural_score={structural.score})")
        if strong_claim:
            reasons.append("paper explicitly claims plug-in / insertable behavior")
        return "yes", reasons

    if positive_hits and structural.score >= 3:
        reasons.append(
            f"some evidence of a component but too little detail to isolate it (structural_score={structural.score})"
        )
        return "uncertain", reasons

    reasons.append("insufficient evidence for a reusable component")
    return "no", reasons



def decide(
    legacy_score: int,
    positive_hits: list[str],
    candidates: list[ModuleCandidate],
    structural: StructuralEvidence,
    contribution_type: str,
    text_chars: int = 10_000,
) -> tuple[str, list[str]]:
    verdict, reasons = _decide(legacy_score, positive_hits, candidates, structural, contribution_type)
    if verdict == "no" and contribution_type not in NON_MODULE_TYPES and text_chars < MIN_RELIABLE_TEXT_CHARS:
        reasons.append(
            f"only {text_chars} characters of text were available; a scanned or truncated extraction "
            "cannot support a 'no' verdict, so re-extract (OCR) or read the PDF directly"
        )
        return "uncertain", reasons
    return verdict, reasons


def analyze(text: str, max_evidence: int = 8) -> PaperAnalysis:
    text = strip_extractor_metadata(text)
    normalized = normalize_space(text)
    lowered = normalized.lower()
    title = extract_title(text)
    positive_hits = sorted(term for term in POSITIVE_TERMS if term in lowered)
    negative_hits = sorted(term for term in NEGATIVE_TERMS if term in lowered)

    sentences = split_sentences(text)
    evidence_sentences: list[str] = []
    candidate_map: dict[str, ModuleCandidate] = {}

    for sentence in sentences:
        sentence_lower = sentence.lower()
        hits = [term for term in POSITIVE_TERMS if term in sentence_lower]
        if not hits:
            continue
        if len(evidence_sentences) < max_evidence:
            evidence_sentences.append(sentence[:500])
        for name in candidate_names_from_sentence(sentence):
            candidate = candidate_map.setdefault(name, ModuleCandidate(name=name, score=0, evidence=[]))
            candidate.score += 2 + len(hits)
            if len(candidate.evidence) < 3:
                candidate.evidence.append(sentence[:350])

    if title:
        for name in candidate_names_from_title(title):
            candidate = candidate_map.setdefault(name, ModuleCandidate(name=name, score=0, evidence=[]))
            candidate.score += 8
            if title not in candidate.evidence:
                candidate.evidence.append(title)

    candidates = sorted(candidate_map.values(), key=lambda item: (-item.score, item.name.lower()))
    score = len(positive_hits) * 2 + min(len(evidence_sentences), 6) + sum(1 for item in candidates if item.score >= 3)
    score -= min(len(negative_hits), 3)

    contribution_type, _ = detect_contribution_type(title, text)
    structural = collect_structural_evidence(text, lowered, candidates)
    found_module, reasons = decide(
        score, positive_hits, candidates, structural, contribution_type, text_chars=len(normalized)
    )

    return PaperAnalysis(
        found_module=found_module,
        title=title,
        score=score,
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        candidates=candidates[:12],
        evidence=evidence_sentences,
        contribution_type=contribution_type,
        structural=structural,
        reasons=reasons,
    )


def render_text(result: PaperAnalysis) -> str:
    lines = [
        f"found_module: {result.found_module}",
        f"title: {result.title}",
        f"score: {result.score}",
        f"contribution_type: {result.contribution_type}",
        f"structural_score: {result.structural.score if result.structural else 'n/a'}",
        f"positive_hits: {', '.join(result.positive_hits) if result.positive_hits else 'none'}",
        f"negative_hits: {', '.join(result.negative_hits) if result.negative_hits else 'none'}",
        "candidates:",
    ]
    if result.candidates:
        for candidate in result.candidates:
            lines.append(f"  - {candidate.name} (score={candidate.score})")
    else:
        lines.append("  none")
    lines.append("reasons:")
    for reason in result.reasons or ["none"]:
        lines.append(f"  - {reason}")
    lines.append("evidence:")
    if result.evidence:
        for sentence in result.evidence:
            lines.append(f"  - {sentence}")
    else:
        lines.append("  none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=Path, help="Extracted paper text file.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    if not args.text.exists():
        parser.error(f"text file not found: {args.text}")

    text = args.text.read_text(encoding="utf-8", errors="replace")
    result = analyze(text)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(render_text(result))
    return 0 if result.found_module != "no" else 1


if __name__ == "__main__":
    raise SystemExit(main())
