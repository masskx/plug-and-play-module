#!/usr/bin/env python3
"""Evaluate the skill on real papers and real repositories listed in a manifest.

For every manifest entry it runs, as far as the network allows:

1. paper   - fetch the arXiv PDF (or use ``--papers-dir/<arxiv_id>.pdf``), extract text, run the paper
             analysis, compare with ``expected_paper`` (a list of acceptable verdicts);
2. code    - download the listed files from raw.githubusercontent.com (cached in ``--workdir``),
             run the code-only review gate, check the verdict and that each target symbol is surfaced;
3. export  - extract each target with a generated smoke test, run the contract check, execute it
             (``expect_smoke`` lists acceptable outcomes, default ``passed``);
4. fidelity- for targets with ``"equiv": true``: import the original module and the export, copy the
             weights, and compare outputs (needs PyTorch).

A step whose data could not be fetched is reported as SKIP, never as a failure or a pass.
All downloaded material is only read or imported inside this process for the equivalence check - run
the benchmark in a sandbox you are comfortable executing the listed repositories' module files in.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import resolve_paper_by_title as resolver  # noqa: E402
from analyze_paper_modules import analyze  # noqa: E402
from check_module_contract import check as contract_check  # noqa: E402
from check_module_contract import run_module  # noqa: E402
from extract_pdf_text import extract_pages, render_pages  # noqa: E402
from extract_python_module import extract  # noqa: E402
from review_gate import run_gate  # noqa: E402


@dataclass
class StepResult:
    status: str  # PASS | FAIL | SKIP
    detail: str = ""


@dataclass
class EntryResult:
    id: str
    steps: dict[str, StepResult] = field(default_factory=dict)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.steps[name] = StepResult(status, detail)


def http_download(url: str, destination: Path, timeout: int = 60) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": resolver.USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
            destination.write_bytes(response.read())
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def paper_text_for(entry: dict, workdir: Path, papers_dir: Path | None) -> tuple[str | None, str]:
    arxiv_id = entry.get("arxiv_id")
    if not arxiv_id:
        return None, "no arxiv_id"
    pdf = None
    if papers_dir and (papers_dir / f"{arxiv_id}.pdf").exists():
        pdf = papers_dir / f"{arxiv_id}.pdf"
    else:
        cached = workdir / "papers" / f"{arxiv_id}.pdf"
        if not cached.exists():
            resolution = resolver.resolve_arxiv_id(arxiv_id)
            if resolution.status != "resolved" or not resolution.paper:
                return None, f"arXiv lookup {resolution.status}"
            if not http_download(resolution.paper.pdf_url, cached):
                return None, "PDF download failed"
        pdf = cached
    try:
        backend, pages = extract_pages(pdf)
    except RuntimeError as exc:
        return None, f"PDF text extraction failed: {exc}"
    return render_pages(pdf, backend, pages), f"{len(pages)} pages via {backend}"


def fetch_code(entry: dict, workdir: Path) -> tuple[Path | None, str]:
    code = entry.get("code")
    if not code:
        return None, "no code listed"
    root = workdir / "code" / entry["id"]
    missing = []
    for relative in code["files"]:
        target = root / relative
        if target.exists():
            continue
        url = f"https://raw.githubusercontent.com/{code['repo']}/{code['branch']}/{relative}"
        if not http_download(url, target):
            missing.append(relative)
    if missing:
        return None, f"could not download: {', '.join(missing)}"
    return root, f"{len(code['files'])} files"


def load_module(path: Path, name: str, extra_path: Path | None = None):
    if extra_path:
        sys.path.insert(0, str(extra_path))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    finally:
        if extra_path:
            sys.path.remove(str(extra_path))


def equivalence(original: Path, exported: Path, symbol: str, init: str, shape: str, root: Path) -> StepResult:
    try:
        import torch
    except ImportError:
        return StepResult("SKIP", "PyTorch not installed")
    from extract_python_module import parse_init_kwargs, parse_input_specs

    kwargs = parse_init_kwargs(init)
    specs = parse_input_specs(shape)
    try:
        orig = load_module(original, f"orig_{symbol}", extra_path=root)
        exp = load_module(exported, f"exp_{symbol}")
        torch.manual_seed(0)
        model_a = getattr(orig, symbol)(**kwargs).eval()
        torch.manual_seed(1)
        model_b = getattr(exp, symbol)(**kwargs).eval()
        model_b.load_state_dict(model_a.state_dict())
        inputs = [torch.randn(*dims) if dtype == "float" else torch.randint(0, 8, dims) for dims, dtype in specs]
        with torch.no_grad():
            out_a, out_b = model_a(*inputs), model_b(*inputs)
        out_a = out_a[0] if isinstance(out_a, (tuple, list)) else out_a
        out_b = out_b[0] if isinstance(out_b, (tuple, list)) else out_b
        diff = (out_a - out_b).abs().max().item()
        return StepResult("PASS" if diff <= 1e-6 else "FAIL", f"max|diff|={diff:.2e}")
    except Exception as exc:  # the original may need packages that are not installed
        return StepResult("SKIP", f"could not import original/export: {type(exc).__name__}: {str(exc)[:80]}")


def evaluate_entry(entry: dict, workdir: Path, papers_dir: Path | None, python: str, do_paper: bool) -> EntryResult:
    result = EntryResult(entry["id"])

    # 1. paper
    expected = entry.get("expected_paper")
    text = None
    if do_paper and expected:
        text, detail = paper_text_for(entry, workdir, papers_dir)
        if text is None:
            result.add("paper", "SKIP", detail)
        else:
            analysis = analyze(text)
            ok = analysis.found_module in expected
            result.add(
                "paper",
                "PASS" if ok else "FAIL",
                f"verdict={analysis.found_module} (acceptable: {'/'.join(expected)}) type={analysis.contribution_type} "
                f"structural={analysis.structural.score if analysis.structural else '-'} [{detail}]",
            )
    elif expected:
        result.add("paper", "SKIP", "--no-paper")

    # 2. code
    root, detail = fetch_code(entry, workdir)
    if root is None:
        if entry.get("code"):
            result.add("code", "SKIP", detail)
        return result
    gate = run_gate(None, root, None)
    surfaced = {t["name"]: t for t in gate.targets}
    expected_code = entry.get("expected_code")
    if expected_code == "none":
        ok = gate.found_module == "no"
        result.add("code", "PASS" if ok else "FAIL", f"verdict={gate.found_module} (expected no usable module)")
    else:
        names = [t["symbol"] for t in entry.get("targets", [])]
        missing = [n for n in names if n not in surfaced]
        ok = gate.found_module != "no" and not missing
        result.add(
            "code",
            "PASS" if ok else "FAIL",
            f"verdict={gate.found_module}; targets surfaced: {len(names) - len(missing)}/{len(names)}"
            + (f"; missing: {', '.join(missing)}" if missing else ""),
        )

    # 3. export and 4. fidelity
    for target in entry.get("targets", []):
        label = target["symbol"]
        out_dir = workdir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{entry['id']}__{label}.py"
        try:
            exported, notes = extract(
                root / target["file"],
                [label],
                root=root,
                prune_imports=True,
                force_smoke=True,
                smoke_input=target.get("smoke_input"),
                init=target.get("init"),
            )
        except Exception as exc:
            result.add(f"export:{label}", "FAIL", f"{type(exc).__name__}: {exc}")
            continue
        out_path.write_text(exported, encoding="utf-8")
        report = contract_check(out_path)
        errors = [f"{f.code}:{f.message[:50]}" for f in report.findings if f.severity == "error"]
        run = run_module(out_path, python, 180)
        outcome = run.get("classification", "failed")
        acceptable = target.get("expect_smoke", ["passed"])
        ok = report.status == "passed" and outcome in acceptable
        result.add(
            f"export:{label}",
            "PASS" if ok else "FAIL",
            f"contract={report.status} smoke={outcome}" + (f" errors={errors}" if errors else "") + (f" notes={len(notes)}" if notes else ""),
        )
        if target.get("equiv"):
            result.steps[f"fidelity:{label}"] = equivalence(
                root / target["file"], out_path, label, target.get("init", ""), target["smoke_input"], root
            )
    return result


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="JSON manifest (see tests/real_world/manifest.json).")
    parser.add_argument("--workdir", type=Path, default=Path("real_world_workdir"), help="Cache/output directory.")
    parser.add_argument("--papers-dir", type=Path, help="Directory of <arxiv_id>.pdf files to use instead of downloading.")
    parser.add_argument("--python", default=sys.executable, help="Interpreter used to run exported modules.")
    parser.add_argument("--only", action="append", help="Run only these entry ids.")
    parser.add_argument("--no-paper", action="store_true", help="Skip the paper step (code-side only).")
    parser.add_argument("--json", type=Path, help="Write the full results as JSON.")
    args = parser.parse_args(argv)

    entries = json.loads(args.manifest.read_text(encoding="utf-8"))["entries"]
    if args.only:
        entries = [entry for entry in entries if entry["id"] in args.only]
    args.workdir.mkdir(parents=True, exist_ok=True)

    results = [evaluate_entry(e, args.workdir.resolve(), args.papers_dir, args.python, not args.no_paper) for e in entries]

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for result in results:
        print(f"\n{result.id}")
        for name, step in result.steps.items():
            counts[step.status] += 1
            print(f"  [{step.status}] {name:<28} {step.detail}")
    print(f"\nsummary: PASS={counts['PASS']} FAIL={counts['FAIL']} SKIP={counts['SKIP']}  (SKIP = data could not be fetched/imported)")
    if args.json:
        args.json.write_text(
            json.dumps([{"id": r.id, "steps": {k: asdict(v) for k, v in r.steps.items()}} for r in results], indent=2),
            encoding="utf-8",
        )
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
