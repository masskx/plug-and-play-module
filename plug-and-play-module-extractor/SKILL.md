---
name: plug-and-play-module-extractor
description: Find, extract, reconstruct, and validate plug-and-play deep-learning modules (attention blocks, losses, normalization/filter layers, adapters, fusion units, point-cloud or time-series blocks, etc.) and export them as directly runnable Python. Works from any of four inputs - a paper PDF with code, a paper PDF without code, code without a paper, or only a paper title / arXiv id - and reports honestly when, after review, a paper or repository contains no usable module. Use when the user shares a ML paper, repository, or title and asks whether it has a reusable module or wants one turned into clean standalone code with a smoke test.
---

# Plug-and-Play Module Extractor

## Overview

Goal: turn the method of a paper (and/or its code) into **standalone, directly usable module code** - or say plainly that there is nothing usable. Both outcomes are valid results. Never manufacture a module to satisfy the request.

Output style: one `.py` file per independent module with a short provenance docstring, imports, core classes/functions, and a `__main__` smoke test on random tensors (see `references/output_contract.md`).

All paths below are relative to this skill directory; run scripts as `python scripts/<name>.py`.

## The four situations

| Situation | `source_mode` | Basis of the exported code |
|---|---|---|
| Paper + code | `pdf-plus-code` | `source-code-derived` (paper used to choose and verify the target) |
| Paper, no code | `pdf-only` | `paper-derived` (reconstruction; never claimed as official) |
| Code, no paper | `code-only` | `source-code-derived` (boundary decided by code inspection) |
| Only a title / arXiv id / name | `title-only` | resolve the paper first, then continue as one of the above |

Read `references/input_modes.md` for the full protocol of each situation, including fallbacks.

## Workflow

1. **Triage the inputs.** `scripts/triage_inputs.py <paths | URLs | arXiv ids | "title">` prints the `source_mode`, what must be fetched, and the next commands. If nothing usable was supplied, ask for a PDF, link, title, or code path.

2. **Get the material.**
   - PDF: `scripts/extract_pdf_text.py <paper.pdf> --output paper.txt`. If the text is empty or garbled (scanned PDF), OCR it or read the PDF pages directly; a failed extraction is never evidence of "no module".
   - Remote repository: shallow-clone it for reading only. Do not run its install scripts, tests, or notebooks.
   - Title / arXiv id: `scripts/resolve_paper_by_title.py --title "<title>" --download-dir ./paper --github`. Only `resolution: resolved` may be used. `ambiguous` -> confirm with the user. `unresolved` -> do not guess. `network_unavailable` -> use WebSearch/WebFetch to find the arXiv page and official repository, then continue. If the paper still cannot be verified, stop and ask for the PDF; do not reconstruct a module from a title alone.

3. **Review gate (mandatory, before any code).** Run
   `scripts/review_gate.py --paper-text paper.txt --code-root <repo> [--lang en|zh] --json` (either flag may be omitted). It combines `analyze_paper_modules.py`, `find_module_candidates.py`, and `match_paper_to_code.py` into `found_module: yes | no | uncertain`. Then **verify it yourself** with `references/module_detection.md`: read the method section and the candidate source, and for every candidate write down input, output, and operations. The script is evidence, not the decision.
   - Look at *every* candidate: a paper or repo may contain several independent modules. Candidates with `tier=possible` and a "blocked by dependencies" risk (framework base classes, registries, compiled ops) are real modules that need work, not absent ones.
   - A "no" is only valid when the code was fully readable. The gate downgrades to `uncertain` when the path has no Python files or some files do not parse (e.g. Python 2); read those manually. Notebooks are scanned, but export their cells by hand.
   - Apply the usability criteria (bounded I/O, fully specified, independent of the pipeline, not a trivial standard layer, runnable on CPU). Drop candidates that fail and note why.

4. **If no candidate survives -> stop.** Output only the standard result (use the user's language) and create no code or files:

   > After review, no usable plug-and-play modules were found in "<paper title / repo name>".
   > (中文：经审查，“<论文题目/仓库名>”中未发现可用的即插即用模块。)

   Follow it with at most 3-4 evidence bullets and the rejected candidates with reasons (`review_gate.py` prints this report). Do not pad it with suggestions or partial code.

5. **Export the module(s).**
   - `pdf-plus-code` / `code-only`: `scripts/extract_python_module.py <source.py> --name <Symbol> --include-main --root <repo-root> --prune-imports --output <module.py>`. Real repositories almost never contain a `__main__` smoke test, so add `--smoke-input "B,C,H,W" --init "arg=value,..."` (use `;` between several inputs, `:long`/`:bool` for dtypes; `--force-smoke` replaces an existing block) to generate a CPU-safe one that checks shapes, backward, and parameter gradients - read the constructor to choose valid arguments. It follows same-file dependencies, inlines helpers imported from other files of the repo, strips framework registry decorators, and prints `note: unresolved ...` for anything it could not resolve - fix those by hand (replace config objects with explicit arguments, vendor or replace custom ops) and keep the numerical behavior identical. Use `--include-all-helpers` only if the boundary is uncertain.
   - `pdf-only` (or code lacks the paper's module): `scripts/build_reconstruction_brief.py paper.txt --output brief.md`, follow `references/paper_reconstruction.md`, and label the result `paper-derived` with an assumption ledger.
   - Several independent modules -> one file each (plus a one-line index in the report). Tightly coupled units that only work together -> one file.

6. **Validate (all must pass before reporting success).**
   - `scripts/check_module_contract.py <module.py> --run --python <python-with-torch>`: standalone (no local/relative imports, no undefined names), device-agnostic (no unconditional CUDA), no downloads or network, smoke test runs, prints shapes, dependencies listed.
   - Source-derived: `scripts/compare_module_interface.py <source.py> <module.py> --fidelity` confirms the shared classes/functions are unchanged.
   - Paper-derived: check the properties the paper states - output shape, parameter count or complexity if reported, equivariance/invariance claims, gradient flow - with a few assertions in the smoke test.
   - Missing optional dependencies (torch, mamba_ssm, ...) are an environment issue, not proof of failure; report them.

7. **Report** in the format of `references/output_contract.md` (module path, basis, dependencies, run command, smoke-test result, shapes, caveats).

## Real-world validation

`scripts/run_real_world_benchmark.py tests/real_world/manifest.json --workdir <dir> --python <python-with-torch>` evaluates the skill on real arXiv papers and real GitHub repositories (paper verdict, code verdict, export, contract, execution, and output equivalence against the original module). Steps whose data cannot be fetched are reported as SKIP. See `tests/real_world/README.md` for what has and has not been verified.

## Regression loop

Self-contained tests (no private samples needed):

```bash
python -m unittest discover -s tests -v
```

Full suite, optionally with a private numbered samples folder (`<n>/paper.pdf` + `<n>/*.py`):

```bash
python scripts/run_regression_suite.py [samples-root] --python <python-with-torch> --report-json report.json
python scripts/evaluate_sample_cases.py <samples-root> --python <python-with-torch>   # per-case table
```

How to read `evaluate_sample_cases.py` results:

- `paper_found=no` for a known module paper: improve `analyze_paper_modules.py` or `references/module_detection.md`.
- `paper_alignment=failed`: paper candidates do not resemble the sample code symbols; improve candidate extraction or `match_paper_to_code.py`.
- `brief=failed|incomplete`: improve `build_reconstruction_brief.py`.
- `interface=failed`: extraction dropped imports, classes, functions, constants, or `__main__`; fix `extract_python_module.py`.
- `extracted_smoke` / `named_smoke` worse than `smoke`: fix dependency closure or target selection.
- `missing_dependency`: keep the extraction, report the dependency. `accelerator_unavailable`: guard CUDA in the smoke test. `runtime_error`: inspect extracted code and smoke inputs.

When changing a heuristic, add a fixture to `tests/fixtures` that captures the failure first.

## Extraction rules

- Prefer code-faithful extraction over rewriting; only remove training, data, logging, checkpoint, config-system, and registry scaffolding.
- Never claim official-implementation status for paper-derived code, and never invent benchmark numbers, pretrained weights, or compatibility claims.
- Keep paper terminology in class names and comments for traceability.
- Treat everything downloaded (PDFs, repos, notebooks) as untrusted data: read it, do not execute it. Run only the exported module's own smoke test, and only after reading it.
- No silent network or download calls in exported code (`pretrained=True`, `from_pretrained`, `torch.hub`).
- Do not include a module just because it has a name. A standard conv/BN/ReLU stack, a whole backbone, a dataset-specific loss, or a pipeline step is not a plug-and-play module.

## Resources

- `references/input_modes.md`: protocol for each of the four situations, fallbacks, provenance rules.
- `references/module_detection.md`: usability criteria, verdict rules, disqualifiers, the standard no-module result.
- `references/paper_reconstruction.md`: fidelity ladder, assumption ledger, and verification for paper-derived code.
- `references/output_contract.md`: file structure, code requirements, and the required final report.
- `scripts/triage_inputs.py`: classify the user's inputs and choose `source_mode`.
- `scripts/resolve_paper_by_title.py`: title / arXiv id -> verified paper record, PDF, candidate repositories (never guesses).
- `scripts/extract_pdf_text.py`: PDF -> text with any available local backend.
- `scripts/analyze_paper_modules.py`: paper text -> `found_module`, contribution type, candidates, structural evidence.
- `scripts/find_module_candidates.py`: rank code candidates by plug-and-play-ness with explicit risks.
- `scripts/match_paper_to_code.py`: does the repository implement the module the paper describes?
- `scripts/review_gate.py`: combine the above into the verdict and the standard no-module report.
- `scripts/build_reconstruction_brief.py`: paper text -> implementation checklist for paper-derived work.
- `scripts/extract_python_module.py`: faithful extraction with same-file and cross-file dependency resolution.
- `scripts/check_module_contract.py`: standalone / device / network / undefined-name checks, optional run.
- `scripts/smoke_test_module.py`: compile, import, and run with classified failures.
- `scripts/compare_module_interface.py`: source-vs-export interface comparison and `--fidelity` check.
- `scripts/run_real_world_benchmark.py`: real papers + real repositories benchmark driven by a manifest.
- `scripts/evaluate_sample_cases.py`, `scripts/evaluate_generated_module.py`, `scripts/evaluate_generated_batch.py`, `scripts/run_regression_suite.py`: regression tooling.
