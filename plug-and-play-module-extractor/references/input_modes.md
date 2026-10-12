# Input Modes

Pick the mode with `scripts/triage_inputs.py`. Every mode ends at the same review gate, the same export rules, and the same validation. What differs is where the module boundary comes from and how the result is labelled.

## A. Paper + code (`pdf-plus-code`)

Basis: `source-code-derived`.

1. Extract the paper text and run `review_gate.py --paper-text ... --code-root ...`.
2. Check `paper_code_alignment`:
   - `matched`: a code symbol or file clearly corresponds to a paper module (name, abbreviation, or truncated words). Extract that symbol.
   - `weak` / `none`: do not trust the code yet. Grep the repo for the paper's module name, its equations' operations (e.g. the specific pooling + gating), and the README's paper reference. Possible outcomes:
     - the module is implemented under another name -> extract it and record the mapping in the docstring;
     - the repo is a different/earlier version or a baseline -> say so, and fall back to a `paper-derived` reconstruction (`implementation_basis: mixed` if any code is reused);
     - the code contains only the full model/training pipeline -> extract the separable unit if one exists, else treat as pdf-only.
3. Compare behavior with the paper: tensor layout, hidden sizes/reduction ratios, activation choices, residual connections. If code and paper disagree, keep the code's behavior (it produced the paper's numbers) and list the difference in the notes.
4. Several modules in the paper (e.g. a spatial block and a channel block plus a loss): extract each independent one separately; keep only truly coupled units together.
5. Official code may rely on custom CUDA ops, registries, or config objects. Replace configs with explicit arguments and registries with plain classes. For custom CUDA/C++ ops, check whether a pure-PyTorch fallback exists in the repo; if the module is unusable without the compiled op and no fallback exists, report it as `uncertain` with the exact dependency rather than silently reimplementing it.

## B. Paper, no code (`pdf-only`)

Basis: `paper-derived`. Never say "official" or "reproduced from the authors' code".

1. Extract the paper text; check the page count and whether equations survived extraction. Read the method section and appendix/supplement directly if the extraction is lossy.
2. Run the review gate. Decide `found_module` using `module_detection.md`.
3. `no` -> output the standard no-module result and stop.
4. `yes` / `uncertain` -> follow `paper_reconstruction.md` (fidelity ladder, assumption ledger, property checks).
5. Before reconstructing, search for code that already exists (the title resolver's `--github`, or WebSearch). If official or well-regarded code turns up, switch to mode A with that repository (and mention the switch). Do not trust an unofficial repo blindly: verify it against the paper's equations.

## C. Code, no paper (`code-only`)

Basis: `source-code-derived`; `paper: none` in the report.

1. Run `find_module_candidates.py <repo>` (or the review gate). Candidates are ranked by tier (`strong` / `possible` / `weak`) with explicit risks.
2. Read the top candidates. A good plug-and-play unit is: an `nn.Module`/function with a small `forward` (tensor in, tensor out or loss), explicit constructor arguments, and no dependence on dataset/config/registry. Docstrings, README, and comments often name the source paper; record it if present, but do not claim details the code does not show.
3. Decide per candidate whether it is **meaningful on its own** (novel or at least non-trivial operation, reusable across models). Plain `ConvBNReLU`, `Identity`, or one-line wrappers are not modules worth exporting. Whole models (`ResNet`, `Detector`, anything taking `num_classes`) are not plug-and-play units; export a block from inside them only if it is a separable class.
4. Nothing qualifies (only trainers, datasets, scripts, whole models, or trivial layers) -> standard no-module result.
5. Export with `extract_python_module.py --root <repo>`; the script inlines cross-file helpers. Fix whatever it reports as unresolved. Real files rarely have a smoke test - generate one with `--smoke-input`/`--init`.
   Typical real-world blockers, all seen on real repositories: framework base classes and registries (mmcv/mmengine `BaseModule`, `@MODELS.register_module()`), config objects, helpers in sibling files (inlined automatically), and compiled kernels (Mamba's `selective_scan_cuda`, `causal_conv1d`). The first three are removable; for the last, look for a pure-PyTorch reference path in the repo, otherwise report `uncertain` with the dependency named.
6. Notebooks (`.ipynb`) are scanned (code cells, magics dropped) but `extract_python_module.py` only exports `.py` files: copy the cells into a `.py` first. Non-Python code (C++/CUDA) is not scanned: read it manually. TensorFlow/JAX files are Python and are scanned. Port to a standalone PyTorch module only when the user asked for PyTorch or the original framework is unavailable, and label it `mixed`.

## D. Title only (`title-only`)

A title is not evidence. The aim is to turn it into mode A, B, or C with verified material.

1. `resolve_paper_by_title.py --title "..." --download-dir ./paper --github`
   - `resolved`: confirm title/authors/year look right, extract the downloaded PDF, continue.
   - `ambiguous`: show the alternatives to the user and ask which one (or pick only if they gave authors/year that disambiguate).
   - `unresolved`: tell the user the paper could not be verified; ask for the PDF or a link.
   - `network_unavailable`: use WebSearch/WebFetch to find the arXiv/OpenReview/publisher page and the official repository. Save what you fetch to files so the normal scripts can read it.
2. Repository candidates from search are *unverified* until the README or paper links to them. Prefer the repo linked from the paper/abstract. Skim the README for a paper reference and the code for a class matching the paper's module name.
3. Continue as A (paper + verified code) or B (paper only). If the paper itself is paywalled but the official code is available, continue as C and say the paper was not reviewed.
4. If nothing can be retrieved, do not reconstruct the module from memory of the title. Report `found_module: uncertain` with `reason: paper not retrievable`, and ask for the PDF. A "no usable modules" result is never given for a paper that was not actually reviewed.

## Provenance labels

| Situation | `implementation_basis` | Also state |
|---|---|---|
| Code extracted unchanged (plus inlining) | `source-code-derived` | source file and symbols |
| Code adapted (configs removed, ops replaced) | `source-code-derived` | what was changed |
| Paper reconstruction | `paper-derived` | equations/figures used, assumptions |
| Paper reconstruction that reuses some repo code | `mixed` | which parts came from where |
| Nothing usable | `none` | the standard no-module result |

## Combining modes

- Multiple papers/repos in one request: run the pipeline once per paper/repo and report each separately.
- A paper whose method is a full architecture but whose code contains a separable block (e.g. a novel attention layer inside the model): the paper review may say `no`, the code scan `strong`. `review_gate.py` returns `uncertain` for this; decide by reading the block - export it only if it is non-trivial and its role in the paper is clear.
