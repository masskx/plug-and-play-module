# Output Contract

Use this contract when producing an extracted plug-and-play module.

## Default artifact

One standalone Python file per independent module:

```text
<module_name>.py
```

Use a package directory only when the implementation cannot reasonably run as a single file. When several independent modules are exported, list them together in the report (one line each); do not merge unrelated modules into one file.

## File structure

In order:

1. Module docstring (first), covering:
   - paper/module name (and the paper reference when known);
   - `Implementation basis:` source-code-derived | paper-derived | mixed;
   - for source-derived code: source file path and selected symbols; for paper-derived code: equations/figures used and assumptions;
   - the intended plug-in use case;
   - expected input and output shapes.
2. Imports (only what is needed).
3. Core classes/functions and minimal helpers. Project-local helpers are inlined, not imported.
4. Optional lightweight mock config class only if the original constructor needs a config object; prefer explicit arguments.
5. `if __name__ == "__main__":` smoke test with synthetic tensors.

## Code requirements

- Standalone: no relative imports, no imports from the original repository, no undefined names. `scripts/check_module_contract.py` enforces this.
- Faithful: keep source-derived logic unchanged except for removing scaffolding; `compare_module_interface.py --fidelity` should report no modified symbols. Document every deliberate change in the docstring.
- Remove training loops, dataset loading, checkpoint paths, loggers, distributed launch code, experiment configuration systems, and framework registry decorators.
- Preserve meaningful class names, argument names, tensor-layout comments, and paper terminology.
- Comments only for non-obvious tensor shapes or extraction decisions.
- No silent downloads or network calls (`pretrained=True`, `from_pretrained`, `torch.hub`, `requests`).
- CPU-compatible. Never leave unconditional `.cuda()`, `.to("cuda")`, or `device="cuda"`. Use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and move model and tensors through it.
- Configurable sizes: channels/dimensions as constructor arguments, no hard-coded input resolution unless the operation requires it.
- Smoke test: prints input and output shapes; asserts the documented output shape; for modules with parameters runs `backward()` on a scalar of the output so gradient flow is exercised; seeds randomness for reproducibility.

## Dependency notes

At the top of the report, list:

- standard library dependencies;
- pip/conda dependencies (`check_module_contract.py` prints them);
- dependencies that are optional or missing in the current environment;
- any reason a dependency was not vendored.

A missing dependency during validation is an environment issue unless the exported code imports the wrong package.

## Required final report

After an export:

```text
found_module: yes|uncertain
source_mode: pdf-plus-code|pdf-only|code-only|title-only
paper: <title / arXiv id / none>   (resolved-from-title: yes|no)
modules:
  - name: <ClassOrFunction>
    module_path: <path>
    implementation_basis: paper-derived|source-code-derived|mixed
    source: <file:symbol, or paper sections/equations>
    input_output_shapes: <summary>
rejected_candidates: <name: reason, ...>   (or none)
dependencies: <summary>
run_command: python <module.py>
smoke_test: passed|failed|not-run
contract_check: passed|failed|not-run
fidelity_check: passed|failed|not-run     (source-derived only)
notes: <assumptions, deviations from the source/paper, caveats>
```

For `uncertain`, add `missing: <what would settle it>` and make clear in `notes` that the export (if any) is conservative.

## "No module" report

For `found_module: no` produce only the standard result from `references/module_detection.md`:

```text
After review, no usable plug-and-play modules were found in "<title or repository>".
Basis: ...
Candidates considered and rejected: ...
```

`module_path: none`, `implementation_basis: none`; create no code and no files.

## Title-only that could not be resolved

Report `found_module: uncertain`, `reason: paper not retrievable/verified`, the resolver status, and what the user can provide (PDF, arXiv link, repository). Do not export code and do not say "no usable modules".
