# Output Contract

Use this contract when producing an extracted plug-and-play module.

## Default Artifact

Produce one standalone Python file by default:

```text
<module_name>.py
```

Use a package directory only when the extracted implementation cannot reasonably run as a single file.

## File Structure

The Python file should contain, in order:

1. Imports.
2. A short module docstring or block comment covering:
   - paper/module name;
   - whether the implementation is source-code-derived or paper-derived;
   - the intended plug-in use case;
   - expected input and output shapes.
3. Core classes/functions and minimal helper code.
4. Optional lightweight mock config class when the original code requires config objects.
5. `if __name__ == "__main__":` smoke test with synthetic tensors or minimal synthetic inputs.

## Code Requirements

- Keep source-code-derived logic faithful to the provided repository.
- Remove unrelated training loops, dataset loading, checkpoint paths, loggers, distributed launch code, and experiment configuration systems.
- Preserve meaningful class names, argument names, tensor layout comments, and paper terminology.
- For source-code-derived extraction, record the source file path and selected top-level symbols in the module docstring or final report.
- When using `scripts/extract_python_module.py`, prefer named extraction for the paper-matched class/function. The script auto-includes same-file top-level dependencies by default; use `--include-all-helpers` only when the module boundary is uncertain or multiple cooperating modules should stay together.
- Add comments only for non-obvious tensor shape or extraction decisions.
- Avoid silent downloads or network calls in the smoke test.
- Prefer CPU-compatible smoke tests. Move to CUDA only if the original operation requires it and guard with availability checks.
- Never leave an unconditional `.cuda()` or `.to("cuda")` in the smoke test. Use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and move both model and tensors through that device when acceleration is optional.
- Print input and output shapes in the smoke test.

## Dependency Notes

At the top-level report, list:

- standard library dependencies;
- pip/conda dependencies;
- dependencies that are optional or missing in the current environment;
- any reason a dependency was not vendored into the output.

If a dependency is missing during validation, report it as an environment issue unless the generated code imports the wrong package.

## Required Final Report

The final response after extraction must include:

```text
found_module: yes|no|uncertain
source_mode: pdf-only|pdf-plus-code
module_path: <path or none>
implementation_basis: paper-derived|source-code-derived|mixed|none
dependencies: <summary>
run_command: python <module.py>
smoke_test: passed|failed|not-run
interface_compare: passed|failed|not-run
input_output_shapes: <summary>
notes: <short caveats>
```

For `found_module: no`, set `module_path: none`, `implementation_basis: none`, and do not create code.
