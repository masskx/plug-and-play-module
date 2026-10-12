# Real-world validation

`manifest.json` lists real top-venue papers and their public repositories. Run it with:

```bash
python plug-and-play-module-extractor/scripts/run_real_world_benchmark.py tests/real_world/manifest.json \
  --workdir /tmp/pnp_bench --python python          # python needs torch (+ timm for Swin/ConvNeXt)
# papers behind a blocked network: put <arxiv_id>.pdf files in a folder and add --papers-dir <folder>
```

Each entry is checked in up to four steps - **paper** verdict, **code** verdict, **export** (extract + generated smoke test + contract check + execution) and **fidelity** (original vs exported output with identical weights). A step whose data could not be fetched is `SKIP`, never `PASS`.

## Status of the evidence (last run)

| Part | Verified on real data? | Result |
|---|---|---|
| Code side: 8 modules from CVPR / ECCV / ICCV repos (SE, CBAM, ECA, Coordinate Attention, RepVGG, Swin `WindowAttention`, ConvNeXt `Block`, DINO `Attention`) | **Yes**, fetched from GitHub | surfaced by the code-only gate; exported standalone; contract check and execution pass; output identical to the original (max diff 0.0) |
| Code side: compiled-op / framework modules (Mamba, mmdet `FPN`) | **Yes** | surfaced as `uncertain` with the exact blockers; exports report `missing_dependency` (honest, not a fake pass) |
| Code side: negative control (PyTorch MNIST example) | **Yes** | `no usable plug-and-play modules` |
| Paper side: verdicts for the 6 positive and 6 negative papers in the manifest | **No** - arXiv/CVF/OpenReview were unreachable from the development sandbox | only synthetic fixtures were tested; run the benchmark where arXiv is reachable and treat `expected_paper` as the maintainer's judgement, not ground truth |
| Title lookup against live arXiv | **No** (same reason) | parsing/matching covered by an offline feed fixture |

## Bugs this benchmark found (all fixed, with regression tests)

- A registry-decorated framework module with a plain name (mmdet `FPN`) was invisible to the scanner, which produced a false "no usable module".
- Optional compiled-op imports inside `try/except` were not seen, so Mamba's dependencies were missed.
- The main class of a file (CBAM) was mislabelled as a tiny helper; factory functions for whole models and trivial helpers cluttered the "strong" list.
- An empty path or a Python 2 file produced "no usable module" instead of "uncertain".
- No real repository ships a `__main__` smoke test, so every export failed the contract until the extractor learned to generate one (`--smoke-input`, `--init`).
