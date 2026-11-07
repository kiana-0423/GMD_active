# PHAL manuscript source

英文预印本初稿，聚焦 **GMD 生态中的 PHAL 平台与已完成的烷烃 pilot 验证**。作者、单位、邮箱及作者声明保留占位。正文未加入正在运行的 `alkane_repair` 实验结果；没有编译 LaTeX，也没有启动新的模拟或训练。

**Title:** PHAL: Physics-Aware Hotspot Active Learning for Local Reference Labeling of Machine-Learned Force Fields

## Start here

- [main.tex](main.tex): main manuscript, abstract, conclusions, and declarations.
- [metadata.tex](metadata.tex): author, affiliation, contact, and funding placeholders.
- [sections/](sections/): introduction, methods, setup, completed results, discussion, and appendix.
- [figures/](figures/): six original scientific figures, each in PDF, SVG, and PNG.
- [references.tex](references.tex): explicit numbered bibliography used by the manuscript.
- [references.bib](references.bib): editable BibTeX metadata for the same 11 references.
- [audit/claims.md](audit/claims.md): claim-to-evidence and figure-to-data map.
- [audit/review_notes.md](audit/review_notes.md): source review, interpretation limits, and checks performed.

This is an author-review draft, not a submitted or accepted arXiv manuscript. No pending-results section or invented repair outcome has been inserted. The current scientific scope is local reference fidelity, measured pilot costs, exploratory acquisition comparisons, and a real closed-loop integration example. The draft explicitly reports the absence of demonstrated acquisition superiority and the limitations of the initially inaccurate model.

## Figures

| File stem | Content |
|---|---|
| `fig01_workflow` | GMD, PHAL/GMD_active, CP2K, and GMD_SGT responsibilities and closed loop |
| `fig02_configurations` | Actual held-out C16 coordinates: ordinary, torsion, bending, contact |
| `fig03_fragment_fidelity` | Six-setting fidelity calibration, held-out C16, and C32 transfer |
| `fig04_boundary_and_cost` | Atomwise core errors and measured parent/fragment reference costs |
| `fig05_acquisition` | Every observed round for two seeds in five exploratory acquisition arms |
| `fig06_closed_loop` | Events, validation errors, and per-seed topology retention |

The PDF and SVG files are vector graphics produced with Matplotlib. The manuscript includes the PDFs directly; figure PDFs are not a compiled manuscript. Molecular pictures use stored coordinates. No AI-generated molecular geometry or illustrative results are used.

## Regenerate figures and validate

Run from this directory in a Python environment with `numpy`, `matplotlib`, and `PyYAML` available. The tested versions are in [audit/environment.json](audit/environment.json).

```sh
python scripts/make_figures.py
python scripts/validate_bundle.py
```

Figure generation reads only the bundled `data/plot_data.json` and rewrites the six figure triplets and two table fragments. It does not read live experiments, train models, invoke external simulation programs, or compile LaTeX. Validation checks source hashes, per-atom error aggregation, acquisition costs and pairing, topology denominators, configured targets, figure files, LaTeX references, citation keys, and table reproduction.

The already supplied numerical snapshot is sufficient to regenerate every figure. `scripts/snapshot_data.py` is an archival maintenance tool for explicitly refreshing that snapshot from the original completed pilot archive. It requires both the GMD_active repository and its sibling `alkane_artifacts` directory. Do not refresh it merely because a new experiment has started: the allowlist deliberately excludes `alkane_repair`.

```sh
python scripts/snapshot_data.py --repo ../..
```

The JSON snapshots replace workstation paths with provenance tokens. `data/source_manifest.json` contains both original-source and bundled-snapshot SHA-256 hashes, the inspected repository commit, and the transformation applied. Configurations under `data/configs` are archival copies whose inheritance paths refer to the original repository; they are not relocated runnable experiment configurations.

## Before submission

1. Fill and approve author information, contributions, funding, and competing-interest declarations. Review the AI-use statement.
2. Have the authors review the scientific interpretation and references, particularly the exploratory comparison and short-trajectory limitations.
3. Decide how to release the complete code, raw outputs, and checkpoints, then update the data-availability statement with real access details. No public archive identifier has been invented.
4. Compile `main.tex` and inspect page layout, equation breaks, tables, bibliography, and figure placement. Compilation was intentionally not performed for this request.

The document uses standard LaTeX packages and an explicit bibliography, without requiring a generated `.bbl` or shell escape. Keep `main.tex`, `metadata.tex`, `references.tex`, `sections/`, `tables/`, and the six figure PDFs together when preparing a submission source archive. The evidence, scripts, SVGs, PNGs, and audit notes form the accompanying reproduction package.
