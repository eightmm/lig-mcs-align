# LigAlign

LigAlign is a ligand pose prediction pipeline built around MCS-guided alignment, PyTorch-based Vina scoring, and optional gradient-based torsion optimization.

## What It Does

- Uses a reference ligand to anchor query poses with MCS matching
- Scores candidate poses with differentiable Vina-style energy terms
- Refines poses with batched torsion optimization on GPU
- Exports ranked poses and visualization assets for analysis

## Why This Repository Exists

LigAlign targets a practical workflow for structure-guided ligand placement:

- reference ligand available
- query molecule given as SMILES or SDF
- fast pose generation and rescoring needed
- optimization and inspection should stay scriptable

## Quick Start

### Environment

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Run The Pipeline

```bash
uv run python scripts/run_pipeline.py \
  -p examples/10gs/10gs_pocket.pdb \
  -r examples/10gs/10gs_ligand.sdf \
  -q "CCO" \
  -o output/
```

### Use From Python

```python
from lig_align import run_pipeline

results = run_pipeline(
    protein_pdb="examples/10gs/10gs_pocket.pdb",
    ref_ligand="examples/10gs/10gs_ligand.sdf",
    query_ligand="CCO",
    output_dir="output",
)

print(results["best_score"])
print(results["output_file"])
```

## Example Assets

The repository includes a ready-to-run example set under `examples/10gs`:

- `examples/10gs/10gs_pocket.pdb`: example protein pocket
- `examples/10gs/10gs_ligand.sdf`: reference ligand used as the alignment anchor
- `examples/10gs/visualizations/`: GIFs used in reports and method comparisons

## Notes On Output Metadata

Exported SDF files can contain useful run metadata for inspection:

- `LigAlign_MCS_Mode` and `LigAlign_MCS_Mode_Requested`
- `LigAlign_MMFF_Requested`, `LigAlign_MMFF_Optimized`, and `LigAlign_Relaxation_Summary`
- `Vina_Score_Initial`, `Vina_Score_Final`, and `Vina_Score_Delta`

This makes it easier to tell whether relaxation ran, how `auto` MCS resolved, and whether optimization improved the score.

## Repository Guide

- [docs/README.md](docs/README.md): documentation index
- [docs/USAGE.md](docs/USAGE.md): installation, CLI usage, and common workflows
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md): Python and script-level interfaces
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): pipeline stages and implementation notes
- [reports/README.md](reports/README.md): progress summary and presentation assets

## Current Status

- Core pipeline implemented in `src/lig_align`
- Example input and output assets included under `examples/10gs`
- Documentation reorganized around usage, API, and architecture
- Progress summary and presentation assets added under `reports/`

## Project Layout

```text
lig-align/
├── src/lig_align/      # package source
├── scripts/            # command-line entry points and visualization scripts
├── tests/              # regression and feature tests
├── examples/           # sample inputs and generated visualization assets
├── docs/               # detailed usage, API, architecture, and design notes
└── reports/            # progress summary and presentation-oriented material
```
