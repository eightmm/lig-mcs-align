# Usage Guide

This document holds practical setup and execution details that are too verbose for the top-level README.

## Installation

```bash
uv venv
source .venv/bin/activate
uv sync
```

If `uv sync` is not suitable for your environment, install the dependencies declared in [pyproject.toml](/home/jaemin/project/protein-ligand/lig-align/pyproject.toml).

## Main CLI

Primary entry point:

```bash
uv run python scripts/run_pipeline.py \
  -p examples/10gs/10gs_pocket.pdb \
  -r examples/10gs/10gs_ligand.sdf \
  -q "CCO" \
  -o output/
```

Required arguments:

```text
-p, --protein        Protein pocket PDB file
-r, --ref_ligand     Reference ligand SDF file
-q, --query_ligand   Query ligand as SMILES or SDF path
```

Common optional arguments:

```text
-o, --out_dir             Output directory
-n, --num_confs           Number of conformers to generate
--rmsd_threshold          RMSD threshold for clustering
--mcs_mode                single | multi | cross
--optimize                Enable gradient optimization
--opt_batch_size          Number of poses optimized together
--optimizer               adam | adamw | lbfgs
--weight_preset           vina | vina_lp | vinardo
--free_mcs                Allow MCS atoms to move during optimization
--torsion_penalty         Apply torsional entropy penalty
```

## Common Workflows

### Basic Prediction

```bash
uv run python scripts/run_pipeline.py \
  -p examples/10gs/10gs_pocket.pdb \
  -r examples/10gs/10gs_ligand.sdf \
  -q "CCO" \
  -o output/
```

### Recommended Optimized Run

```bash
uv run python scripts/run_pipeline.py \
  -p protein.pdb \
  -r ref_ligand.sdf \
  -q "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \
  -n 1000 \
  --rmsd_threshold 1.0 \
  --optimize \
  --optimizer lbfgs \
  --opt_batch_size 8
```

### Query From SDF

```bash
uv run python scripts/run_pipeline.py \
  -p protein.pdb \
  -r ref_ligand.sdf \
  -q query_ligand.sdf \
  -o output_sdf/
```

### Optimize Existing Pose Only

```bash
uv run python scripts/optimize_pose.py \
  -p protein.pdb \
  -l ligand.sdf \
  -o optimized.sdf \
  --steps 200 \
  --optimizer lbfgs
```

## Example Assets

Sample inputs and visual outputs live under `examples/10gs`.

Useful files:

- `examples/10gs/10gs_pocket.pdb`
- `examples/10gs/10gs_ligand.sdf`
- `examples/10gs/visualizations/test_run.gif`
- `examples/10gs/visualizations/test_ref_run.gif`
- `examples/10gs/visualizations/combinatorial/`

## Testing

Run the existing tests with `uv` so the same environment definition is reused.

```bash
uv run python tests/test_mcs_modes_cli.py
uv run python tests/test_pipeline_api.py
uv run python tests/test_unified_mcs_api.py
```
