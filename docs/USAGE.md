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
--mcs_mode                auto | single | multi | cross
--optimize                Enable gradient optimization
--opt_batch_size          Number of poses optimized together (default: 128)
--optimizer               adam | adamw | lbfgs
--weight_preset           vina | vina_lp | vinardo
--free_mcs                Allow MCS atoms to move during optimization
--no_torsion_penalty      Disable the default Vina torsional entropy penalty
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
  --optimizer lbfgs
```

Scoring note:

- `Vina` scores include the standard torsional entropy penalty by default
- use `--no_torsion_penalty` only when you explicitly want interaction-only scores
- `opt_batch_size=128` is now the default for same-molecule multi-pose optimization on GPU
- reduce it if your ligand leaves many representative poses or if GPU memory becomes limiting

### MCS Mode Guidance

- `auto`
  - recommended default
  - chooses `multi` for symmetry-equivalent placements
  - chooses `cross` only when multi-fragment matching increases total mapped atoms
  - otherwise uses `single`
- `single`
  - use for the fastest and most conservative contiguous-core match
- `multi`
  - use when the reference is symmetric and you want all equivalent placements enumerated
  - current pipeline still continues with the first candidate after enumeration
- `cross`
  - use when one contiguous MCS is too restrictive and multiple fragments matter
  - current pipeline still continues with the first generated combination

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

## Relaxation And Score Metadata

Each exported SDF can include run metadata that explains what happened during placement and optimization.

Important fields:

- `LigAlign_MCS_Mode`: the mode actually used after `auto` resolution
- `LigAlign_MCS_Mode_Requested`: the mode requested by the user
- `LigAlign_MMFF_Requested`: whether relaxation was requested
- `LigAlign_MMFF_Optimized`: whether relaxation actually ran successfully
- `LigAlign_Relaxation_Summary`: why relaxation was applied, skipped, or fell back
- `Vina_Score_Initial`: score before gradient optimization
- `Vina_Score_Final`: score after optimization or final ranking pass
- `Vina_Score_Delta`: final minus initial score

Practical interpretation:

- if `LigAlign_MMFF_Requested=True` and `LigAlign_MMFF_Optimized=False`, the pipeline judged that relaxation was not safe or not meaningful for that pose
- if `Vina_Score_Delta` is negative, optimization improved the score
- if `Vina_Score_Delta` is near zero, either the pose was already near a local minimum or there were no useful torsional moves available

## Testing

Run the existing tests with `uv` so the same environment definition is reused.

```bash
uv run python tests/test_mcs_modes_cli.py
uv run python tests/test_pipeline_api.py
uv run python tests/test_unified_mcs_api.py
uv run python tests/test_mcs_auto_mode.py
```
