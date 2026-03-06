"""
High-level pipeline API for easy use in Jupyter notebooks and scripts.

This module provides a single function that wraps the entire pipeline
with all CLI options available as parameters.
"""
import os
import time
import torch
from typing import Optional, Union, Literal
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Geometry import Point3D

from .aligner import LigandAligner
from .scoring import compute_intramolecular_mask
from .io import load_pocket_bundle, process_query_ligand
from .molecular.mcs import find_mcs_with_positions, auto_select_mcs_mapping
from .molecular.relax import relax_pose_with_fixed_core


def run_pipeline(
    protein_pdb: str,
    ref_ligand: str,
    query_ligand: str,
    output_dir: str = "output_predictions",
    # Conformer generation
    num_confs: int = 1000,
    rmsd_threshold: float = 1.0,
    # MCS options
    mcs_mode: Literal["auto", "single", "multi", "cross"] = "auto",
    min_fragment_size: int = 5,
    max_fragments: int = 3,
    # Force field
    mmff_optimize: bool = True,
    # Optimization
    optimize: bool = False,
    optimizer: Literal["adam", "adamw", "lbfgs"] = "adam",
    opt_steps: int = 100,
    opt_lr: float = 0.05,
    opt_batch_size: int = 8,
    freeze_mcs: bool = True,
    # Scoring
    weight_preset: Literal["vina", "vina_lp", "vinardo"] = "vina",
    torsion_penalty: bool = False,
    # Output
    save_all_poses: Optional[bool] = None,
    top_k: Optional[int] = None,
    # Device
    device: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """
    Run the complete LigAlign pipeline with all options.

    This is a high-level API that wraps the entire pipeline, making it easy
    to use in Jupyter notebooks or Python scripts without dealing with CLI.

    Args:
        protein_pdb: Path to protein pocket PDB file
        ref_ligand: Path to reference ligand SDF file
        query_ligand: SMILES string or path to SDF file
        output_dir: Directory to save results (default: "output_predictions")

        # Conformer Generation
        num_confs: Number of conformers to generate (default: 1000)
        rmsd_threshold: RMSD threshold for clustering in Å (default: 1.0)

        # MCS Options
        mcs_mode: MCS alignment mode (default: "auto")
            - "auto": choose single, multi, or cross based on the molecules
            - "single": 1:1 alignment, fastest
            - "multi": 1:N alignment for symmetric reference
            - "cross": N:M alignment for complex molecules
        min_fragment_size: Min atoms per fragment for cross mode (default: 5)
        max_fragments: Max fragments to find for cross mode (default: 3)

        # Force Field
        mmff_optimize: Apply MMFF94 relaxation after alignment (default: True)

        # Gradient Optimization
        optimize: Enable gradient-based torsion optimization (default: False)
        optimizer: Optimizer type: "adam", "adamw", "lbfgs" (default: "adam")
        opt_steps: Optimization steps (default: 100)
        opt_lr: Learning rate (default: 0.05)
        opt_batch_size: Batch size for optimization (default: 8)
        freeze_mcs: Keep MCS atoms fixed during optimization (default: True)

        # Scoring
        weight_preset: Vina scoring weights (default: "vina")
            - "vina": Standard AutoDock Vina
            - "vina_lp": Vina with local preference
            - "vinardo": Vinardo scoring function
        torsion_penalty: Apply torsional entropy penalty (default: False)

        # Output
        save_all_poses: Save all poses or just top-k (default: None, auto-decided by optimize flag)
        top_k: Number of top poses to save (default: None, saves all if optimize=True else 3)

        # System
        device: Device to use: "cuda", "cpu", or None for auto (default: None)
        verbose: Print progress messages (default: True)

    Returns:
        dict with results:
            - "output_file": Path to output SDF file
            - "num_poses": Number of poses saved
            - "best_score": Best Vina score (kcal/mol)
            - "runtime": Total runtime in seconds
            - "num_conformers": Number of conformers generated
            - "num_representatives": Number of cluster representatives
            - "mcs_size": Number of MCS atoms
            - "mcs_positions": Number of MCS positions found (for multi/cross modes)

    Example:
        >>> from lig_align import run_pipeline
        >>>
        >>> # Basic usage
        >>> results = run_pipeline(
        ...     protein_pdb="pocket.pdb",
        ...     ref_ligand="reference.sdf",
        ...     query_ligand="CC(C)Cc1ccc(cc1)C(C)C(=O)O"
        ... )
        >>> print(f"Best score: {results['best_score']:.2f} kcal/mol")
        >>>
        >>> # With optimization
        >>> results = run_pipeline(
        ...     protein_pdb="pocket.pdb",
        ...     ref_ligand="reference.sdf",
        ...     query_ligand="SMILES",
        ...     optimize=True,
        ...     optimizer="lbfgs",
        ...     num_confs=500
        ... )
        >>>
        >>> # Multi-position alignment
        >>> results = run_pipeline(
        ...     protein_pdb="pocket.pdb",
        ...     ref_ligand="reference.sdf",
        ...     query_ligand="SMILES",
        ...     mcs_mode="multi"
        ... )
    """
    t0 = time.time()

    # Suppress RDKit warnings unless verbose
    if not verbose:
        RDLogger.DisableLog('rdApp.warning')

    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    aligner = LigandAligner(device=device)

    if verbose:
        print(f"Loading reference ligand from {ref_ligand}...")
    ref_suppl = Chem.SDMolSupplier(ref_ligand)
    ref_mol = ref_suppl[0]
    if ref_mol is None:
        raise ValueError(f"Failed to load reference ligand from {ref_ligand}")

    if verbose:
        print(f"Loading protein pocket from {protein_pdb}...")
    pocket_bundle = load_pocket_bundle(protein_pdb, device, aligner.compute_vina_features)
    pocket_mol = pocket_bundle.mol

    query_mol, canonical_smiles = process_query_ligand(query_ligand)
    if verbose:
        print(f"\nProcessing Query Ligand: {canonical_smiles}")

    # 1. MCS Search
    requested_mcs_mode = mcs_mode

    if mcs_mode == "auto":
        auto_choice = auto_select_mcs_mapping(
            ref_mol,
            query_mol,
            min_atoms=3,
            min_fragment_size=min_fragment_size,
            max_fragments=max_fragments,
        )
        mcs_mode = auto_choice["mode"]
        mapping = auto_choice["mapping"]
        num_mcs_positions = len(auto_choice["mappings"])
        if verbose:
            print(f"MCS Mode: auto -> {mcs_mode}")
            print(f"  Reason: {auto_choice['reason']}")
    elif mcs_mode == "single":
        if verbose:
            print("MCS Mode: single")
        mapping = aligner.step2_find_mcs(ref_mol, query_mol, return_all_positions=False)
        num_mcs_positions = 1
    elif mcs_mode == "multi":
        if verbose:
            print("MCS Mode: multi")
        mappings = find_mcs_with_positions(ref_mol, query_mol, return_all=True, min_atoms=3)
        num_mcs_positions = len(mappings)
        if verbose:
            print(f"Found {num_mcs_positions} possible MCS alignment positions")
        mapping = mappings[0]  # Use first for now
        if verbose:
            print(f"Using position 1/{num_mcs_positions} for alignment")
    elif mcs_mode == "cross":
        if verbose:
            print("MCS Mode: cross")
        mappings = find_mcs_with_positions(ref_mol, query_mol,
                                          cross_match=True,
                                          min_fragment_size=min_fragment_size,
                                          max_fragments=max_fragments)
        num_mcs_positions = len(mappings)
        if verbose:
            print(f"Found {num_mcs_positions} cross-matching combinations")
        mapping = mappings[0]  # Use first for now
        if verbose:
            print(f"Using combination 1/{num_mcs_positions} for alignment")
    else:
        raise ValueError(f"Invalid mcs_mode: {mcs_mode}")

    # 2. Extract Exact Coordinates for Constraints
    coordMap = {}
    ref_conf = ref_mol.GetConformer()

    num_mcs_atoms = len(mapping)
    ref_heavy = ref_mol.GetNumHeavyAtoms()
    query_heavy = query_mol.GetNumHeavyAtoms()
    ref_cov = (num_mcs_atoms / ref_heavy * 100) if ref_heavy else 0
    query_cov = (num_mcs_atoms / query_heavy * 100) if query_heavy else 0

    query_mol.SetProp("MCS_Num_Atoms", str(num_mcs_atoms))
    query_mol.SetProp("MCS_Ref_Coverage", f"{ref_cov:.1f}%")
    query_mol.SetProp("MCS_Query_Coverage", f"{query_cov:.1f}%")
    query_mol.SetProp("LigAlign_MCS_Mode", mcs_mode)
    query_mol.SetProp("LigAlign_MCS_Mode_Requested", requested_mcs_mode)

    for ref_idx, query_idx in mapping:
        pos = ref_conf.GetAtomPosition(ref_idx)
        coordMap[query_idx] = Point3D(pos.x, pos.y, pos.z)

    # 3. Generate Conformers with Constraints
    query_mol, rep_cids = aligner.step1_generate_conformers(
        query_mol, num_confs=num_confs, rmsd_threshold=rmsd_threshold, coordMap=coordMap
    )

    if len(rep_cids) == 0:
        raise RuntimeError("Failed to generate conformers")

    # 4. Exact Coordinate Surgery & Force Field Relaxation
    if verbose:
        print("Enforcing exact MCS topology and relaxing query appendages via MMFF94...")
    batch_size = len(rep_cids)
    num_atoms = query_mol.GetNumAtoms()
    aligned_coords = torch.zeros((batch_size, num_atoms, 3))

    mcs_query_indices = {query_idx for _, query_idx in mapping}
    relaxation_messages = []
    relaxation_applied = False

    for j, cid in enumerate(rep_cids):
        conf = query_mol.GetConformer(cid)

        # Teleport exactly
        for ref_idx, query_idx in mapping:
            pos = ref_conf.GetAtomPosition(ref_idx)
            conf.SetAtomPosition(query_idx, Point3D(pos.x, pos.y, pos.z))

        # Relax non-MCS atoms
        if mmff_optimize:
            applied, message = relax_pose_with_fixed_core(query_mol, cid, mcs_query_indices, max_iters=500)
            relaxation_messages.append(message)
            relaxation_applied = relaxation_applied or applied
            if verbose:
                print(f"  Conformer {cid}: {message}")

        aligned_coords[j] = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    aligned_coords = aligned_coords.to(aligner.device)

    # Save pipeline parameters to SDF
    query_mol.SetProp("LigAlign_Num_Confs_Generated", str(num_confs))
    query_mol.SetProp("LigAlign_MMFF_Requested", str(mmff_optimize))
    query_mol.SetProp("LigAlign_MMFF_Optimized", str(relaxation_applied))
    if mmff_optimize:
        query_mol.SetProp("LigAlign_Relaxation_Summary", relaxation_messages[0] if relaxation_messages else "not attempted")

    # 5. Extract features for Vina Scoring
    pocket_coords = pocket_bundle.coords
    pocket_features = pocket_bundle.features
    query_features = aligner.compute_vina_features(query_mol)

    num_rotatable_bonds = None
    if torsion_penalty:
        from rdkit.Chem import rdMolDescriptors
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(query_mol)

    # 6. Score & Save Best Poses
    if verbose:
        print(f"Scoring conformations against protein pocket using '{weight_preset}' weights...")
    intra_mask = compute_intramolecular_mask(query_mol, aligner.device)
    scores = aligner.step4_vina_scoring(aligned_coords, pocket_coords, query_features,
                                       pocket_features, num_rotatable_bonds, weight_preset,
                                       intramolecular_mask=intra_mask)
    initial_scores = scores.clone()

    # 7. Optional Gradient Optimization
    if optimize:
        if verbose:
            print(f"\n--- Running Gradient-Based Torsion Optimization on ALL {len(rep_cids)} Cluster Representatives ---")
        query_indices = [m[1] for m in mapping]

        aligned_coords = aligner.step6_refine_pose(
            mol=query_mol,
            ref_indices=query_indices,
            init_coords=aligned_coords,
            pocket_coords=pocket_coords,
            query_features=query_features,
            pocket_features=pocket_features,
            num_steps=opt_steps,
            lr=opt_lr,
            freeze_mcs=freeze_mcs,
            num_rotatable_bonds=num_rotatable_bonds,
            weight_preset=weight_preset,
            batch_size=opt_batch_size,
            optimizer=optimizer
        )

        # Rescore all optimized poses
        if verbose:
            print("Rescoring optimized poses...")
        new_scores = aligner.step4_vina_scoring(aligned_coords, pocket_coords, query_features,
                                               pocket_features, num_rotatable_bonds, weight_preset,
                                               intramolecular_mask=intra_mask)

        # Report improvement
        score_diffs = new_scores - scores
        best_idx = torch.argmin(new_scores).item()
        if verbose:
            print(f"✓ Optimization complete!")
            print(f"  Best pose: {best_idx} with score {new_scores[best_idx]:.3f} kcal/mol (Δ = {score_diffs[best_idx]:.3f})")
            print(f"  Average improvement: {score_diffs.mean():.3f} kcal/mol")
        scores = new_scores

        # Document in SDF
        query_mol.SetProp("LigAlign_Gradient_Optimized", "True")
        query_mol.SetProp("LigAlign_Optimized_Poses", str(len(rep_cids)))

    # 8. Decide output behavior
    if save_all_poses is None:
        save_all_poses = optimize  # Default: save all if optimized

    if top_k is None:
        if save_all_poses:
            top_k = None  # Save all
        else:
            top_k = 3  # Save top 3

    os.makedirs(output_dir, exist_ok=True)

    # Save poses
    if top_k is None:
        out_sdf = os.path.join(output_dir, "predicted_poses_all.sdf")
        aligner.step5_final_selection(query_mol, rep_cids, aligned_coords, scores,
                                     initial_scores=initial_scores, top_k=None, output_path=out_sdf)
        num_saved = len(rep_cids)
    else:
        out_sdf = os.path.join(output_dir, f"predicted_pose_top{top_k}.sdf")
        aligner.step5_final_selection(query_mol, rep_cids, aligned_coords, scores,
                                     initial_scores=initial_scores, top_k=top_k, output_path=out_sdf)
        num_saved = min(top_k, len(rep_cids))

    t1 = time.time()
    runtime = t1 - t0

    best_score = float(torch.min(scores).item())

    if verbose:
        print(f"\n-> Prediction Completed successfully in {runtime:.2f}s!")
        print(f"-> Results saved to: {out_sdf}")

    # Return results summary
    results = {
        "output_file": out_sdf,
        "num_poses": num_saved,
        "best_score": best_score,
        "runtime": runtime,
        "num_conformers": num_confs,
        "num_representatives": len(rep_cids),
        "mcs_size": num_mcs_atoms,
        "mcs_positions": num_mcs_positions,
        "canonical_smiles": canonical_smiles,
        "device": str(device),
    }

    return results
