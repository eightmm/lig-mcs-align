import argparse
import os
import time
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Geometry import Point3D

from lig_align.aligner import LigandAligner
from lig_align.scoring import compute_intramolecular_mask
from lig_align.io import process_query_ligand
from lig_align.molecular.mcs import auto_select_mcs_mapping
from lig_align.molecular.relax import relax_pose_with_fixed_core

def run_prediction(protein_pdb: str, ref_sdf: str, query_arg: str, out_dir: str,
                   num_confs: int = 1000, rmsd_threshold: float = 1.0,
                   mmff_opt: bool = True, optimize: bool = False,
                   freeze_mcs: bool = True, torsion_penalty: bool = False,
                   weight_preset: str = 'vina', opt_batch_size: int = 8,
                   optimizer: str = 'adam', mcs_mode: str = 'auto',
                   min_fragment_size: int = 5, max_fragments: int = 3):
    t0 = time.time()
    
    # Suppress verbose RDKit warnings (e.g., implicit Hs during MMFF94) 
    RDLogger.DisableLog('rdApp.warning')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aligner = LigandAligner(device=device)

    print(f"Loading reference ligand from {ref_sdf}...")
    ref_suppl = Chem.SDMolSupplier(ref_sdf)
    ref_mol = ref_suppl[0]
    if ref_mol is None:
        raise ValueError(f"Failed to load reference ligand from {ref_sdf}")
        
    print(f"Loading protein pocket from {protein_pdb}...")
    pocket_mol = Chem.MolFromPDBFile(protein_pdb, sanitize=False, removeHs=True)
    if pocket_mol is None:
        raise ValueError(f"Failed to load protein pocket from {protein_pdb}")
        
    query_mol, canonical_smiles = process_query_ligand(query_arg)
    print(f"\nProcessing Query Ligand: {canonical_smiles}")
    
    # 1. MCS Search
    requested_mcs_mode = mcs_mode
    try:
        if mcs_mode == 'auto':
            choice = auto_select_mcs_mapping(
                ref_mol,
                query_mol,
                min_atoms=3,
                min_fragment_size=min_fragment_size,
                max_fragments=max_fragments,
            )
            mcs_mode = choice["mode"]
            mapping = choice["mapping"]
            print(f"MCS Mode: auto -> {mcs_mode}")
            print(f"  Reason: {choice['reason']}")
        elif mcs_mode == 'single':
            print("MCS Mode: single")
            # Mode 1: Single position (1:1) - fastest
            mapping = aligner.step2_find_mcs(ref_mol, query_mol, return_all_positions=False)
        elif mcs_mode == 'multi':
            print("MCS Mode: multi")
            # Mode 2: Multi-position (1:N) - find all positions
            mappings = aligner.step2_find_mcs(ref_mol, query_mol, return_all_positions=True)
            print(f"Found {len(mappings)} possible MCS alignment positions")
            # For now, use the first one (TODO: implement trying all)
            mapping = mappings[0]
            print(f"Using position 1/{len(mappings)} for alignment")
        elif mcs_mode == 'cross':
            print("MCS Mode: cross")
            # Mode 3: Cross-matching (N:M) - multiple fragments in both ref and query
            from lig_align.molecular.mcs import find_mcs_with_positions
            mappings = find_mcs_with_positions(ref_mol, query_mol,
                                              cross_match=True,
                                              min_fragment_size=min_fragment_size,
                                              max_fragments=max_fragments)
            print(f"Found {len(mappings)} cross-matching combinations")
            # For now, use the first one (TODO: implement trying all)
            mapping = mappings[0]
            print(f"Using combination 1/{len(mappings)} for alignment")
        else:
            raise ValueError(f"Invalid mcs_mode: {mcs_mode}. Must be 'auto', 'single', 'multi', or 'cross'")
    except Exception as e:
        print(f"MCS Search Failed: {e}")
        return

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
    # ETKDG will use coordMap as distance bounds to approximate the core structure initially
    query_mol, rep_cids = aligner.step1_generate_conformers(
        query_mol, num_confs=num_confs, rmsd_threshold=rmsd_threshold, coordMap=coordMap
    )
    
    if len(rep_cids) == 0:
        print("Failed to generate conformers.")
        return

    # 4. Exact Coordinate Surgery & Force Field Relaxation
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
        if mmff_opt:
            applied, message = relax_pose_with_fixed_core(query_mol, cid, mcs_query_indices, max_iters=500)
            relaxation_messages.append(message)
            relaxation_applied = relaxation_applied or applied
            print(f"  Conformer {cid}: {message}")
                    
        aligned_coords[j] = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        
    aligned_coords = aligned_coords.to(aligner.device)
    
    # Save Pipeline run parameters to SDF
    query_mol.SetProp("LigAlign_Num_Confs_Generated", str(num_confs))
    query_mol.SetProp("LigAlign_MMFF_Requested", str(mmff_opt))
    query_mol.SetProp("LigAlign_MMFF_Optimized", str(relaxation_applied))
    if mmff_opt:
        query_mol.SetProp("LigAlign_Relaxation_Summary", relaxation_messages[0] if relaxation_messages else "not attempted")

    # 5. Extract features for Vina Scoring
    pocket_coords = torch.tensor(pocket_mol.GetConformer().GetPositions(), dtype=torch.float32, device=aligner.device)
    pocket_features = aligner.compute_vina_features(pocket_mol)
    query_features = aligner.compute_vina_features(query_mol)
    
    num_rotatable_bonds = None
    if torsion_penalty:
        from rdkit.Chem import rdMolDescriptors
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(query_mol)
    
    # 6. Score & Save Best Poses
    print(f"Scoring conformations against protein pocket using '{weight_preset}' weights...")
    intra_mask = compute_intramolecular_mask(query_mol, aligner.device)
    scores = aligner.step4_vina_scoring(aligned_coords, pocket_coords, query_features, pocket_features, num_rotatable_bonds, weight_preset, intramolecular_mask=intra_mask)
    initial_scores = scores.clone()
    
    if optimize:
        print(f"\n--- Running Gradient-Based Torsion Optimization on ALL {len(rep_cids)} Cluster Representatives ---")
        query_indices = [m[1] for m in mapping]

        # Optimize all poses (automatically batched)
        aligned_coords = aligner.step6_refine_pose(
            mol=query_mol,
            ref_indices=query_indices,
            init_coords=aligned_coords,  # Accepts both [N,3] and [B,N,3]
            pocket_coords=pocket_coords,
            query_features=query_features,
            pocket_features=pocket_features,
            num_steps=100,
            lr=0.05,
            freeze_mcs=freeze_mcs,
            num_rotatable_bonds=num_rotatable_bonds,
            weight_preset=weight_preset,
            batch_size=opt_batch_size,
            optimizer=optimizer
        )

        # Rescore all optimized poses
        print("Rescoring optimized poses...")
        new_scores = aligner.step4_vina_scoring(aligned_coords, pocket_coords, query_features, pocket_features, num_rotatable_bonds, weight_preset, intramolecular_mask=intra_mask)

        # Report improvement
        score_diffs = new_scores - scores
        best_idx = torch.argmin(new_scores).item()
        print(f"✓ Optimization complete!")
        print(f"  Best pose: {best_idx} with score {new_scores[best_idx]:.3f} kcal/mol (Δ = {score_diffs[best_idx]:.3f})")
        print(f"  Average improvement: {score_diffs.mean():.3f} kcal/mol")
        scores = new_scores

        # Document in SDF
        query_mol.SetProp("LigAlign_Gradient_Optimized", "True")
        query_mol.SetProp("LigAlign_Optimized_Poses", str(len(rep_cids)))
    
    os.makedirs(out_dir, exist_ok=True)

    # Save all poses if optimized, otherwise top 3
    if optimize:
        out_sdf = os.path.join(out_dir, "predicted_poses_all.sdf")
        aligner.step5_final_selection(query_mol, rep_cids, aligned_coords, scores, initial_scores=initial_scores, top_k=None, output_path=out_sdf)
    else:
        out_sdf = os.path.join(out_dir, "predicted_pose_top3.sdf")
        aligner.step5_final_selection(query_mol, rep_cids, aligned_coords, scores, initial_scores=initial_scores, top_k=3, output_path=out_sdf)
    
    t1 = time.time()
    print(f"\n-> Prediction Completed successfully in {t1-t0:.2f}s!")
    print(f"-> Results saved to: {out_sdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LigAlign: High-speed 3D Ligand Pose Predicton Pipeline")
    parser.add_argument("-p", "--protein", required=True, help="Path to the protein pocket PDB file")
    parser.add_argument("-r", "--ref_ligand", required=True, help="Path to the reference ligand SDF file")
    parser.add_argument("-q", "--query_ligand", required=True, help="SMILES string or path to an SDF file of the query ligand")
    parser.add_argument("-o", "--out_dir", default="output_predictions", help="Directory to save the resulting SDF file (default: output_predictions)")
    parser.add_argument("-n", "--num_confs", type=int, default=1000, help="Number of conformers to generate (default: 1000)")
    parser.add_argument("--rmsd_threshold", type=float, default=1.0, help="RMSD threshold (Å) for clustering (default: 1.0)")
    parser.add_argument("--no_mmff", action="store_true", help="Disable MMFF94 force field optimization for query appendages")
    parser.add_argument("--optimize", action="store_true", help="Enable Gradient-based Torsion Optimization on ALL cluster representatives")
    parser.add_argument("--opt_batch_size", type=int, default=8, help="Batch size for optimization (default: 8)")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "lbfgs"], default="adam", help="Optimizer for torsion optimization: adam, adamw, or lbfgs (default: adam)")
    parser.add_argument("--free_mcs", action="store_true", help="During optimization, let the MCS also optimize instead of acting as a rigid anchor")
    parser.add_argument("--torsion_penalty", action="store_true", help="Apply AutoDock Vina Torsional Entropy penalty (N_rot) to the score")
    parser.add_argument("--weight_preset", type=str, choices=["vina", "vina_lp", "vinardo"], default="vina", help="Preset dictionary for Vina functional weights")

    # MCS Mode Options
    parser.add_argument("--mcs_mode", type=str, choices=["auto", "single", "multi", "cross"], default="auto",
                        help="MCS alignment mode: 'auto' (choose based on symmetry/fragmentation), 'single' (1:1, fastest), 'multi' (1:N, symmetric ref), 'cross' (N:M, both symmetric) (default: auto)")
    parser.add_argument("--min_fragment_size", type=int, default=5,
                        help="Minimum atoms per fragment for cross-matching mode (default: 5)")
    parser.add_argument("--max_fragments", type=int, default=3,
                        help="Maximum fragments to find for cross-matching mode (default: 3)")

    args = parser.parse_args()

    run_prediction(args.protein, args.ref_ligand, args.query_ligand, args.out_dir,
                   num_confs=args.num_confs, rmsd_threshold=args.rmsd_threshold,
                   mmff_opt=not args.no_mmff,
                   optimize=args.optimize, freeze_mcs=not args.free_mcs,
                   torsion_penalty=args.torsion_penalty, weight_preset=args.weight_preset,
                   opt_batch_size=args.opt_batch_size, optimizer=args.optimizer,
                   mcs_mode=args.mcs_mode, min_fragment_size=args.min_fragment_size,
                   max_fragments=args.max_fragments)
