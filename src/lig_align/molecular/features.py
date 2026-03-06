"""Molecular feature extraction for Vina scoring."""

import torch
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os

# Initialize RDKit feature factory for scoring-related atom annotations.
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


def compute_vina_features(mol: Chem.Mol, device: torch.device) -> dict:
    """
    Extract atomic features needed for full Vina scoring:
    vdW radii, Hydrophobic flag, HBD (Hydrogen Bond Donor) flag, HBA (Hydrogen Bond Acceptor) flag.
    """
    num_atoms = mol.GetNumAtoms()
    ptable = Chem.GetPeriodicTable()

    radii = torch.zeros(num_atoms, dtype=torch.float32, device=device)
    hydro = torch.zeros(num_atoms, dtype=torch.float32, device=device)
    hbd = torch.zeros(num_atoms, dtype=torch.float32, device=device)
    hba = torch.zeros(num_atoms, dtype=torch.float32, device=device)

    for i, atom in enumerate(mol.GetAtoms()):
        radii[i] = ptable.GetRvdw(atom.GetAtomicNum())

    # Ensure properties and ring info are initialized for feature extraction
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass

    # Extract scoring-relevant atom features.
    feats = factory.GetFeaturesForMol(mol)
    for feat in feats:
        f_type = feat.GetFamily()
        atom_ids = feat.GetAtomIds()

        if f_type == 'Hydrophobe':
            for idx in atom_ids:
                hydro[idx] = 1.0
        elif f_type == 'Donor':
            for idx in atom_ids:
                hbd[idx] = 1.0
        elif f_type == 'Acceptor':
            for idx in atom_ids:
                hba[idx] = 1.0

    return {
        'vdw': radii,
        'hydro': hydro,
        'hbd': hbd,
        'hba': hba
    }
