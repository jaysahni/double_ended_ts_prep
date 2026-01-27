"""Module for force field-based molecular optimization.

This module provides methods to optimize a list of molecules in 3D space using
the MMFF94 force field. The workflow involves:

1. Preparing individual molecules from SMILES with explicit hydrogens and 3D coordinates
2. Spatially arranging molecules to avoid overlaps
3. Combining molecules into a single system
4. Constructing and running the MMFF94 force field optimization
"""

from rdkit import Chem
from rdkit.Chem import AllChem

# Conversion factor: 1 Hartree = 627.5094740631 kcal/mol
KCAL_MOL_TO_HARTREE = 1.0 / 627.5094740631


def prepare_molecule_from_smiles(smiles: str) -> Chem.Mol:
    """Prepare a single molecule from SMILES with 3D coordinates.

    Converts a SMILES string to an RDKit molecule with explicit hydrogens
    and generates initial 3D coordinates using distance geometry embedding.
    Preserves any existing atom mapping labels and explicit hydrogens.

    Args:
        smiles: A valid SMILES string representing the molecule

    Returns:
        RDKit Mol object with explicit hydrogens and 3D coordinates

    Raises:
        ValueError: If the SMILES is invalid or embedding fails

    Examples:
        >>> mol = prepare_molecule_from_smiles("CCO")
        >>> mol.GetNumAtoms()
        9
        >>> mol.GetConformer().Is3D()
        True
        >>> mol = prepare_molecule_from_smiles("[CH3:1][OH:2]")
        >>> mol.GetAtomWithIdx(0).GetAtomMapNum()
        1
    """
    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    mol = Chem.MolFromSmiles(smiles, ps)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol_with_h = Chem.AddHs(mol, addCoords=False)

    embed_result = AllChem.EmbedMolecule(  # type: ignore[attr-defined]
        mol_with_h,
        AllChem.ETKDGv3(),  # type: ignore[attr-defined]
    )
    if embed_result == -1:
        raise ValueError(f"Failed to generate 3D coordinates for: {smiles}")

    return mol_with_h


def translate_molecule(mol: Chem.Mol, offset: tuple[float, float, float]) -> None:
    """Translate a molecule's coordinates by a given offset.

    Modifies the molecule's conformer in place by adding the offset
    to all atomic coordinates.

    Args:
        mol: RDKit Mol object with a conformer
        offset: Translation vector (x, y, z) in Angstroms

    Raises:
        ValueError: If the molecule has no conformer

    Examples:
        >>> mol = prepare_molecule_from_smiles("C")
        >>> conf = mol.GetConformer()
        >>> original_x = conf.GetAtomPosition(0).x
        >>> translate_molecule(mol, (5.0, 0.0, 0.0))
        >>> abs(conf.GetAtomPosition(0).x - original_x - 5.0) < 0.001
        True
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer to translate")

    conf = mol.GetConformer()
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        new_pos = (pos.x + offset[0], pos.y + offset[1], pos.z + offset[2])
        conf.SetAtomPosition(atom_idx, new_pos)


def combine_molecules(mols: list[Chem.Mol], spacing: float = 5.0) -> Chem.Mol:
    """Combine multiple molecules into a single system with spatial separation.

    Each molecule is translated along the x-axis to avoid initial overlaps,
    then all molecules are merged into a single RDKit mol object. The molecules
    remain chemically separate (no bonds between them) but share a common
    coordinate system.

    Args:
        mols: List of RDKit Mol objects, each with a conformer
        spacing: Distance in Angstroms between molecule centers along x-axis

    Returns:
        Combined RDKit Mol object containing all input molecules

    Raises:
        ValueError: If the molecule list is empty or any molecule lacks a conformer

    Examples:
        >>> mol1 = prepare_molecule_from_smiles("C")
        >>> mol2 = prepare_molecule_from_smiles("O")
        >>> combined = combine_molecules([mol1, mol2], spacing=10.0)
        >>> combined.GetNumAtoms() == mol1.GetNumAtoms() + mol2.GetNumAtoms()
        True
    """
    if not mols:
        raise ValueError("Cannot combine an empty list of molecules")

    for i, mol in enumerate(mols):
        if mol.GetNumConformers() == 0:
            raise ValueError(f"Molecule at index {i} has no conformer")
        translate_molecule(mol, (i * spacing, 0.0, 0.0))

    combined = mols[0]
    for mol in mols[1:]:
        combined = Chem.CombineMols(combined, mol)

    Chem.FastFindRings(combined)

    return combined


def optimize_system(
    mol: Chem.Mol,
    max_iters: int = 500,
) -> tuple[Chem.Mol, float]:
    """Optimize a molecular system using the MMFF94 force field.

    Constructs an MMFF94 force field for the given molecule and runs
    gradient-based energy minimization. The force field accounts for:
    - Bond stretching
    - Angle bending
    - Torsional rotation
    - Van der Waals interactions (Lennard-Jones)
    - Electrostatic interactions between partial atomic charges

    Args:
        mol: RDKit Mol object with a conformer
        max_iters: Maximum number of minimization iterations

    Returns:
        Tuple of (optimized molecule, final energy in Hartrees)

    Raises:
        ValueError: If the molecule has no conformer or MMFF94 setup fails

    Examples:
        >>> mol = prepare_molecule_from_smiles("CCO")
        >>> opt_mol, energy = optimize_system(mol, max_iters=100)
        >>> opt_mol.GetNumAtoms() == mol.GetNumAtoms()
        True
        >>> isinstance(energy, float)
        True
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer to optimize")

    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)  # type: ignore[attr-defined]
    if mmff_props is None:
        raise ValueError("Failed to get MMFF94 properties for molecule")

    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)  # type: ignore[attr-defined]
    if ff is None:
        raise ValueError("Failed to construct MMFF94 force field")

    ff.Minimize(maxIts=max_iters)

    energy_kcal = ff.CalcEnergy()
    energy_hartree = energy_kcal * KCAL_MOL_TO_HARTREE

    return mol, energy_hartree
