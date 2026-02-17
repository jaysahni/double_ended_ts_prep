"""Module for atom labelling from molecule inputs.

This module contains methods to generate SMIRKS expressions from molecules,
as well as the labeling methods that process SMIRKS expressions into atom-labeled RDKit molecules
for later atom-tracking steps.
"""

import functools
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem
from rxnmapper import RXNMapper

# minimum confidence for successful use of rxnmapper without warning
CONFIDENCE_MINIMUM = 0.7


@functools.lru_cache(maxsize=1)
def _get_rxn_mapper() -> RXNMapper:
    """Return a cached RXNMapper instance (avoids reloading the model)."""
    return RXNMapper()


def build_smirks(reactants: list[str], products: list[str]) -> str:
    """Build a SMIRKS string from lists of reactant and product SMILES.

    Args:
        reactants: List of SMILES strings for reactants
        products: List of SMILES strings for products

    Returns:
    -------
        A valid SMIRKS string
    Raises:
        ValueError: If reactants or products are invalid or empty

    Example:
        >>> smirks = build_smirks(["CCO", "CC(=O)O"], ["CC(=O)OCC", "O"])
        >>> print(smirks)
        [CH3][CH2][OH].[CH3][C](=[O])[OH]>>[CH3][CH2][O][C]([CH3])=[O].[OH2]
    """
    if not reactants:
        raise ValueError("Reactants list cannot be empty")
    if not products:
        raise ValueError("Products list cannot be empty")
    # Validate all SMILES strings
    validated_reactants = []
    for smiles_raw in reactants:
        if not (smiles := smiles_raw.strip()):
            raise ValueError("Empty SMILES string in reactants")
        if (mol := Chem.MolFromSmiles(smiles)) is None:
            raise ValueError(f"Invalid reactant SMILES: {smiles}")
        # Use canonical SMILES with explicit hydrogens and stereochemistry
        validated_reactants.append(Chem.MolToSmiles(mol, allHsExplicit=True, isomericSmiles=True))

    validated_products = []
    for smiles_raw in products:
        if not (smiles := smiles_raw.strip()):
            raise ValueError("Empty SMILES string in products")
        if (mol := Chem.MolFromSmiles(smiles)) is None:
            raise ValueError(f"Invalid product SMILES: {smiles}")
        # Use canonical SMILES with explicit hydrogens and stereochemistry
        validated_products.append(Chem.MolToSmiles(mol, allHsExplicit=True, isomericSmiles=True))

    # Join reactants with '.'
    reactants_str = ".".join(validated_reactants)
    products_str = ".".join(validated_products)

    # No reagents: use >> format
    return f"{reactants_str}>>{products_str}"


def map_smirks(unmapped_smirks: str) -> str:
    """Convert an unlabeled smirks to an entirely labeled smirks.

    Args:
        unmapped_smirks: An unlabeled SMIRKS string

    Returns:
        A labeled SMIRKS string with atom mapping numbers

    Warns:
        UserWarning: If the confidence of the mapping is lower than the confidence minimum

    Examples:
        >>> unmapped = "CC(C)S>>CC(C)Sc1ncccc1F"
        >>> mapped = map_smirks(unmapped)
        >>> ":" in mapped  # Check that atom mapping numbers are present
        True
        >>> mapped.count(">>")  # Should still have the reaction arrow
        1
    """
    rxn_mapper = _get_rxn_mapper()

    # run rxn_mapper
    results = rxn_mapper.get_attention_guided_atom_maps([unmapped_smirks])

    # Extract mapped reaction and confidence from result dictionary
    result_dict = results[0]
    mapped_rxn = result_dict["mapped_rxn"]
    conf = result_dict["confidence"]

    # give warning if confidence is low
    if conf < CONFIDENCE_MINIMUM:
        warnings.warn(
            f"Confidence of atom mapping is {conf}, which may lead to incorrect optimization.",
            stacklevel=2,
        )

    return mapped_rxn


def smirks_to_molecules(smirks: str) -> dict[str, list[Chem.Mol]]:
    """Parse a SMIRKS string and return molecules with 3D coordinates.

    Combines SMIRKS parsing with 3D coordinate generation. Each molecule
    is prepared with explicit hydrogens and embedded using ETKDGv3 distance
    geometry for realistic initial conformations. Atom mapping labels are preserved.

    Args:
        smirks: A SMIRKS string in the format "reactants>>products"

    Returns:
        Dictionary with keys:
            - 'reactants': List of RDKit Mol objects with 3D coordinates
            - 'products': List of RDKit Mol objects with 3D coordinates

    Raises:
        ValueError: If SMIRKS contains invalid SMILES or embedding fails

    Examples:
        >>> result = smirks_to_molecules("[CH3:1][OH:2]>>[CH3:1][O:2][CH3:3]")
        >>> len(result['reactants'])
        1
        >>> result['reactants'][0].GetConformer().Is3D()
        True
        >>> result['reactants'][0].GetAtomWithIdx(0).GetAtomMapNum()
        1
    """
    reactants = []
    products = []

    reactants_str, products_str, *_rest = smirks.split(">>")

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    if reactants_str.strip():
        for smiles_raw in reactants_str.split("."):
            if smiles := smiles_raw.strip():
                if (mol := Chem.MolFromSmiles(smiles, ps)) is None:
                    raise ValueError(f"Invalid reactant SMILES: {smiles}")
                mol_with_h = Chem.AddHs(mol, addCoords=False)
                embed_result = AllChem.EmbedMolecule(  # type: ignore[attr-defined]
                    mol_with_h,
                    AllChem.ETKDGv3(),  # type: ignore[attr-defined]
                )
                if embed_result == -1:
                    raise ValueError(f"Failed to embed reactant: {smiles}")
                reactants.append(mol_with_h)

    if products_str.strip():
        for smiles_raw in products_str.split("."):
            if smiles := smiles_raw.strip():
                if (mol := Chem.MolFromSmiles(smiles, ps)) is None:
                    raise ValueError(f"Invalid product SMILES: {smiles}")
                mol_with_h = Chem.AddHs(mol, addCoords=False)
                embed_result = AllChem.EmbedMolecule(  # type: ignore[attr-defined]
                    mol_with_h,
                    AllChem.ETKDGv3(),  # type: ignore[attr-defined]
                )
                if embed_result == -1:
                    raise ValueError(f"Failed to embed product: {smiles}")
                products.append(mol_with_h)

    return {"reactants": reactants, "products": products}


def transfer_atom_mapping(
    mapped_smirks: str,
    xyz_reactants: list[Chem.Mol],
    xyz_products: list[Chem.Mol],
) -> dict[str, list[Chem.Mol]]:
    """Transfer atom mapping numbers from a mapped SMIRKS to pre-built molecules.

    Parses the mapped SMIRKS to extract per-molecule mapped SMILES, then uses
    substructure matching on heavy atoms to assign atom mapping numbers onto the
    XYZ-loaded molecules.  Hydrogen atoms are left unmapped (map_num 0), which
    is consistent with how ``get_atom_mapping_correspondence`` operates.

    Args:
        mapped_smirks: Atom-mapped SMIRKS from RXNMapper.
        xyz_reactants: RDKit Mol objects loaded from XYZ (with 3D coords).
        xyz_products: RDKit Mol objects loaded from XYZ (with 3D coords).

    Returns:
        Dictionary with keys ``'reactants'`` and ``'products'``, each a list
        of the input Mol objects with atom mapping numbers set.

    Raises:
        ValueError: If molecule counts don't match or substructure matching
            fails.
    """
    reactants_str, products_str, *_ = mapped_smirks.split(">>")

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    def _parse_mapped_smiles(side_str: str) -> list[Chem.Mol]:
        mols = []
        for raw_smi in side_str.split("."):
            smi = raw_smi.strip()
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi, ps)
            if mol is None:
                raise ValueError(f"Invalid mapped SMILES: {smi}")
            mols.append(mol)
        return mols

    mapped_r = _parse_mapped_smiles(reactants_str)
    mapped_p = _parse_mapped_smiles(products_str)

    if len(mapped_r) != len(xyz_reactants):
        raise ValueError(
            f"Mapped SMIRKS has {len(mapped_r)} reactants but "
            f"{len(xyz_reactants)} XYZ reactants were provided"
        )
    if len(mapped_p) != len(xyz_products):
        raise ValueError(
            f"Mapped SMIRKS has {len(mapped_p)} products but "
            f"{len(xyz_products)} XYZ products were provided"
        )

    def _apply_mapping(mapped_mol: Chem.Mol, xyz_mol: Chem.Mol) -> None:
        """Set atom map numbers on *xyz_mol* using heavy-atom substructure match."""
        # Clear any existing mapping
        for atom in xyz_mol.GetAtoms():
            atom.SetAtomMapNum(0)

        # Build heavy-atom-only versions for matching
        xyz_noH = Chem.RemoveAllHs(xyz_mol)
        # mapped_mol from SMIRKS already has no explicit Hs (parsed without removeHs=False
        # but SMIRKS atoms are typically heavy-atom only from RXNMapper)
        mapped_noH = Chem.RemoveAllHs(mapped_mol)

        match = xyz_noH.GetSubstructMatch(mapped_noH)
        if not match:
            raise ValueError(
                f"Substructure match failed between XYZ molecule and mapped "
                f"SMILES: {Chem.MolToSmiles(mapped_mol)}"
            )

        # Build index map: heavy atom idx in xyz_noH -> full idx in xyz_mol
        heavy_to_full: list[int] = []
        for atom in xyz_mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                heavy_to_full.append(atom.GetIdx())

        # Transfer mapping numbers
        for mapped_idx, xyz_noH_idx in enumerate(match):
            map_num = mapped_noH.GetAtomWithIdx(mapped_idx).GetAtomMapNum()
            if map_num > 0:
                full_idx = heavy_to_full[xyz_noH_idx]
                xyz_mol.GetAtomWithIdx(full_idx).SetAtomMapNum(map_num)

    for mapped_mol, xyz_mol in zip(mapped_r, xyz_reactants, strict=True):
        _apply_mapping(mapped_mol, xyz_mol)
    for mapped_mol, xyz_mol in zip(mapped_p, xyz_products, strict=True):
        _apply_mapping(mapped_mol, xyz_mol)

    return {"reactants": list(xyz_reactants), "products": list(xyz_products)}
