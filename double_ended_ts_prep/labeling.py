"""Module for atom labelling from molecule inputs.

This module contains methods to generate SMIRKS expressions from molecules,
as well as the labeling methods that process SMIRKS expressions into atom-labeled RDKit molecules
for later atom-tracking steps.
"""

import warnings

from rdkit import Chem
from rxnmapper import RXNMapper

# minimum confidence for successful use of rxnmapper without warning
CONFIDENCE_MINIMUM = 0.7


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
    rxn_mapper = RXNMapper()

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


def parse_smirks(smirks: str) -> dict[str, list[Chem.Mol]]:
    """Parse a SMIRKS string and return a dictionary containing RDKit molecule objects
    for reactants and products.

    Args:
        smirks: A SMIRKS string in the format "reactants>>products"

    Returns:
        Dictionary with keys:
            - 'reactants': List of RDKit Mol objects
            - 'products': List of RDKit Mol objects

    Raises:
        ValueError: If SMIRKS contains invalid SMILES strings

    Examples:
        >>> result = parse_smirks("[CH3:1][OH:2]>>[CH3:1][O:2][CH3:1]")
        >>> len(result['reactants'])
        1
        >>> len(result['products'])
        1
        >>> result = parse_smirks("CCO.CC(=O)O>>CCOC(C)=O.O")
        >>> len(result['reactants'])
        2
        >>> len(result['products'])
        2
        >>> parse_smirks("invalid>>C")
        Traceback (most recent call last):
            ...
        ValueError: Invalid reactant SMILES: invalid
        >>> parse_smirks("C>>invalid")
        Traceback (most recent call last):
            ...
        ValueError: Invalid product SMILES: invalid
    """
    reactants = []
    products = []

    # Split SMIRKS by reaction arrow
    reactants_str, products_str, *_rest = smirks.split(">>")

    # Configure SMILES parser to preserve explicit hydrogens and atom mapping
    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    # Parse reactants with explicit hydrogens preserved
    if reactants_str.strip():
        for smiles_raw in reactants_str.split("."):
            if smiles := smiles_raw.strip():
                if (mol := Chem.MolFromSmiles(smiles, ps)) is None:
                    raise ValueError(f"Invalid reactant SMILES: {smiles}")
                # Add any implicit hydrogens while preserving existing explicit ones
                mol_with_h = Chem.AddHs(mol, addCoords=False)
                reactants.append(mol_with_h)
    # Parse products with explicit hydrogens preserved
    if products_str.strip():
        for smiles_raw in products_str.split("."):
            if smiles := smiles_raw.strip():
                if (mol := Chem.MolFromSmiles(smiles, ps)) is None:
                    raise ValueError(f"Invalid product SMILES: {smiles}")
                # Add any implicit hydrogens while preserving existing explicit ones
                mol_with_h = Chem.AddHs(mol, addCoords=False)
                products.append(mol_with_h)

    return {"reactants": reactants, "products": products}
