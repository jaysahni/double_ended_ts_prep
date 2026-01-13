"""Module for atom labelling from molecule inputs.

This module contains methods to generate SMIRKS expressions from molecules,
as well as the labeling methods that process SMIRKS expressions into atom-labeled RDKit molecules
for later atom-tracking steps.
"""

import warnings
from typing import Dict, List, Optional, Union

from rdkit import Chem
from rxnmapper import RXNMapper

# minimum confidence for successful use of rxnmapper without warning
CONFIDENCE_MINIMUM = 0.7


def build_smirks(reactants: List[str], products: List[str]) -> str:
    """
    Build a SMIRKS string from lists of reactant and product SMILES.

    Args:
        reactants: List of SMILES strings for reactants
        products: List of SMILES strings for products

    Returns
    -------
        A valid SMIRKS string
    Raises:
        ValueError: If reactants or products are invalid or empty

    Example:
        >>> smirks = build_smirks(["CCO", "CC(=O)O"], ["CC(=O)OCC", "O"])
        >>> print(smirks)
        CCO.CC(=O)O>>CCOC(C)=O.O
    """
    if not reactants:
        raise ValueError("Reactants list cannot be empty")
    if not products:
        raise ValueError("Products list cannot be empty")

    # Validate all SMILES strings
    validated_reactants = []
    for smiles_raw in reactants:
        smiles = smiles_raw.strip()
        if not smiles:
            raise ValueError("Empty SMILES string in reactants")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid reactant SMILES: {smiles}")
        # Use canonical SMILES for consistency
        validated_reactants.append(Chem.MolToSmiles(mol))

    validated_products = []
    for smiles_raw in products:
        smiles = smiles_raw.strip()
        if not smiles:
            raise ValueError("Empty SMILES string in products")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid product SMILES: {smiles}")
        validated_products.append(Chem.MolToSmiles(mol))

    # Join reactants with '.'
    reactants_str = ".".join(validated_reactants)
    products_str = ".".join(validated_products)

    # No reagents: use >> format
    return f"{reactants_str}>>{products_str}"


def map_smirks(unmapped_smirks: str) -> str:
    """Convert an unlabeled smirks to an entirely labeled smirks.

    Input: unlabeled SMIRKS string
    Output: labeled SMIRKS string

    Raises warning if the confidence of the mapping is lower than the confidence minimum
    """
    rxn_mapper = RXNMapper()

    # run rxn_mapper
    mapper_results = rxn_mapper.get_attention_guided_atom_maps([unmapped_smirks])[0]

    # convert from dict to tuple
    mapper_results = next(iter(mapper_results.items()))
    conf = mapper_results[1]
    # give warning if confidence is low
    if conf < CONFIDENCE_MINIMUM:
        warnings.warn(
            f"Confidence of atom mapping is {conf}, which may lead to incorrect optimization.",
            stacklevel=2,
        )

    return mapper_results[0]


def parse_smirks(smirks: str) -> Dict[str, Union[List[Chem.Mol], bool, Optional[str]]]:
    """
    Parse a SMIRKS string and return a dictionary containing RDKit molecule objects
    for reactants and products.

    Args:
        smirks: A SMIRKS string in the format "reactants>>products"

    Returns
    -------
        Dictionary with keys:
            - 'reactants': List of RDKit Mol objects
            - 'products': List of RDKit Mol objects
            - 'valid': Boolean indicating if parsing was successful
            - 'error': Error message if parsing failed

    Example:
        >>> result = parse_smirks("[CH3:1][OH:2]>>[CH3:1][O:2][CH3:1]")
        >>> print(f"Reactants: {len(result['reactants'])}")
        Reactants: 1
        >>> print(f"Products: {len(result['products'])}")
        Products: 1
    """
    result = {"reactants": [], "products": [], "valid": False, "error": None}

    try:
        # Split SMIRKS by reaction arrow
        parts = smirks.split(">>")

        # Format: reactants>>products
        reactants_str, products_str = parts[0], parts[1]

        # Parse reactants
        if reactants_str.strip():
            reactant_smiles = reactants_str.split(".")
            for smiles_raw in reactant_smiles:
                smiles = smiles_raw.strip()
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        result["error"] = f"Invalid reactant SMILES: {smiles}"
                        return result
                    result["reactants"].append(mol)
        # Parse products
        if products_str.strip():
            product_smiles = products_str.split(".")
            for smiles_raw in product_smiles:
                smiles = smiles_raw.strip()
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        result["error"] = f"Invalid product SMILES: {smiles}"
                        return result
                    result["products"].append(mol)

        result["valid"] = True

    except Exception as e:
        result["error"] = f"Parsing error: {e!s}"
        return result
    return result
