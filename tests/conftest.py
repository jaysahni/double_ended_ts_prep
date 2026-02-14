"""Fixtures and helpers shared across the test suite.

We define a small set of chemically meaningful reactions as fixtures so that
every test module can reuse them without re-running RXNMapper (expensive).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pytest
from rdkit import Chem

from double_ended_ts_prep.force_fields import (
    _build_molecule_state,
    _build_openmm_simulation,
    _extract_nonbonded_params,
    get_atom_mapping_correspondence,
    get_molecule_coordinates,
    prepare_molecule_from_smiles,
)
from double_ended_ts_prep.labeling import build_smirks, map_smirks, smirks_to_molecules


# ---------------------------------------------------------------------------
# Reaction dataclass for test fixtures
# ---------------------------------------------------------------------------
@dataclass
class ReactionFixture:
    """A fully-prepared reaction ready for optimization tests."""

    name: str
    mapped_smirks: str
    reactants: list[Chem.Mol]
    products: list[Chem.Mol]
    n_mapped_atoms: int


def _make_reaction(
    name: str, reactant_smiles: list[str], product_smiles: list[str]
) -> ReactionFixture:
    """Build a ReactionFixture, catching low-confidence warnings."""
    smirks = build_smirks(reactant_smiles, product_smiles)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mapped = map_smirks(smirks)
    mols = smirks_to_molecules(mapped)
    corr = get_atom_mapping_correspondence(mols["reactants"], mols["products"])
    return ReactionFixture(
        name=name,
        mapped_smirks=mapped,
        reactants=mols["reactants"],
        products=mols["products"],
        n_mapped_atoms=len(corr),
    )


# ---------------------------------------------------------------------------
# Session-scoped fixtures (expensive OpenMM / RXNMapper work done once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def claisen_rxn() -> ReactionFixture:
    """Claisen rearrangement: 1 reactant -> 1 product, all atoms mapped.

    Allyl vinyl ether -> pentenone.  A clean [3,3]-sigmatropic shift with
    no change in molecular formula and no intermolecular issues.
    """
    return _make_reaction("Claisen", ["C=CC(=C)OC=C"], ["C=CCC(=O)C=C"])


@pytest.fixture(scope="session")
def ester_hydrolysis_rxn() -> ReactionFixture:
    """Ester hydrolysis: 2 reactants -> 2 products, multi-molecule system.

    Methyl acetate + water -> acetic acid + methanol
    """
    return _make_reaction("Ester hydrolysis", ["CC(=O)OC", "O"], ["CC(=O)O", "CO"])


@pytest.fixture(scope="session")
def diels_alder_rxn() -> ReactionFixture:
    """Diels-Alder cycloaddition: 2 reactants -> 1 product.

    Butadiene + ethylene -> cyclohexene
    """
    return _make_reaction("Diels-Alder", ["C=CC=C", "C=C"], ["C1CC=CCC1"])


@pytest.fixture(scope="session")
def isocyanate_hydration_rxn() -> ReactionFixture:
    """Isocyanate hydration: 2 reactants -> 1 product.

    Methyl isocyanate + water -> carbamic acid (N-methyl)
    This is the doctest example from optimize_ts_prep.
    """
    return _make_reaction("Isocyanate hydration", ["N=C=O", "O"], ["NC(=O)O"])


@pytest.fixture(scope="session")
def cope_rxn() -> ReactionFixture:
    """Cope rearrangement: 1:1 with undefined E/Z stereochemistry in product.

    Tests that allow_undefined_stereo handling works correctly.
    """
    return _make_reaction("Cope", ["C=CCC(=C)C=C"], ["C=CC=CCC=C"])


@pytest.fixture(scope="session")
def aldol_rxn() -> ReactionFixture:
    """Aldol condensation: 2:1 with undefined chiral center in product.

    Tests that undefined chiral centers are handled correctly.
    """
    return _make_reaction("Aldol", ["CC=O", "CC=O"], ["CC(O)CC=O"])


@pytest.fixture(scope="session")
def sn2_rxn() -> ReactionFixture:
    """SN2 with monatomic ions: CCl + Br⁻ -> CBr + Cl⁻.

    Tests that monatomic ions get charges via fallback (not AM1-BCC).
    """
    return _make_reaction("SN2", ["CCl", "[Br-]"], ["CBr", "[Cl-]"])


@pytest.fixture(scope="session")
def proton_transfer_rxn() -> ReactionFixture:
    """Proton transfer between acetic acid and ammonia.

    Tests charged species: AcOH + NH3 -> AcO⁻ + NH4⁺.
    """
    return _make_reaction(
        "Proton transfer", ["CC(=O)O", "[NH3]"], ["CC(=O)[O-]", "[NH4+]"]
    )


@pytest.fixture(scope="session")
def amide_formation_rxn() -> ReactionFixture:
    """Amide bond formation: 2 reactants -> 2 products.

    Acetic acid + ethylamine -> N-ethylacetamide + water.
    """
    return _make_reaction("Amide formation", ["CC(=O)O", "NCC"], ["CC(=O)NCC", "O"])


# ---------------------------------------------------------------------------
# Simple molecule fixtures (no RXNMapper needed)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ethanol_mol() -> Chem.Mol:
    return prepare_molecule_from_smiles("CCO")


@pytest.fixture(scope="session")
def methane_mol() -> Chem.Mol:
    return prepare_molecule_from_smiles("C")
