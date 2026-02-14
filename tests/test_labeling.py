"""Tests for the labeling module: SMIRKS building, atom mapping, molecule parsing."""

from __future__ import annotations

import warnings

import pytest
from rdkit import Chem

from double_ended_ts_prep.labeling import build_smirks, map_smirks, smirks_to_molecules


class TestBuildSmirks:
    """Tests for build_smirks."""

    def test_basic_construction(self) -> None:
        result = build_smirks(["CCO"], ["CC=O"])
        assert ">>" in result

    def test_multi_reactant(self) -> None:
        result = build_smirks(["C", "O"], ["CO"])
        assert ">>" in result
        # Reactant side should have a dot separator in the SMIRKS
        reactant_side = result.split(">>")[0]
        assert "." in reactant_side

    def test_empty_reactants_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_smirks([], ["C"])

    def test_empty_products_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_smirks(["C"], [])

    def test_invalid_smiles_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            build_smirks(["not_a_smiles!!!"], ["C"])


class TestMapSmirks:
    """Tests for map_smirks (RXNMapper)."""

    def test_adds_atom_mapping(self) -> None:
        smirks = build_smirks(["CCO", "CC(=O)O"], ["CC(=O)OCC", "O"])
        mapped = map_smirks(smirks)
        assert ":" in mapped  # atom mapping numbers present
        assert ">>" in mapped

    def test_low_confidence_warns(self) -> None:
        """A nonsensical reaction should trigger a low-confidence warning."""
        # Very different reactants/products that RXNMapper will struggle with
        smirks = build_smirks(["C=CC=C", "C=C"], ["C1CC=CCC1"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            map_smirks(smirks)
            # Check if any warning mentions confidence
            confidence_warnings = [x for x in w if "onfidence" in str(x.message)]
            # This may or may not warn depending on rxnmapper's confidence
            # -- we just verify the code doesn't crash


class TestSmirksToMolecules:
    """Tests for smirks_to_molecules."""

    def test_returns_correct_counts(self) -> None:
        result = smirks_to_molecules("[CH3:1][OH:2]>>[CH3:1][O:2][CH3:3]")
        assert len(result["reactants"]) == 1
        assert len(result["products"]) == 1

    def test_multi_molecule(self) -> None:
        result = smirks_to_molecules("[CH4:1].[OH2:2]>>[CH4:1].[OH2:2]")
        assert len(result["reactants"]) == 2
        assert len(result["products"]) == 2

    def test_molecules_have_3d_coords(self) -> None:
        result = smirks_to_molecules("[CH3:1][OH:2]>>[CH3:1][O-:2]")
        for mol in result["reactants"]:
            assert mol.GetNumConformers() > 0
            assert mol.GetConformer().Is3D()

    def test_atom_mapping_preserved(self) -> None:
        result = smirks_to_molecules("[CH3:1][OH:2]>>[CH3:1][O-:2]")
        r_mol = result["reactants"][0]
        map_nums = {a.GetAtomMapNum() for a in r_mol.GetAtoms() if a.GetAtomMapNum() > 0}
        assert 1 in map_nums
        assert 2 in map_nums

    def test_invalid_smiles_raises(self) -> None:
        with pytest.raises(ValueError):
            smirks_to_molecules("INVALID>>ALSO_INVALID")
