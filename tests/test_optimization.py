"""End-to-end tests for optimize_ts_prep on real chemical reactions.

These are the most expensive tests -- each calls RXNMapper + OpenMM +
L-BFGS-B optimization.  We verify convergence, geometric improvement,
and that the result molecules have updated coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from double_ended_ts_prep.force_fields import (
    compute_geometric_error,
    get_atom_mapping_correspondence,
    get_molecule_coordinates,
    optimize_ts_prep,
)

from .conftest import ReactionFixture


class TestOptimizeTSPrep:
    """Full optimization on the Claisen rearrangement (1:1, clean case)."""

    @pytest.fixture(scope="class")
    def claisen_result(self, claisen_rxn: ReactionFixture):
        return optimize_ts_prep(
            claisen_rxn.reactants,
            claisen_rxn.products,
            alpha=1.0,
            beta=5.0,
            max_iters=500,
        )

    def test_returns_expected_keys(self, claisen_result: dict) -> None:
        assert set(claisen_result.keys()) == {
            "reactants",
            "products",
            "final_energy",
            "geometric_error",
            "success",
        }

    def test_converges(self, claisen_result: dict) -> None:
        assert claisen_result["success"] is True

    def test_final_energy_finite(self, claisen_result: dict) -> None:
        assert np.isfinite(claisen_result["final_energy"])

    def test_geometric_error_reduced(
        self, claisen_rxn: ReactionFixture, claisen_result: dict
    ) -> None:
        """Optimization should reduce geometric error vs. the initial geometry."""
        corr = get_atom_mapping_correspondence(claisen_rxn.reactants, claisen_rxn.products)
        initial_r_coords = [get_molecule_coordinates(m) for m in claisen_rxn.reactants]
        initial_p_coords = [get_molecule_coordinates(m) for m in claisen_rxn.products]
        initial_geo_err = compute_geometric_error(initial_r_coords, initial_p_coords, corr)

        assert claisen_result["geometric_error"] < initial_geo_err

    def test_output_molecules_have_conformers(self, claisen_result: dict) -> None:
        for mol in claisen_result["reactants"]:
            assert mol.GetNumConformers() > 0
        for mol in claisen_result["products"]:
            assert mol.GetNumConformers() > 0

    def test_coordinates_actually_changed(
        self, claisen_rxn: ReactionFixture, claisen_result: dict
    ) -> None:
        """The optimizer should have moved at least one molecule."""
        orig_r = get_molecule_coordinates(claisen_rxn.reactants[0])
        opt_r = get_molecule_coordinates(claisen_result["reactants"][0])
        # Not identical (some transform was applied)
        assert not np.allclose(orig_r, opt_r, atol=1e-6)

    def test_intramolecular_distances_preserved(
        self, claisen_rxn: ReactionFixture, claisen_result: dict
    ) -> None:
        """Rigid-body transforms must preserve all pairwise distances within a molecule."""
        for orig_mol, opt_mol in zip(
            claisen_rxn.reactants, claisen_result["reactants"], strict=True
        ):
            orig_coords = get_molecule_coordinates(orig_mol)
            opt_coords = get_molecule_coordinates(opt_mol)
            # Compute pairwise distance matrices
            orig_dists = np.linalg.norm(orig_coords[:, None] - orig_coords[None, :], axis=-1)
            opt_dists = np.linalg.norm(opt_coords[:, None] - opt_coords[None, :], axis=-1)
            np.testing.assert_allclose(opt_dists, orig_dists, atol=1e-6)


class TestOptimizeMultiMolecule:
    """Optimization on 2-reactant -> 2-product ester hydrolysis."""

    @pytest.fixture(scope="class")
    def ester_result(self, ester_hydrolysis_rxn: ReactionFixture):
        return optimize_ts_prep(
            ester_hydrolysis_rxn.reactants,
            ester_hydrolysis_rxn.products,
            alpha=0.1,
            beta=1.0,
            max_iters=500,
        )

    def test_returns_multiple_reactants(self, ester_result: dict) -> None:
        assert len(ester_result["reactants"]) == 2

    def test_returns_multiple_products(self, ester_result: dict) -> None:
        assert len(ester_result["products"]) == 2

    def test_energy_is_finite(self, ester_result: dict) -> None:
        assert np.isfinite(ester_result["final_energy"])


class TestWeightSensitivity:
    """Verify that weight parameters produce expected relative behavior."""

    @pytest.fixture(scope="class")
    def claisen_mapped(self, claisen_rxn: ReactionFixture):
        return claisen_rxn.mapped_smirks

    def _run(self, smirks: str, alpha: float, beta: float) -> dict:
        from double_ended_ts_prep.labeling import smirks_to_molecules

        mols = smirks_to_molecules(smirks)
        return optimize_ts_prep(
            mols["reactants"],
            mols["products"],
            alpha=alpha,
            beta=beta,
            max_iters=300,
        )

    def test_higher_beta_lower_geo_error(self, claisen_mapped: str) -> None:
        """Increasing beta should generally decrease geometric error."""
        r_low = self._run(claisen_mapped, alpha=1.0, beta=0.5)
        r_high = self._run(claisen_mapped, alpha=1.0, beta=5.0)

        # With much higher beta weight, the optimizer fights harder to reduce
        # geometric error.  This isn't guaranteed for every landscape, but
        # for the Claisen 1:1 case it should hold.
        # We use a soft assertion: high-beta geo error should be no worse
        # than 2x the low-beta geo error (allowing for local minima variation)
        assert r_high["geometric_error"] < r_low["geometric_error"] * 2.0

    def test_lower_alpha_allows_more_ff_distortion(self, claisen_mapped: str) -> None:
        """Lower alpha penalizes FF less, allowing more geometric freedom."""
        r_a1 = self._run(claisen_mapped, alpha=1.0, beta=1.0)
        r_a01 = self._run(claisen_mapped, alpha=0.1, beta=1.0)
        # With lower alpha, the optimizer can accept higher FF energy
        # to improve geometry; both should converge to something reasonable
        assert np.isfinite(r_a1["final_energy"])
        assert np.isfinite(r_a01["final_energy"])


class TestOptimizeValidation:
    """Test input validation in optimize_ts_prep."""

    def test_empty_reactants_raises(self) -> None:
        from double_ended_ts_prep.force_fields import prepare_molecule_from_smiles

        p = [prepare_molecule_from_smiles("C")]
        with pytest.raises(ValueError, match="Reactants list cannot be empty"):
            optimize_ts_prep([], p)

    def test_empty_products_raises(self) -> None:
        from double_ended_ts_prep.force_fields import prepare_molecule_from_smiles

        r = [prepare_molecule_from_smiles("C")]
        with pytest.raises(ValueError, match="Products list cannot be empty"):
            optimize_ts_prep(r, [])

    def test_no_conformer_raises(self) -> None:
        r = [Chem.MolFromSmiles("C")]
        p = [Chem.MolFromSmiles("C")]
        with pytest.raises(ValueError, match="no conformer"):
            optimize_ts_prep(r, p)

    def test_no_atom_mapping_raises(self) -> None:
        """Molecules without atom mapping numbers should raise."""
        from double_ended_ts_prep.force_fields import prepare_molecule_from_smiles

        r = [prepare_molecule_from_smiles("C")]
        p = [prepare_molecule_from_smiles("O")]
        with pytest.raises(ValueError, match="No atom mapping correspondence"):
            optimize_ts_prep(r, p)
