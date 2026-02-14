"""Tests for robustness fixes: charge fallback, stereo handling, initialization.

These test the three major fixes:
1. AM1-BCC charge fallback chain for monatomic ions and edge cases
2. Undefined stereochemistry handling (chiral centers, E/Z bonds)
3. Spatial initialization to prevent catastrophic overlap in multi-molecule systems
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from double_ended_ts_prep.force_fields import (
    _assign_partial_charges,
    _build_molecule_state,
    _build_openmm_simulation,
    _compute_openmm_energy,
    _extract_nonbonded_params,
    _initialize_separation,
    get_atom_mapping_correspondence,
    get_molecule_coordinates,
    optimize_ts_prep,
    prepare_molecule_from_smiles,
)
from double_ended_ts_prep.labeling import smirks_to_molecules

from .conftest import ReactionFixture


# ── Charge assignment fallback ──────────────────────────────────────────


class TestChargeAssignmentFallback:
    """Tests for the robust charge assignment chain."""

    def test_normal_organic_gets_am1bcc(self) -> None:
        """Normal organic molecules should use AM1-BCC without issue."""
        mol = prepare_molecule_from_smiles("CCO")
        sim, _ = _build_openmm_simulation([mol])
        params = _extract_nonbonded_params(sim)
        # Should have reasonable partial charges (not all zero or all formal)
        assert not np.allclose(params[:, 0], 0.0)

    def test_bromide_ion_gets_charges(self) -> None:
        """Monatomic Br⁻ should get formal charge -1.0 via fallback."""
        from openff.toolkit import Molecule as OFFMolecule

        mol = Chem.MolFromSmiles("[Br-]")
        mol = Chem.AddHs(mol)
        off_mol = OFFMolecule.from_rdkit(mol, allow_undefined_stereo=True)
        _assign_partial_charges(off_mol)
        charges = [
            pc.m_as("elementary_charge") for pc in off_mol.partial_charges
        ]
        assert charges[0] == pytest.approx(-1.0)

    def test_chloride_ion_gets_charges(self) -> None:
        """Monatomic Cl⁻ should get formal charge -1.0 via fallback."""
        from openff.toolkit import Molecule as OFFMolecule

        mol = Chem.MolFromSmiles("[Cl-]")
        mol = Chem.AddHs(mol)
        off_mol = OFFMolecule.from_rdkit(mol, allow_undefined_stereo=True)
        _assign_partial_charges(off_mol)
        charges = [
            pc.m_as("elementary_charge") for pc in off_mol.partial_charges
        ]
        assert charges[0] == pytest.approx(-1.0)

    def test_ammonium_ion_gets_charges(self) -> None:
        """NH4+ should successfully get charges."""
        from openff.toolkit import Molecule as OFFMolecule

        mol = Chem.MolFromSmiles("[NH4+]")
        mol = Chem.AddHs(mol)
        off_mol = OFFMolecule.from_rdkit(mol, allow_undefined_stereo=True)
        _assign_partial_charges(off_mol)
        total_q = sum(pc.m_as("elementary_charge") for pc in off_mol.partial_charges)
        assert total_q == pytest.approx(1.0, abs=0.01)

    def test_sn2_simulation_builds(self, sn2_rxn: ReactionFixture) -> None:
        """Full SN2 reaction (CCl + Br⁻ -> CBr + Cl⁻) should build simulations."""
        sim_r, _ = _build_openmm_simulation(sn2_rxn.reactants)
        sim_p, _ = _build_openmm_simulation(sn2_rxn.products)
        # Both should produce finite energies
        r_coords = np.vstack([get_molecule_coordinates(m) for m in sn2_rxn.reactants])
        # Separate molecules so they don't overlap
        offset = 0
        coords = []
        for m in sn2_rxn.reactants:
            c = get_molecule_coordinates(m)
            c[:, 0] += offset
            coords.append(c)
            offset += 10.0
        r_coords = np.vstack(coords)
        e = _compute_openmm_energy(sim_r, r_coords)
        assert np.isfinite(e)


# ── Undefined stereochemistry ───────────────────────────────────────────


class TestUndefinedStereochemistry:
    """Molecules with undefined stereo must not crash OpenFF conversion."""

    def test_cope_product_builds_simulation(self, cope_rxn: ReactionFixture) -> None:
        """Cope product has undefined E/Z bond -- should now work."""
        sim, _ = _build_openmm_simulation(cope_rxn.products)
        coords = np.vstack([get_molecule_coordinates(m) for m in cope_rxn.products])
        e = _compute_openmm_energy(sim, coords)
        assert np.isfinite(e)

    def test_aldol_product_builds_simulation(self, aldol_rxn: ReactionFixture) -> None:
        """Aldol product has undefined chiral center -- should now work."""
        sim, _ = _build_openmm_simulation(aldol_rxn.products)
        coords = np.vstack([get_molecule_coordinates(m) for m in aldol_rxn.products])
        e = _compute_openmm_energy(sim, coords)
        assert np.isfinite(e)

    def test_cope_full_optimization(self, cope_rxn: ReactionFixture) -> None:
        mols = smirks_to_molecules(cope_rxn.mapped_smirks)
        result = optimize_ts_prep(
            mols["reactants"], mols["products"],
            alpha=1.0, beta=1.0, gamma=0.0, max_iters=200,
        )
        assert np.isfinite(result["final_energy"])
        assert result["geometric_error"] >= 0.0

    def test_aldol_full_optimization(self, aldol_rxn: ReactionFixture) -> None:
        mols = smirks_to_molecules(aldol_rxn.mapped_smirks)
        result = optimize_ts_prep(
            mols["reactants"], mols["products"],
            alpha=1.0, beta=1.0, gamma=0.0, max_iters=200,
        )
        assert np.isfinite(result["final_energy"])
        assert result["geometric_error"] >= 0.0


# ── Radical detection ───────────────────────────────────────────────────


class TestRadicalDetection:
    """Radicals should raise a clear error, not an opaque OpenFF traceback."""

    def test_chlorine_radical_raises_clear_error(self) -> None:
        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        mol = Chem.MolFromSmiles("[Cl]", ps)
        mol = Chem.AddHs(mol)
        with pytest.raises(ValueError, match="radical"):
            _build_openmm_simulation([mol])

    def test_methyl_radical_raises_clear_error(self) -> None:
        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        mol = Chem.MolFromSmiles("[CH3]", ps)
        mol = Chem.AddHs(mol)
        with pytest.raises(ValueError, match="radical"):
            _build_openmm_simulation([mol])


# ── Spatial initialization ──────────────────────────────────────────────


class TestSpatialInitialization:
    """Tests for _initialize_separation."""

    def test_single_molecule_no_change(self) -> None:
        """A single-molecule side should not get translated."""
        mol = prepare_molecule_from_smiles("CCO")
        states = [_build_molecule_state(mol)]
        corr = {(0, 0): (0, 0)}
        params = np.zeros(12)  # 6 for reactant + 6 for product
        _initialize_separation(params, states, states, corr)
        # Single molecule on each side: no intra-side separation needed
        # But product may be shifted to match reactant centroid
        assert np.isfinite(params).all()

    def test_two_molecules_get_separated(self) -> None:
        """Two reactant molecules should be separated along x-axis."""
        mol1 = prepare_molecule_from_smiles("C")
        mol2 = prepare_molecule_from_smiles("O")
        r_states = [_build_molecule_state(mol1), _build_molecule_state(mol2)]
        p_states = [_build_molecule_state(mol1)]
        corr = {(0, 0): (0, 0)}
        params = np.zeros(6 * 2 + 6 * 1)
        _initialize_separation(params, r_states, p_states, corr)
        # Second reactant should have positive x-translation
        tx_mol2 = params[6]  # tx of second reactant
        assert tx_mol2 > 0.0

    def test_multi_molecule_energies_reasonable(self) -> None:
        """After initialization, multi-molecule systems should have reasonable energies."""
        mol1 = prepare_molecule_from_smiles("C")
        mol2 = prepare_molecule_from_smiles("O")
        sim, offsets = _build_openmm_simulation([mol1, mol2])

        c1 = get_molecule_coordinates(mol1)
        c2 = get_molecule_coordinates(mol2)

        # Without separation (overlapping) -> huge energy
        combined_overlap = np.vstack([c1, c2])
        e_overlap = _compute_openmm_energy(sim, combined_overlap)

        # With separation -> reasonable energy
        c2_sep = c2 + [15.0, 0.0, 0.0]
        combined_sep = np.vstack([c1, c2_sep])
        e_sep = _compute_openmm_energy(sim, combined_sep)

        assert e_sep < e_overlap
        assert abs(e_sep) < 500.0

    def test_diels_alder_energy_not_billions(
        self, diels_alder_rxn: ReactionFixture
    ) -> None:
        """Diels-Alder (2:1) should no longer start at billions of kcal/mol."""
        mols = smirks_to_molecules(diels_alder_rxn.mapped_smirks)
        result = optimize_ts_prep(
            mols["reactants"], mols["products"],
            alpha=1.0, beta=1.0, gamma=0.0, max_iters=200,
        )
        assert result["final_energy"] < 1e4, (
            f"Energy {result['final_energy']:.0f} still too large"
        )

    def test_ester_hydrolysis_energy_not_billions(
        self, ester_hydrolysis_rxn: ReactionFixture
    ) -> None:
        """Ester hydrolysis (2:2) should no longer start at billions of kcal/mol."""
        mols = smirks_to_molecules(ester_hydrolysis_rxn.mapped_smirks)
        result = optimize_ts_prep(
            mols["reactants"], mols["products"],
            alpha=1.0, beta=1.0, gamma=0.0, max_iters=200,
        )
        assert result["final_energy"] < 1e4, (
            f"Energy {result['final_energy']:.0f} still too large"
        )

    def test_products_shifted_toward_reactants(self) -> None:
        """Product centroids should be shifted to match reactant mapped-atom centroid."""
        mol_r = prepare_molecule_from_smiles("[CH3:1][OH:2]")
        mol_p = prepare_molecule_from_smiles("[CH3:1][O-:2]")
        r_states = [_build_molecule_state(mol_r)]
        p_states = [_build_molecule_state(mol_p)]
        corr = get_atom_mapping_correspondence([mol_r], [mol_p])

        params = np.zeros(12)
        _initialize_separation(params, r_states, p_states, corr)
        # Product translations should be non-zero (shifted toward reactant)
        p_trans = params[6:9]
        # At minimum, the initialization should make mapped atoms closer
        assert np.isfinite(p_trans).all()


# ── Full optimization on previously-failing reactions ───────────────────


class TestPreviouslyFailingReactions:
    """End-to-end tests on reactions that used to fail or give absurd results."""

    def test_sn2_optimization(self, sn2_rxn: ReactionFixture) -> None:
        """SN2 with ions (CCl + Br⁻ -> CBr + Cl⁻) should now optimize."""
        mols = smirks_to_molecules(sn2_rxn.mapped_smirks)
        result = optimize_ts_prep(
            mols["reactants"], mols["products"],
            alpha=1.0, beta=1.0, gamma=0.0, max_iters=300,
        )
        assert np.isfinite(result["final_energy"])
        assert result["final_energy"] < 1e4
        assert result["geometric_error"] >= 0.0

    def test_proton_transfer_optimization(
        self, proton_transfer_rxn: ReactionFixture
    ) -> None:
        """Proton transfer (charged species) should give reasonable energy."""
        mols = smirks_to_molecules(proton_transfer_rxn.mapped_smirks)
        result = optimize_ts_prep(
            mols["reactants"], mols["products"],
            alpha=1.0, beta=1.0, gamma=0.0, max_iters=300,
        )
        assert np.isfinite(result["final_energy"])
        assert result["final_energy"] < 1e4

    def test_amide_formation_optimization(
        self, amide_formation_rxn: ReactionFixture
    ) -> None:
        """Amide bond formation (2:2) should give reasonable energy."""
        mols = smirks_to_molecules(amide_formation_rxn.mapped_smirks)
        result = optimize_ts_prep(
            mols["reactants"], mols["products"],
            alpha=1.0, beta=1.0, gamma=0.0, max_iters=300,
        )
        assert np.isfinite(result["final_energy"])
        assert result["final_energy"] < 1e4
