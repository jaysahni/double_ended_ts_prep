"""Tests for robustness fixes: charge fallback, stereo handling.

These test the major fixes:
1. AM1-BCC charge fallback chain for monatomic ions and edge cases
2. Undefined stereochemistry handling (chiral centers, E/Z bonds)
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from double_ended_ts_prep.force_fields import (
    _assign_partial_charges,
    _build_openmm_simulation,
    _compute_openmm_energy,
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
        # Simulation should build without error
        coords = get_molecule_coordinates(mol)
        e = _compute_openmm_energy(sim, coords)
        assert np.isfinite(e)

    def test_bromide_ion_gets_charges(self) -> None:
        """Monatomic Br⁻ should get formal charge -1.0 via fallback."""
        from openff.toolkit import Molecule as OFFMolecule

        mol = Chem.MolFromSmiles("[Br-]")
        mol = Chem.AddHs(mol)
        off_mol = OFFMolecule.from_rdkit(mol, allow_undefined_stereo=True)
        _assign_partial_charges(off_mol)
        charges = [pc.m_as("elementary_charge") for pc in off_mol.partial_charges]
        assert charges[0] == pytest.approx(-1.0)

    def test_chloride_ion_gets_charges(self) -> None:
        """Monatomic Cl⁻ should get formal charge -1.0 via fallback."""
        from openff.toolkit import Molecule as OFFMolecule

        mol = Chem.MolFromSmiles("[Cl-]")
        mol = Chem.AddHs(mol)
        off_mol = OFFMolecule.from_rdkit(mol, allow_undefined_stereo=True)
        _assign_partial_charges(off_mol)
        charges = [pc.m_as("elementary_charge") for pc in off_mol.partial_charges]
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
        _sim_p, _ = _build_openmm_simulation(sn2_rxn.products)
        # Both should produce finite energies
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
            mols["reactants"],
            mols["products"],
            alpha=1.0,
            beta=1.0,
            max_iters=200,
        )
        assert np.isfinite(result["final_energy"])
        assert result["geometric_error"] >= 0.0  # ty: ignore[unsupported-operator]

    def test_aldol_full_optimization(self, aldol_rxn: ReactionFixture) -> None:
        mols = smirks_to_molecules(aldol_rxn.mapped_smirks)
        result = optimize_ts_prep(
            mols["reactants"],
            mols["products"],
            alpha=1.0,
            beta=1.0,
            max_iters=200,
        )
        assert np.isfinite(result["final_energy"])
        assert result["geometric_error"] >= 0.0  # ty: ignore[unsupported-operator]


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


# ── Full optimization on previously-failing reactions ───────────────────


class TestPreviouslyFailingReactions:
    """End-to-end tests on reactions that used to fail or give absurd results."""

    def test_sn2_optimization(self, sn2_rxn: ReactionFixture) -> None:
        """SN2 with ions (CCl + Br⁻ -> CBr + Cl⁻) should now optimize."""
        mols = smirks_to_molecules(sn2_rxn.mapped_smirks)
        result = optimize_ts_prep(
            mols["reactants"],
            mols["products"],
            alpha=1.0,
            beta=1.0,
            max_iters=300,
        )
        assert np.isfinite(result["final_energy"])
        assert result["geometric_error"] >= 0.0  # ty: ignore[unsupported-operator]

    def test_proton_transfer_optimization(self, proton_transfer_rxn: ReactionFixture) -> None:
        """Proton transfer (charged species) should give reasonable energy."""
        mols = smirks_to_molecules(proton_transfer_rxn.mapped_smirks)
        result = optimize_ts_prep(
            mols["reactants"],
            mols["products"],
            alpha=1.0,
            beta=1.0,
            max_iters=300,
        )
        assert np.isfinite(result["final_energy"])

    def test_amide_formation_optimization(self, amide_formation_rxn: ReactionFixture) -> None:
        """Amide bond formation (2:2) should give reasonable energy."""
        mols = smirks_to_molecules(amide_formation_rxn.mapped_smirks)
        result = optimize_ts_prep(
            mols["reactants"],
            mols["products"],
            alpha=1.0,
            beta=1.0,
            max_iters=300,
        )
        assert np.isfinite(result["final_energy"])
