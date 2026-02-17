"""Tests for energy computation: MMFF and OpenMM.

These verify that force field energies are physically sensible -- the right
sign, the right order of magnitude, and correct behavior under known
perturbations (stretching a bond raises energy, separating molecules
lowers interaction energy, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from double_ended_ts_prep.force_fields import (
    _build_pairwise_params,
    _compute_pairwise_energy,
    compute_mmff_energy,
    get_molecule_coordinates,
    prepare_molecule_from_smiles,
)

# ── MMFF94 energy ───────────────────────────────────────────────────────


class TestMMFFEnergy:
    """Sanity checks for the MMFF94 energy wrapper."""

    def test_energy_is_finite(self, ethanol_mol: Chem.Mol) -> None:
        e = compute_mmff_energy(ethanol_mol)
        assert np.isfinite(e)

    def test_energy_is_reasonable_magnitude(self, ethanol_mol: Chem.Mol) -> None:
        """A relaxed small organic should be < ~100 kcal/mol."""
        e = compute_mmff_energy(ethanol_mol)
        assert abs(e) < 200.0

    def test_distorted_geometry_higher_energy(self) -> None:
        """Stretching a C-C bond should increase energy."""
        mol = prepare_molecule_from_smiles("CC")
        e_relaxed = compute_mmff_energy(mol)

        # Stretch C-C bond by pulling atom 0 away
        coords = get_molecule_coordinates(mol)
        coords[0] += [3.0, 0.0, 0.0]  # 3 A displacement
        from double_ended_ts_prep.force_fields import set_molecule_coordinates

        mol_copy = Chem.RWMol(mol)
        set_molecule_coordinates(mol_copy, coords)
        e_stretched = compute_mmff_energy(mol_copy)
        assert e_stretched > e_relaxed


# ── Pairwise (intermolecular LJ + Coulomb) energy ─────────────────────


class TestPairwiseEnergy:
    """Tests for the pairwise intermolecular energy calculations."""

    def test_single_molecule_zero_energy(self) -> None:
        """A single molecule has no intermolecular interactions → energy = 0."""
        mol = prepare_molecule_from_smiles("CCO")
        params, _ = _build_pairwise_params([mol])
        coords = get_molecule_coordinates(mol)
        e = _compute_pairwise_energy(coords, params)
        assert e == pytest.approx(0.0)

    def test_multi_molecule_finite_energy(self) -> None:
        """Two well-separated molecules should have finite, small energy."""
        mol1 = prepare_molecule_from_smiles("C")
        mol2 = prepare_molecule_from_smiles("O")
        params, offsets = _build_pairwise_params([mol1, mol2])

        assert len(offsets) == 2
        assert offsets[0] == 0
        assert offsets[1] == mol1.GetNumAtoms()

        c1 = get_molecule_coordinates(mol1)
        c2 = get_molecule_coordinates(mol2) + np.array([20.0, 0.0, 0.0])
        combined = np.vstack([c1, c2])
        e = _compute_pairwise_energy(combined, params)
        assert np.isfinite(e)
        assert abs(e) < 500.0

    def test_overlapping_molecules_huge_energy(self) -> None:
        """Two molecules at the same position should have enormous repulsive energy."""
        mol1 = prepare_molecule_from_smiles("C")
        mol2 = prepare_molecule_from_smiles("C")
        params, _offsets = _build_pairwise_params([mol1, mol2])

        c1 = get_molecule_coordinates(mol1)
        c2 = get_molecule_coordinates(mol2)
        combined = np.vstack([c1, c2])  # overlapping!
        e = _compute_pairwise_energy(combined, params)
        assert e > 1e4, f"Overlapping energy {e:.1f} should be >> 10000 kcal/mol"

    def test_energy_decreases_with_separation(self) -> None:
        """Moving molecules apart should decrease repulsive energy."""
        mol1 = prepare_molecule_from_smiles("C")
        mol2 = prepare_molecule_from_smiles("C")
        params, _ = _build_pairwise_params([mol1, mol2])

        c1 = get_molecule_coordinates(mol1)
        c2 = get_molecule_coordinates(mol2)

        # Close together
        combined_close = np.vstack([c1, c2 + np.array([2.0, 0.0, 0.0])])
        e_close = _compute_pairwise_energy(combined_close, params)

        # Far apart
        combined_far = np.vstack([c1, c2 + np.array([20.0, 0.0, 0.0])])
        e_far = _compute_pairwise_energy(combined_far, params)

        assert e_close > e_far
