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
    _build_openmm_simulation,
    _compute_openmm_energy,
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


# ── OpenMM energy ───────────────────────────────────────────────────────


class TestOpenMMEnergy:
    """Tests for the OpenMM Sage force field energy calculations."""

    @pytest.fixture(scope="class")
    def ethanol_sim(self):
        mol = prepare_molecule_from_smiles("CCO")
        sim, offsets = _build_openmm_simulation([mol])
        return sim, offsets, mol

    def test_energy_is_finite(self, ethanol_sim) -> None:
        sim, _, mol = ethanol_sim
        coords = get_molecule_coordinates(mol)
        e = _compute_openmm_energy(sim, coords)
        assert np.isfinite(e)

    def test_equilibrium_energy_reasonable(self, ethanol_sim) -> None:
        """Embedded geometry should give a moderate energy, not billions."""
        sim, _, mol = ethanol_sim
        coords = get_molecule_coordinates(mol)
        e = _compute_openmm_energy(sim, coords)
        assert abs(e) < 500.0, f"Equilibrium energy {e:.1f} kcal/mol is too large"

    def test_distorted_energy_increases(self, ethanol_sim) -> None:
        sim, _, mol = ethanol_sim
        coords = get_molecule_coordinates(mol)
        e_eq = _compute_openmm_energy(sim, coords)

        # Distort: push first atom 5 A away
        distorted = coords.copy()
        distorted[0] += [5.0, 0.0, 0.0]
        e_dist = _compute_openmm_energy(sim, distorted)
        assert e_dist > e_eq

    def test_multi_molecule_simulation(self) -> None:
        """Two molecules in one simulation should have finite energy at equilibrium."""
        mol1 = prepare_molecule_from_smiles("C")
        mol2 = prepare_molecule_from_smiles("O")
        sim, offsets = _build_openmm_simulation([mol1, mol2])

        assert len(offsets) == 2
        assert offsets[0] == 0
        assert offsets[1] == mol1.GetNumAtoms()

        # Build combined coordinate array with molecules well separated
        c1 = get_molecule_coordinates(mol1)
        c2 = get_molecule_coordinates(mol2) + np.array([20.0, 0.0, 0.0])
        combined = np.vstack([c1, c2])
        e = _compute_openmm_energy(sim, combined)
        assert np.isfinite(e)
        assert abs(e) < 500.0

    def test_overlapping_molecules_huge_energy(self) -> None:
        """Two molecules at the same position should have enormous repulsive energy."""
        mol1 = prepare_molecule_from_smiles("C")
        mol2 = prepare_molecule_from_smiles("C")
        sim, _offsets = _build_openmm_simulation([mol1, mol2])

        c1 = get_molecule_coordinates(mol1)
        c2 = get_molecule_coordinates(mol2)
        combined = np.vstack([c1, c2])  # overlapping!
        e = _compute_openmm_energy(sim, combined)
        assert e > 1e4, f"Overlapping energy {e:.1f} should be >> 10000 kcal/mol"
