"""Tests for energy computation: MMFF, OpenMM, ghost cross-interaction.

These verify that force field energies are physically sensible -- the right
sign, the right order of magnitude, and correct behavior under known
perturbations (stretching a bond raises energy, separating molecules
lowers cross-interaction, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from double_ended_ts_prep.force_fields import (
    COULOMB_K,
    _build_molecule_state,
    _build_openmm_simulation,
    _compute_ghost_cross_energy,
    _compute_openmm_energy,
    _extract_nonbonded_params,
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
        c2 = get_molecule_coordinates(mol2) + [20.0, 0.0, 0.0]
        combined = np.vstack([c1, c2])
        e = _compute_openmm_energy(sim, combined)
        assert np.isfinite(e)
        assert abs(e) < 500.0

    def test_overlapping_molecules_huge_energy(self) -> None:
        """Two molecules at the same position should have enormous repulsive energy."""
        mol1 = prepare_molecule_from_smiles("C")
        mol2 = prepare_molecule_from_smiles("C")
        sim, offsets = _build_openmm_simulation([mol1, mol2])

        c1 = get_molecule_coordinates(mol1)
        c2 = get_molecule_coordinates(mol2)
        combined = np.vstack([c1, c2])  # overlapping!
        e = _compute_openmm_energy(sim, combined)
        assert e > 1e4, f"Overlapping energy {e:.1f} should be >> 10000 kcal/mol"


# ── Nonbonded parameter extraction ──────────────────────────────────────


class TestExtractNonbondedParams:
    """Verify we correctly extract charges, sigma, epsilon from OpenMM."""

    @pytest.fixture(scope="class")
    def water_params(self):
        mol = prepare_molecule_from_smiles("O")
        sim, _ = _build_openmm_simulation([mol])
        return _extract_nonbonded_params(sim), mol

    def test_shape_matches_atoms(self, water_params) -> None:
        params, mol = water_params
        assert params.shape == (mol.GetNumAtoms(), 3)

    def test_charges_sum_near_zero(self, water_params) -> None:
        """Total charge of a neutral molecule must be ~0."""
        params, _ = water_params
        total_q = np.sum(params[:, 0])
        assert abs(total_q) < 0.01, f"Total charge = {total_q:.4f} e"

    def test_sigma_positive(self, water_params) -> None:
        params, _ = water_params
        assert np.all(params[:, 1] >= 0.0), "Sigma must be non-negative"

    def test_epsilon_non_negative(self, water_params) -> None:
        params, _ = water_params
        assert np.all(params[:, 2] >= 0.0), "Epsilon must be non-negative"

    def test_sigma_physical_range(self, water_params) -> None:
        """Sigma should be 0-5 Angstroms for light atoms."""
        params, _ = water_params
        assert np.all(params[:, 1] < 5.0)


# ── Ghost cross-interaction energy ──────────────────────────────────────


class TestGhostCrossEnergy:
    """Tests for the soft-core cross-interaction potential.

    This potential uses:
    - Attractive-only LJ: -4*eps*sigma^6 / (alpha*sigma^6 + r^6)
    - Soft-core Coulomb:  COULOMB_K*q_i*q_j / sqrt(alpha*sigma^2 + r^2)

    Key properties:
    - Finite at r=0 (soft-core prevents singularity)
    - Dispersion is always attractive (negative)
    - Decays to zero at large r
    """

    def _make_params(self, q: float, sigma: float, eps: float, n: int = 1):
        """Build (n, 3) nonbonded parameter array."""
        return np.array([[q, sigma, eps]] * n, dtype=np.float64)

    def test_finite_at_zero_distance(self) -> None:
        """Molecules at the same position should give finite energy (soft-core)."""
        coords = np.array([[0.0, 0.0, 0.0]])
        params = self._make_params(0.1, 3.0, 0.1)
        e = _compute_ghost_cross_energy(coords, coords, params, params)
        assert np.isfinite(e)

    def test_dispersion_attractive(self) -> None:
        """Neutral atoms: ghost energy should be purely attractive (negative)."""
        r_coords = np.array([[0.0, 0.0, 0.0]])
        p_coords = np.array([[4.0, 0.0, 0.0]])
        params = self._make_params(0.0, 3.5, 0.1)  # q=0, pure dispersion
        e = _compute_ghost_cross_energy(r_coords, p_coords, params, params)
        assert e < 0.0, f"Dispersion should be attractive, got {e:.4f}"

    def test_energy_decays_with_distance(self) -> None:
        """Ghost energy magnitude decreases as molecules separate."""
        r_coords = np.array([[0.0, 0.0, 0.0]])
        params = self._make_params(0.0, 3.5, 0.1)

        energies = []
        for d in [3.0, 6.0, 12.0, 24.0]:
            p_coords = np.array([[d, 0.0, 0.0]])
            e = _compute_ghost_cross_energy(r_coords, p_coords, params, params)
            energies.append(abs(e))

        # Each subsequent energy should be smaller
        for i in range(len(energies) - 1):
            assert energies[i] > energies[i + 1], (
                f"|E| at shorter distance ({energies[i]:.6f}) should exceed "
                f"|E| at longer distance ({energies[i+1]:.6f})"
            )

    def test_coulomb_like_charges_repel(self) -> None:
        """Two positive charges at moderate distance: Coulomb contribution positive."""
        r_coords = np.array([[0.0, 0.0, 0.0]])
        p_coords = np.array([[5.0, 0.0, 0.0]])
        # Large q, tiny epsilon so dispersion is negligible
        r_params = self._make_params(1.0, 0.01, 1e-10)
        p_params = self._make_params(1.0, 0.01, 1e-10)
        e = _compute_ghost_cross_energy(r_coords, p_coords, r_params, p_params)
        # Coulomb: K*q1*q2 / r_eff > 0 for like charges
        assert e > 0.0, f"Like charges should repel, got {e:.4f}"

    def test_coulomb_opposite_charges_attract(self) -> None:
        r_coords = np.array([[0.0, 0.0, 0.0]])
        p_coords = np.array([[5.0, 0.0, 0.0]])
        r_params = self._make_params(1.0, 0.01, 1e-10)
        p_params = self._make_params(-1.0, 0.01, 1e-10)
        e = _compute_ghost_cross_energy(r_coords, p_coords, r_params, p_params)
        assert e < 0.0, f"Opposite charges should attract, got {e:.4f}"

    def test_alpha_sc_controls_softness(self) -> None:
        """Larger alpha_sc => more damping at short range => smaller magnitude at r=0."""
        coords = np.array([[0.0, 0.0, 0.0]])
        params = self._make_params(0.0, 3.5, 0.1)

        e_soft = _compute_ghost_cross_energy(coords, coords, params, params, alpha_sc=1.0)
        e_hard = _compute_ghost_cross_energy(coords, coords, params, params, alpha_sc=0.1)
        assert abs(e_hard) > abs(e_soft), (
            "Smaller alpha_sc (harder core) should give larger magnitude at r=0"
        )

    def test_lorentz_berthelot_combining(self) -> None:
        """Verify combining rules by checking a known two-atom case analytically."""
        r_coords = np.array([[0.0, 0.0, 0.0]])
        p_coords = np.array([[0.0, 0.0, 0.0]])  # r=0

        sigma_r, eps_r = 3.0, 0.2
        sigma_p, eps_p = 5.0, 0.8
        r_params = self._make_params(0.0, sigma_r, eps_r)
        p_params = self._make_params(0.0, sigma_p, eps_p)

        e = _compute_ghost_cross_energy(r_coords, p_coords, r_params, p_params, alpha_sc=0.5)

        # At r=0: V_disp = -4*eps_ij*sigma_ij^6 / (alpha*sigma_ij^6)
        #        = -4*eps_ij / alpha
        sigma_ij = (sigma_r + sigma_p) / 2
        eps_ij = np.sqrt(eps_r * eps_p)
        expected_disp = -4.0 * eps_ij / 0.5  # = -8*eps_ij
        assert e == pytest.approx(expected_disp, rel=1e-8)
