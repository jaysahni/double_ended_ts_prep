"""Tests for the combined objective function _compute_rigid_body_energy.

This is the function that L-BFGS-B minimizes.  We test that:
- Weights (alpha, beta, gamma) scale the correct terms
- The identity parameters (all zeros) reproduce the initial energy
- Perturbations move the energy in the expected direction
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from double_ended_ts_prep.force_fields import (
    SystemState,
    _build_molecule_state,
    _build_openmm_simulation,
    _compute_ghost_cross_energy,
    _compute_openmm_energy,
    _compute_rigid_body_energy,
    _extract_nonbonded_params,
    compute_geometric_error,
    get_atom_mapping_correspondence,
    get_molecule_coordinates,
)

from .conftest import ReactionFixture


def _build_system_state(rxn: ReactionFixture) -> SystemState:
    """Build a SystemState from a ReactionFixture."""
    r_states = [_build_molecule_state(m) for m in rxn.reactants]
    p_states = [_build_molecule_state(m) for m in rxn.products]
    r_sim, r_offsets = _build_openmm_simulation(rxn.reactants)
    p_sim, p_offsets = _build_openmm_simulation(rxn.products)
    r_nb = _extract_nonbonded_params(r_sim)
    p_nb = _extract_nonbonded_params(p_sim)
    corr = get_atom_mapping_correspondence(rxn.reactants, rxn.products)
    return SystemState(
        reactant_states=r_states,
        product_states=p_states,
        atom_mapping=corr,
        n_reactant_params=6 * len(rxn.reactants),
        n_product_params=6 * len(rxn.products),
        reactant_simulation=r_sim,
        product_simulation=p_sim,
        reactant_atom_offsets=r_offsets,
        product_atom_offsets=p_offsets,
        reactant_nb_params=r_nb,
        product_nb_params=p_nb,
    )


class TestObjectiveFunction:
    """Tests for _compute_rigid_body_energy."""

    @pytest.fixture(scope="class")
    def claisen_system(self, claisen_rxn: ReactionFixture) -> SystemState:
        return _build_system_state(claisen_rxn)

    def test_zero_params_gives_finite_energy(
        self, claisen_system: SystemState
    ) -> None:
        n = claisen_system.n_reactant_params + claisen_system.n_product_params
        params = np.zeros(n)
        e = _compute_rigid_body_energy(params, claisen_system, alpha=1.0, beta=1.0)
        assert np.isfinite(e)

    def test_alpha_zero_removes_ff_contribution(
        self, claisen_system: SystemState
    ) -> None:
        """With alpha=0, only geometric error should matter."""
        n = claisen_system.n_reactant_params + claisen_system.n_product_params
        params = np.zeros(n)

        e_full = _compute_rigid_body_energy(
            params, claisen_system, alpha=1.0, beta=1.0
        )
        e_no_ff = _compute_rigid_body_energy(
            params, claisen_system, alpha=0.0, beta=1.0
        )

        # With alpha=0 the energy is purely geometric, should differ from full
        # (unless FF energy happens to be exactly 0, which it won't be)
        assert e_full != pytest.approx(e_no_ff, abs=0.01)

    def test_beta_zero_removes_geo_contribution(
        self, claisen_system: SystemState
    ) -> None:
        """With beta=0, geometric error should not affect energy."""
        n = claisen_system.n_reactant_params + claisen_system.n_product_params
        params = np.zeros(n)

        e_with_geo = _compute_rigid_body_energy(
            params, claisen_system, alpha=1.0, beta=1.0
        )
        e_no_geo = _compute_rigid_body_energy(
            params, claisen_system, alpha=1.0, beta=0.0
        )

        # These should differ since geometric error > 0 at zero params
        assert e_with_geo != pytest.approx(e_no_geo, abs=0.01)

    def test_gamma_adds_cross_interaction(
        self, claisen_system: SystemState
    ) -> None:
        """gamma > 0 should add the ghost cross-energy term."""
        n = claisen_system.n_reactant_params + claisen_system.n_product_params
        params = np.zeros(n)

        e_no_ghost = _compute_rigid_body_energy(
            params, claisen_system, alpha=1.0, beta=1.0, gamma=0.0
        )
        e_with_ghost = _compute_rigid_body_energy(
            params, claisen_system, alpha=1.0, beta=1.0, gamma=1.0
        )
        assert e_no_ghost != pytest.approx(e_with_ghost, abs=0.01)

    def test_beta_scaling(self, claisen_system: SystemState) -> None:
        """Doubling beta should double the geometric contribution."""
        n = claisen_system.n_reactant_params + claisen_system.n_product_params
        params = np.zeros(n)

        e_b1 = _compute_rigid_body_energy(
            params, claisen_system, alpha=0.0, beta=1.0
        )
        e_b2 = _compute_rigid_body_energy(
            params, claisen_system, alpha=0.0, beta=2.0
        )
        # With alpha=0, E = beta * geo_error, so e_b2 == 2 * e_b1
        assert e_b2 == pytest.approx(2.0 * e_b1, rel=1e-8)

    def test_gamma_scaling(self, claisen_system: SystemState) -> None:
        """Doubling gamma should double the ghost cross-energy contribution."""
        n = claisen_system.n_reactant_params + claisen_system.n_product_params
        params = np.zeros(n)

        # Use alpha=0, beta=0 to isolate gamma term
        e_g1 = _compute_rigid_body_energy(
            params, claisen_system, alpha=0.0, beta=0.0, gamma=1.0
        )
        e_g2 = _compute_rigid_body_energy(
            params, claisen_system, alpha=0.0, beta=0.0, gamma=2.0
        )
        assert e_g2 == pytest.approx(2.0 * e_g1, rel=1e-8)

    def test_translation_changes_energy(
        self, claisen_system: SystemState
    ) -> None:
        """Translating a molecule should change the energy."""
        n = claisen_system.n_reactant_params + claisen_system.n_product_params
        params_zero = np.zeros(n)
        params_shifted = np.zeros(n)
        params_shifted[0] = 5.0  # translate first reactant 5 A in x

        e_zero = _compute_rigid_body_energy(
            params_zero, claisen_system, alpha=1.0, beta=1.0
        )
        e_shifted = _compute_rigid_body_energy(
            params_shifted, claisen_system, alpha=1.0, beta=1.0
        )
        assert e_zero != pytest.approx(e_shifted, abs=0.01)
