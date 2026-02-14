"""Tests for rigid-body geometry primitives.

These tests validate the mathematical foundations: coordinate extraction,
rigid transforms, and geometric error computation -- independent of any
force field or OpenMM machinery.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from double_ended_ts_prep.force_fields import (
    apply_rigid_transform,
    compute_geometric_error,
    get_atom_mapping_correspondence,
    get_molecule_coordinates,
    prepare_molecule_from_smiles,
    set_molecule_coordinates,
)


# ── Coordinate extraction / setting ─────────────────────────────────────


class TestCoordinateRoundTrip:
    """get_molecule_coordinates and set_molecule_coordinates must round-trip."""

    def test_shape_matches_atom_count(self, ethanol_mol: Chem.Mol) -> None:
        coords = get_molecule_coordinates(ethanol_mol)
        assert coords.shape == (ethanol_mol.GetNumAtoms(), 3)

    def test_round_trip_preserves_coords(self, ethanol_mol: Chem.Mol) -> None:
        original = get_molecule_coordinates(ethanol_mol).copy()
        shifted = original + 7.77
        mol_copy = Chem.RWMol(ethanol_mol)
        set_molecule_coordinates(mol_copy, shifted)
        recovered = get_molecule_coordinates(mol_copy)
        np.testing.assert_allclose(recovered, shifted, atol=1e-10)

    def test_no_conformer_raises(self) -> None:
        mol = Chem.MolFromSmiles("C")
        with pytest.raises(ValueError, match="no conformer"):
            get_molecule_coordinates(mol)

    def test_set_wrong_shape_raises(self, ethanol_mol: Chem.Mol) -> None:
        mol_copy = Chem.RWMol(ethanol_mol)
        wrong = np.zeros((2, 3))
        with pytest.raises(ValueError, match="Expected coords shape"):
            set_molecule_coordinates(mol_copy, wrong)


# ── Rigid-body transform ────────────────────────────────────────────────


class TestApplyRigidTransform:
    """Tests for the Euler-angle rigid body transformation."""

    @pytest.fixture()
    def simple_coords(self) -> np.ndarray:
        """Two-atom system along x-axis."""
        return np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    def test_identity_transform(self, simple_coords: np.ndarray) -> None:
        centroid = np.mean(simple_coords, axis=0)
        result = apply_rigid_transform(
            simple_coords, centroid, np.zeros(3), np.zeros(3)
        )
        np.testing.assert_allclose(result, simple_coords, atol=1e-12)

    def test_pure_translation(self, simple_coords: np.ndarray) -> None:
        centroid = np.mean(simple_coords, axis=0)
        trans = np.array([5.0, -3.0, 1.0])
        result = apply_rigid_transform(simple_coords, centroid, trans, np.zeros(3))
        np.testing.assert_allclose(result, simple_coords + trans, atol=1e-12)

    def test_rotation_preserves_distances(self, simple_coords: np.ndarray) -> None:
        """Rotation is an isometry -- pairwise distances must not change."""
        centroid = np.mean(simple_coords, axis=0)
        rot = np.array([0.7, -1.2, 2.1])  # arbitrary Euler angles
        result = apply_rigid_transform(simple_coords, centroid, np.zeros(3), rot)
        d_before = np.linalg.norm(simple_coords[0] - simple_coords[1])
        d_after = np.linalg.norm(result[0] - result[1])
        np.testing.assert_allclose(d_after, d_before, atol=1e-12)

    def test_90_degree_rotation_z(self) -> None:
        """90-degree rotation about z maps (1,0,0) -> (0,1,0)."""
        coords = np.array([[1.0, 0.0, 0.0]])
        centroid = np.zeros(3)
        rot = np.array([0.0, 0.0, np.pi / 2])
        result = apply_rigid_transform(coords, centroid, np.zeros(3), rot)
        np.testing.assert_allclose(result[0], [0.0, 1.0, 0.0], atol=1e-12)

    def test_rotation_around_nonzero_centroid(self) -> None:
        """Rotation about a shifted centroid should pivot around that point."""
        coords = np.array([[2.0, 0.0, 0.0]])
        centroid = np.array([1.0, 0.0, 0.0])
        rot = np.array([0.0, 0.0, np.pi])  # 180-deg about z
        result = apply_rigid_transform(coords, centroid, np.zeros(3), rot)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_full_rotation_returns_to_start(self, simple_coords: np.ndarray) -> None:
        """A full 2*pi rotation about any axis is the identity."""
        centroid = np.mean(simple_coords, axis=0)
        rot = np.array([2 * np.pi, 0.0, 0.0])
        result = apply_rigid_transform(simple_coords, centroid, np.zeros(3), rot)
        np.testing.assert_allclose(result, simple_coords, atol=1e-12)


# ── Geometric error ─────────────────────────────────────────────────────


class TestComputeGeometricError:
    """Tests for the atom-mapping geometric penalty."""

    def test_identical_coords_zero_error(self) -> None:
        coords = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])]
        corr = {(0, 0): (0, 0), (0, 1): (0, 1)}
        err = compute_geometric_error(coords, coords, corr)
        assert err == pytest.approx(0.0)

    def test_known_squared_distance(self) -> None:
        """Two atoms each displaced by 1 A in z -> sum of squares = 2.0."""
        r = [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
        p = [np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])]
        corr = {(0, 0): (0, 0), (0, 1): (0, 1)}
        err = compute_geometric_error(r, p, corr)
        assert err == pytest.approx(2.0)

    def test_partial_mapping(self) -> None:
        """Only mapped atoms contribute; unmapped atoms are ignored."""
        r = [np.array([[0.0, 0.0, 0.0], [99.0, 99.0, 99.0]])]
        p = [np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])]
        # Only map atom 0 of mol 0
        corr = {(0, 0): (0, 0)}
        err = compute_geometric_error(r, p, corr)
        assert err == pytest.approx(1.0)

    def test_multi_molecule_mapping(self) -> None:
        """Cross-molecule mapping: reactant mol 0, atom 0 -> product mol 1, atom 0."""
        r = [np.array([[0.0, 0.0, 0.0]]), np.array([[5.0, 5.0, 5.0]])]
        p = [np.array([[10.0, 10.0, 10.0]]), np.array([[1.0, 1.0, 1.0]])]
        corr = {(0, 0): (1, 0), (1, 0): (0, 0)}
        # (0,0,0)-(1,1,1) = 3, (5,5,5)-(10,10,10) = 75 => total = 78
        err = compute_geometric_error(r, p, corr)
        assert err == pytest.approx(3.0 + 75.0)

    def test_empty_correspondence_zero_error(self) -> None:
        r = [np.array([[0.0, 0.0, 0.0]])]
        p = [np.array([[99.0, 99.0, 99.0]])]
        err = compute_geometric_error(r, p, {})
        assert err == pytest.approx(0.0)


# ── Atom mapping correspondence ─────────────────────────────────────────


class TestAtomMappingCorrespondence:
    """Tests for get_atom_mapping_correspondence."""

    def _make_mapped_mol(self, smiles: str) -> Chem.Mol:
        ps = Chem.SmilesParserParams()
        ps.removeHs = False
        mol = Chem.MolFromSmiles(smiles, ps)
        mol = Chem.AddHs(mol, addCoords=False)
        from rdkit.Chem import AllChem

        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        return mol

    def test_simple_1to1_mapping(self) -> None:
        r = [self._make_mapped_mol("[CH3:1][OH:2]")]
        p = [self._make_mapped_mol("[CH3:1][O-:2]")]
        corr = get_atom_mapping_correspondence(r, p)
        # Both map nums 1 and 2 should be present
        assert len(corr) == 2

    def test_no_mapping_returns_empty(self) -> None:
        r = [self._make_mapped_mol("[CH4]")]
        p = [self._make_mapped_mol("[CH4]")]
        corr = get_atom_mapping_correspondence(r, p)
        assert corr == {}

    def test_partial_overlap(self) -> None:
        """Product has map nums 1,2,3 but reactant only has 1,2."""
        r = [self._make_mapped_mol("[CH3:1][OH:2]")]
        p = [self._make_mapped_mol("[CH3:1][O:2][CH3:3]")]
        corr = get_atom_mapping_correspondence(r, p)
        assert len(corr) == 2  # Only 1 and 2 overlap
