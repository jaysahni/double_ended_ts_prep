"""Tests for I/O functions: XYZ string generation and file writing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from double_ended_ts_prep.force_fields import (
    mol_to_xyz_string,
    prepare_molecule_from_smiles,
    write_xyz_files,
)


class TestMolToXYZString:
    """Tests for XYZ format string generation."""

    def test_header_has_atom_count(self) -> None:
        mol = prepare_molecule_from_smiles("O")  # water: 3 atoms (O + 2H)
        xyz = mol_to_xyz_string(mol)
        lines = xyz.splitlines()
        assert lines[0].strip() == "3"

    def test_comment_line(self) -> None:
        mol = prepare_molecule_from_smiles("C")
        xyz = mol_to_xyz_string(mol, comment="methane test")
        lines = xyz.splitlines()
        assert "methane test" in lines[1]

    def test_atom_lines_count(self) -> None:
        mol = prepare_molecule_from_smiles("CCO")  # ethanol: 9 atoms
        xyz = mol_to_xyz_string(mol)
        lines = xyz.strip().splitlines()
        n_atoms = int(lines[0])
        # Lines = header + comment + n_atoms atom lines
        assert len(lines) == n_atoms + 2

    def test_atom_symbols_present(self) -> None:
        mol = prepare_molecule_from_smiles("O")
        xyz = mol_to_xyz_string(mol)
        assert " O " in xyz or xyz.count("O") >= 1
        assert " H " in xyz or xyz.count("H") >= 2

    def test_no_conformer_raises(self) -> None:
        mol = Chem.MolFromSmiles("C")
        with pytest.raises(ValueError, match="no conformer"):
            mol_to_xyz_string(mol)


class TestWriteXYZFiles:
    """Tests for writing XYZ files to disk."""

    def test_creates_files(self) -> None:
        r = [prepare_molecule_from_smiles("C")]
        p = [prepare_molecule_from_smiles("O")]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_xyz_files(r, p, tmpdir)
            assert len(result["reactants"]) == 1
            assert len(result["products"]) == 1
            assert result["reactants"][0].exists()
            assert result["products"][0].exists()

    def test_file_contents_valid_xyz(self) -> None:
        mol = prepare_molecule_from_smiles("CCO")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_xyz_files([mol], [mol], tmpdir)
            content = result["reactants"][0].read_text()
            lines = content.strip().splitlines()
            n_atoms = int(lines[0])
            assert n_atoms == mol.GetNumAtoms()

    def test_subdirectory_structure(self) -> None:
        r = [prepare_molecule_from_smiles("C")]
        p = [prepare_molecule_from_smiles("O")]
        with tempfile.TemporaryDirectory() as tmpdir:
            write_xyz_files(r, p, tmpdir)
            assert (Path(tmpdir) / "reactants").is_dir()
            assert (Path(tmpdir) / "products").is_dir()

    def test_multiple_molecules(self) -> None:
        r = [prepare_molecule_from_smiles("C"), prepare_molecule_from_smiles("O")]
        p = [prepare_molecule_from_smiles("CC")]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_xyz_files(r, p, tmpdir)
            assert len(result["reactants"]) == 2
            assert len(result["products"]) == 1
