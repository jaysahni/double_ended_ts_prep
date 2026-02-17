"""Module for force field-based molecular optimization and TS prep.

This module provides methods to optimize molecular systems using the MMFF94
force field, as well as rigid-body optimization for double-ended transition
state search preparation.

The TS prep workflow involves:
1. Taking labeled reactant and product molecules with atom mapping
2. Placing them spatially apart
3. Optimizing rigid-body positions to minimize a combined energy functional
4. The energy balances individual molecule stability with geometric correspondence
"""

from __future__ import annotations

import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from openff.toolkit import ForceField as OFFForceField
from openff.toolkit import Molecule as OFFMolecule
from openff.toolkit import Topology as OFFTopology
from openff.units import unit as off_unit
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Geometry import Point3D
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Conversion factor: 1 Hartree = 627.5094740631 kcal/mol
KCAL_MOL_TO_HARTREE = 1.0 / 627.5094740631


@dataclass
class MoleculeState:
    """State for a single molecule during rigid-body optimization.

    Attributes:
        mol: Original RDKit molecule (not modified during optimization)
        initial_coords: Original (N, 3) coordinates in Angstroms
        centroid: Center of mass for rotation pivot (3,)
        atom_count: Number of atoms in the molecule
    """

    mol: Chem.Mol
    initial_coords: NDArray[np.floating]
    centroid: NDArray[np.floating]
    atom_count: int


@dataclass
class PairwiseParams:
    """Lightweight per-atom force field parameters for pairwise energy evaluation.

    Stores only the data needed for intermolecular Lennard-Jones and Coulomb
    interactions, avoiding the overhead of a full OpenMM simulation.

    Attributes:
        charges: Partial charges in elementary charge units, shape (N,)
        sigma: LJ sigma parameters in Angstroms, shape (N,)
        epsilon: LJ epsilon parameters in kcal/mol, shape (N,)
        mol_boundaries: Starting atom index of each molecule in the combined system
    """

    charges: NDArray[np.floating]
    sigma: NDArray[np.floating]
    epsilon: NDArray[np.floating]
    mol_boundaries: list[int]


@dataclass
class SystemState:
    """Complete state for rigid-body optimization of reactants and products.

    Reactants and products are completely isolated energy systems. Each side
    uses lightweight pairwise parameters for computing intermolecular
    interactions within that side only. The geometric error (via atom mapping)
    is the only coupling between the two sides.

    Attributes:
        reactant_states: List of MoleculeState for each reactant
        product_states: List of MoleculeState for each product
        atom_mapping: Correspondence between (r_mol_idx, atom_idx) -> (p_mol_idx, atom_idx)
        n_reactant_params: Total DOFs for reactant side (6 * num_reactants)
        n_product_params: Total DOFs for product side (6 * num_products)
        reactant_params: Pairwise LJ + charge parameters for reactant system
        product_params: Pairwise LJ + charge parameters for product system
        reactant_atom_offsets: Starting atom index of each reactant in the combined system
        product_atom_offsets: Starting atom index of each product in the combined system
    """

    reactant_states: list[MoleculeState]
    product_states: list[MoleculeState]
    atom_mapping: dict[tuple[int, int], tuple[int, int]]
    n_reactant_params: int
    n_product_params: int
    reactant_params: PairwiseParams
    product_params: PairwiseParams
    reactant_atom_offsets: list[int]
    product_atom_offsets: list[int]


@dataclass
class XYZData:
    """Parsed contents of a multi-molecule XYZ file.

    Attributes:
        n_atoms: Total number of atoms in the file
        metadata: Comment-line metadata (same format as parse_xyz_metadata)
        elements: Element symbols, length n_atoms
        coords: Atomic coordinates in Angstroms, shape (n_atoms, 3)
        charges: Per-atom partial charges from 5th column, or None if absent
    """

    n_atoms: int
    metadata: dict[str, str | list[str]]
    elements: list[str]
    coords: NDArray[np.floating]
    charges: list[float] | None


def prepare_molecule_from_smiles(smiles: str) -> Chem.Mol:
    """Prepare a single molecule from SMILES with 3D coordinates.

    Converts a SMILES string to an RDKit molecule with explicit hydrogens
    and generates initial 3D coordinates using distance geometry embedding.
    Preserves any existing atom mapping labels and explicit hydrogens.

    Args:
        smiles: A valid SMILES string representing the molecule

    Returns:
        RDKit Mol object with explicit hydrogens and 3D coordinates

    Raises:
        ValueError: If the SMILES is invalid or embedding fails

    Examples:
        >>> mol = prepare_molecule_from_smiles("CCO")
        >>> mol.GetNumAtoms()
        9
        >>> mol.GetConformer().Is3D()
        True
        >>> mol = prepare_molecule_from_smiles("[CH3:1][OH:2]")
        >>> mol.GetAtomWithIdx(0).GetAtomMapNum()
        1
    """
    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    mol = Chem.MolFromSmiles(smiles, ps)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol_with_h = Chem.AddHs(mol, addCoords=False)

    embed_result = AllChem.EmbedMolecule(  # type: ignore[attr-defined]
        mol_with_h,
        AllChem.ETKDGv3(),  # type: ignore[attr-defined]
    )
    if embed_result == -1:
        raise ValueError(f"Failed to generate 3D coordinates for: {smiles}")

    return mol_with_h


# =============================================================================
# Rigid-Body TS Prep Functions
# =============================================================================


def get_molecule_coordinates(mol: Chem.Mol) -> NDArray[np.floating]:
    """Extract atomic coordinates from a molecule as a NumPy array.

    Args:
        mol: RDKit Mol object with a conformer

    Returns:
        Coordinates as (N, 3) NumPy array where N is atom count, in Angstroms

    Raises:
        ValueError: If molecule has no conformer

    Examples:
        >>> mol = prepare_molecule_from_smiles("C")
        >>> coords = get_molecule_coordinates(mol)
        >>> coords.shape[1]
        3
        >>> coords.shape[0] == mol.GetNumAtoms()
        True
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer")

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    coords = np.zeros((n_atoms, 3), dtype=np.float64)

    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]

    return coords


def set_molecule_coordinates(mol: Chem.Mol, coords: NDArray[np.floating]) -> None:
    """Update molecule conformer with new coordinates.

    Args:
        mol: RDKit Mol object with a conformer
        coords: (N, 3) array of new atomic positions in Angstroms

    Raises:
        ValueError: If molecule has no conformer or coord shape doesn't match

    Examples:
        >>> mol = prepare_molecule_from_smiles("C")
        >>> original = get_molecule_coordinates(mol).copy()
        >>> new_coords = original + 5.0
        >>> set_molecule_coordinates(mol, new_coords)
        >>> updated = get_molecule_coordinates(mol)
        >>> np.allclose(updated, original + 5.0)
        True
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer")

    n_atoms = mol.GetNumAtoms()
    if coords.shape != (n_atoms, 3):
        raise ValueError(f"Expected coords shape ({n_atoms}, 3), got {coords.shape}")

    conf = mol.GetConformer()
    for i in range(n_atoms):
        conf.SetAtomPosition(
            i, Point3D(float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
        )


def apply_rigid_transform(
    coords: NDArray[np.floating],
    centroid: NDArray[np.floating],
    translation: NDArray[np.floating],
    rotation_angles: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply rigid body transformation to coordinates.

    Rotation is applied around the molecule centroid using Euler angles
    (XYZ convention), then translation is applied.

    Args:
        coords: (N, 3) original coordinates
        centroid: (3,) rotation center (usually molecule center of mass)
        translation: (3,) translation vector [tx, ty, tz] in Angstroms
        rotation_angles: (3,) Euler angles [rx, ry, rz] in radians

    Returns:
        (N, 3) transformed coordinates

    Examples:
        >>> coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> centroid = np.array([0.5, 0.0, 0.0])
        >>> trans = np.array([1.0, 0.0, 0.0])
        >>> rot = np.array([0.0, 0.0, 0.0])
        >>> new_coords = apply_rigid_transform(coords, centroid, trans, rot)
        >>> np.allclose(new_coords[:, 0], coords[:, 0] + 1.0)
        True
    """
    # Center coordinates at centroid
    centered = coords - centroid

    # Apply rotation using scipy's Rotation class
    R = Rotation.from_euler("xyz", rotation_angles)
    rotated = R.apply(centered)

    # Translate back and apply translation offset
    return rotated + centroid + translation


def get_atom_mapping_correspondence(
    reactant_mols: list[Chem.Mol],
    product_mols: list[Chem.Mol],
) -> dict[tuple[int, int], tuple[int, int]]:
    """Build correspondence between reactant and product atoms using atom mapping.

    Uses atom mapping numbers (from SMIRKS/mapped SMILES) to identify
    corresponding atoms between reactants and products.

    Args:
        reactant_mols: List of reactant RDKit Mol objects with atom mapping
        product_mols: List of product RDKit Mol objects with atom mapping

    Returns:
        Dictionary mapping (reactant_mol_idx, atom_idx) -> (product_mol_idx, atom_idx).
        Only includes atoms that have valid mappings on both sides.

    Note:
        Atoms are matched by their atom mapping number (GetAtomMapNum()).
        Unmapped atoms (map num 0) are not included in correspondence.

    Examples:
        >>> from double_ended_ts_prep.labeling import smirks_to_molecules
        >>> mols = smirks_to_molecules("[CH3:1][OH:2]>>[CH3:1][O-:2]")
        >>> corr = get_atom_mapping_correspondence(mols['reactants'], mols['products'])
        >>> len(corr) > 0
        True
    """
    # Build reactant side lookup: map_num -> (mol_idx, atom_idx)
    reactant_map: dict[int, tuple[int, int]] = {}
    for mol_idx, mol in enumerate(reactant_mols):
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                reactant_map[map_num] = (mol_idx, atom.GetIdx())

    # Build product side lookup
    product_map: dict[int, tuple[int, int]] = {}
    for mol_idx, mol in enumerate(product_mols):
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                product_map[map_num] = (mol_idx, atom.GetIdx())

    # Find common mappings
    correspondence: dict[tuple[int, int], tuple[int, int]] = {}
    for map_num, r_loc in reactant_map.items():
        if map_num in product_map:
            correspondence[r_loc] = product_map[map_num]

    return correspondence


def compute_mmff_energy(mol: Chem.Mol) -> float:
    """Compute MMFF94 energy without minimization.

    Args:
        mol: RDKit Mol with conformer

    Returns:
        Energy in kcal/mol

    Raises:
        ValueError: If MMFF94 setup fails

    Examples:
        >>> mol = prepare_molecule_from_smiles("CCO")
        >>> energy = compute_mmff_energy(mol)
        >>> isinstance(energy, float)
        True
    """
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)  # type: ignore[attr-defined]
    if mmff_props is None:
        raise ValueError("Failed to get MMFF94 properties")

    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)  # type: ignore[attr-defined]
    if ff is None:
        raise ValueError("Failed to construct MMFF94 force field")

    return ff.CalcEnergy()


def compute_geometric_error(
    reactant_coords_list: list[NDArray[np.floating]],
    product_coords_list: list[NDArray[np.floating]],
    correspondence: dict[tuple[int, int], tuple[int, int]],
) -> float:
    """Compute sum of squared distances between mapped atom pairs.

    Args:
        reactant_coords_list: List of (N_i, 3) arrays for each reactant
        product_coords_list: List of (N_j, 3) arrays for each product
        correspondence: Mapping from (r_mol, r_atom) -> (p_mol, p_atom)

    Returns:
        Sum of squared Euclidean distances between corresponding atoms (Angstrom^2)

    Note:
        Using sum of squared distances (not RMSD) provides better gradient behavior.
        The geometric penalty is: sum_i ||r_i - p_i||^2

    Examples:
        >>> r_coords = [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
        >>> p_coords = [np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])]
        >>> corr = {(0, 0): (0, 0), (0, 1): (0, 1)}
        >>> err = compute_geometric_error(r_coords, p_coords, corr)
        >>> bool(np.isclose(err, 2.0))  # Two atoms each 1 Angstrom apart: 1^2 + 1^2 = 2
        True
    """
    total_error = 0.0
    for (r_mol_idx, r_atom_idx), (p_mol_idx, p_atom_idx) in correspondence.items():
        r_pos = reactant_coords_list[r_mol_idx][r_atom_idx]
        p_pos = product_coords_list[p_mol_idx][p_atom_idx]
        total_error += float(np.sum((r_pos - p_pos) ** 2))
    return total_error


def _assign_partial_charges(off_mol: OFFMolecule) -> None:
    """Assign partial charges with a robust fallback chain.

    Tries AM1-BCC first (gold standard for organics), then falls back to
    Gasteiger (fast, reasonable for most organics), and finally to formal
    charges (exact for monatomic ions, crude but finite for everything else).

    Args:
        off_mol: OpenFF Molecule to assign charges to (modified in place)
    """
    smiles = off_mol.to_smiles()
    methods = ["am1bcc", "gasteiger", "formal_charge"]
    for method in methods:
        try:
            off_mol.assign_partial_charges(partial_charge_method=method)
            if method != "am1bcc":
                logger.info("Used %s charges for %s (AM1-BCC unavailable)", method, smiles)
            return
        except Exception:
            continue
    raise ValueError(f"All charge methods ({', '.join(methods)}) failed for molecule: {smiles}")


@functools.lru_cache(maxsize=1)
def _get_forcefield() -> OFFForceField:
    """Load and cache the OpenFF force field (avoids re-parsing XML on each call)."""
    return OFFForceField("openff_unconstrained-2.2.1.offxml")


def _build_pairwise_params(
    mols: list[Chem.Mol],
    preset_charges: list[list[float]] | None = None,
) -> tuple[PairwiseParams, list[int]]:
    """Extract per-atom LJ and charge parameters for pairwise energy evaluation.

    Since the optimization treats molecules as rigid bodies, only intermolecular
    vdW (Lennard-Jones) and Coulomb interactions change. This function extracts
    just the parameters needed for those calculations, skipping the expensive
    Interchange and OpenMM simulation construction.

    Args:
        mols: List of RDKit Mol objects with conformers.
        preset_charges: Optional list of per-molecule charge lists. When
            provided, these charges are used directly (skipping Gasteiger).
            Each inner list has length equal to the molecule's atom count,
            in elementary charge units.  When ``None``, Gasteiger charges
            are computed.

    Returns:
        Tuple of (PairwiseParams, list of atom offsets for each molecule).

    Raises:
        ValueError: If a molecule contains radicals or force field assignment
            fails.
    """
    forcefield = _get_forcefield()
    off_mols: list[OFFMolecule] = []
    atom_offsets: list[int] = []
    running_offset = 0

    for i, mol in enumerate(mols):
        smiles = Chem.MolToSmiles(mol)
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                raise ValueError(
                    f"Molecule '{smiles}' contains radical electrons on atom "
                    f"{atom.GetSymbol()} (idx {atom.GetIdx()}). "
                    f"OpenFF does not support radical species. Consider using a "
                    f"closed-shell representation or a different force field."
                )

        atom_offsets.append(running_offset)
        off_mol = OFFMolecule.from_rdkit(mol, allow_undefined_stereo=True)

        if preset_charges is not None and preset_charges[i] is not None:
            charges_array = np.array(preset_charges[i]) * off_unit.elementary_charge
            off_mol.partial_charges = charges_array
        else:
            off_mol.assign_partial_charges(partial_charge_method="gasteiger")

        off_mols.append(off_mol)
        running_offset += mol.GetNumAtoms()

    total_atoms = running_offset

    # Extract charges from OpenFF molecules
    all_charges = np.zeros(total_atoms, dtype=np.float64)
    for i, off_mol in enumerate(off_mols):
        offset = atom_offsets[i]
        n = off_mol.n_atoms
        all_charges[offset : offset + n] = off_mol.partial_charges.m_as(off_unit.elementary_charge)

    # Extract vdW parameters via label_molecules (much cheaper than Interchange)
    topology = OFFTopology.from_molecules(off_mols)
    labels = forcefield.label_molecules(topology)
    all_sigma = np.zeros(total_atoms, dtype=np.float64)
    all_epsilon = np.zeros(total_atoms, dtype=np.float64)

    global_atom_idx = 0
    for mol_labels in labels:
        vdw_dict = mol_labels["vdW"]
        for atom_indices, parameter in vdw_dict.items():
            idx = global_atom_idx + atom_indices[0]
            all_sigma[idx] = parameter.sigma.m_as(off_unit.angstrom)
            all_epsilon[idx] = parameter.epsilon.m_as(off_unit.kilocalorie_per_mole)
        global_atom_idx += len(vdw_dict)

    params = PairwiseParams(
        charges=all_charges,
        sigma=all_sigma,
        epsilon=all_epsilon,
        mol_boundaries=atom_offsets,
    )
    return params, atom_offsets


# Coulomb constant for kcal/mol when charges are in e and distance in Angstroms
_COULOMB_CONST_KCAL_A = 332.064


def _compute_pairwise_energy(
    coords: NDArray[np.floating],
    params: PairwiseParams,
) -> float:
    """Compute intermolecular LJ + Coulomb energy using vectorized NumPy.

    Only evaluates pairwise interactions between atoms in *different* molecules.
    Intramolecular terms are constant for rigid bodies and omitted.

    Args:
        coords: (N, 3) coordinates in Angstroms
        params: Pre-computed per-atom LJ and charge parameters

    Returns:
        Energy in kcal/mol
    """
    n = len(coords)
    if n == 0:
        return 0.0

    boundaries = params.mol_boundaries
    n_mols = len(boundaries)
    if n_mols <= 1:
        return 0.0  # Single molecule: no intermolecular interactions

    # Build molecule-index array for each atom
    mol_ids = np.zeros(n, dtype=np.int32)
    for m in range(n_mols - 1):
        mol_ids[boundaries[m] : boundaries[m + 1]] = m
    mol_ids[boundaries[-1] :] = n_mols - 1

    # Pairwise distance matrix (upper triangle only)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff**2, axis=2))  # (N, N)

    # Mask: only intermolecular pairs in upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    inter_mask = mol_ids[i_idx] != mol_ids[j_idx]
    i_inter = i_idx[inter_mask]
    j_inter = j_idx[inter_mask]
    r = dist[i_inter, j_inter]

    # Lorentz-Berthelot combining rules
    sig_ij = (params.sigma[i_inter] + params.sigma[j_inter]) / 2.0
    eps_ij = np.sqrt(params.epsilon[i_inter] * params.epsilon[j_inter])

    # Lennard-Jones 12-6
    sr6 = (sig_ij / r) ** 6
    e_lj = np.sum(4.0 * eps_ij * (sr6**2 - sr6))

    # Coulomb
    e_coul = np.sum(_COULOMB_CONST_KCAL_A * params.charges[i_inter] * params.charges[j_inter] / r)

    return float(e_lj + e_coul)


def _compute_rigid_body_energy(
    params: NDArray[np.floating],
    system_state: SystemState,
    alpha: float,
    beta: float = 1.0,
) -> float:
    """Compute total energy for scipy optimizer using separated reactant/product systems.

    Energy = alpha * (E_reactants + E_products) + beta * geometric_error

    Reactants and products are completely isolated energy systems:
    - E_reactants: Intermolecular LJ + Coulomb among reactant molecules
    - E_products: Intermolecular LJ + Coulomb among product molecules
    - geometric_error via atom mapping couples the two sides geometrically

    Args:
        params: 1D array of all rigid body parameters
        system_state: SystemState with pre-built pairwise parameters
        alpha: Weight for force field energy term (kcal/mol)
        beta: Weight for geometric error term (kcal/mol per Angstrom^2)

    Returns:
        Total energy in kcal/mol
    """
    # Unpack parameters for each side
    n_r = system_state.n_reactant_params
    reactant_params = params[:n_r].reshape(-1, 6)
    product_params = params[n_r:].reshape(-1, 6)

    # === REACTANT SYSTEM (isolated) ===
    reactant_coords_list: list[NDArray[np.floating]] = []
    total_r_atoms = sum(s.atom_count for s in system_state.reactant_states)
    combined_reactant_coords = np.zeros((total_r_atoms, 3), dtype=np.float64)

    for i, state in enumerate(system_state.reactant_states):
        trans = reactant_params[i, :3]
        rot = reactant_params[i, 3:]
        new_coords = apply_rigid_transform(state.initial_coords, state.centroid, trans, rot)
        reactant_coords_list.append(new_coords)

        offset = system_state.reactant_atom_offsets[i]
        combined_reactant_coords[offset : offset + state.atom_count] = new_coords

    reactant_energy = _compute_pairwise_energy(
        combined_reactant_coords, system_state.reactant_params
    )

    # === PRODUCT SYSTEM (isolated) ===
    product_coords_list: list[NDArray[np.floating]] = []
    total_p_atoms = sum(s.atom_count for s in system_state.product_states)
    combined_product_coords = np.zeros((total_p_atoms, 3), dtype=np.float64)

    for i, state in enumerate(system_state.product_states):
        trans = product_params[i, :3]
        rot = product_params[i, 3:]
        new_coords = apply_rigid_transform(state.initial_coords, state.centroid, trans, rot)
        product_coords_list.append(new_coords)

        offset = system_state.product_atom_offsets[i]
        combined_product_coords[offset : offset + state.atom_count] = new_coords

    product_energy = _compute_pairwise_energy(combined_product_coords, system_state.product_params)

    # Energy from isolated systems
    ff_energy = alpha * (reactant_energy + product_energy)

    # Geometric error: coupling term between reactants and products
    geo_error = compute_geometric_error(
        reactant_coords_list, product_coords_list, system_state.atom_mapping
    )

    return ff_energy + beta * geo_error


def _build_molecule_state(mol: Chem.Mol) -> MoleculeState:
    """Build MoleculeState from an RDKit molecule.

    Args:
        mol: RDKit Mol with conformer

    Returns:
        MoleculeState with coordinates and centroid computed
    """
    coords = get_molecule_coordinates(mol)
    centroid = np.mean(coords, axis=0)
    return MoleculeState(
        mol=mol,
        initial_coords=coords,
        centroid=centroid,
        atom_count=mol.GetNumAtoms(),
    )


def optimize_ts_prep(
    reactants: list[Chem.Mol],
    products: list[Chem.Mol],
    alpha: float = 1.0,
    beta: float = 5.0,
    max_iters: int = 500,
    ftol: float = 1e-12,
    reactant_charges: list[list[float]] | None = None,
    product_charges: list[list[float]] | None = None,
) -> dict[str, list[Chem.Mol] | float | bool]:
    """Optimize reactant/product geometries for transition state search.

    Performs rigid-body optimization to minimize:
    E = alpha * (E_reactants + E_products) + beta * geometric_error

    Each molecule is treated as a rigid body with 6 degrees of freedom
    (3 translations + 3 rotations). The geometric error penalizes
    distances between corresponding atoms (matched by atom mapping numbers).

    Args:
        reactants: List of reactant RDKit Mol objects with atom mapping and 3D coords
        products: List of product RDKit Mol objects with atom mapping and 3D coords
        alpha: Weight for force field energy term (default 1.0 kcal/mol)
        beta: Weight for geometric error term (default 1.0 kcal/mol per Angstrom^2)
        max_iters: Maximum L-BFGS iterations
        ftol: Relative function value tolerance for convergence (default 1e-12)
        reactant_charges: Optional per-molecule charge lists for reactants.
            Each inner list has per-atom partial charges in elementary charge
            units.  When provided, skips Gasteiger computation for reactants.
        product_charges: Same as *reactant_charges* but for products.

    Returns:
        Dictionary with:
            'reactants': List of optimized reactant Mol objects
            'products': List of optimized product Mol objects
            'final_energy': Final total energy (kcal/mol)
            'geometric_error': Final geometric error (Angstrom^2)
            'success': Whether optimization converged

    Raises:
        ValueError: If molecules lack conformers or no atom mapping overlap exists

    Examples:
        >>> from double_ended_ts_prep.labeling import smirks_to_molecules, map_smirks, build_smirks
        >>> smirks = build_smirks(["N=C=O", "O"], ["NC(=O)O"])
        >>> mapped = map_smirks(smirks)
        >>> mols = smirks_to_molecules(mapped)
        >>> result = optimize_ts_prep(mols['reactants'], mols['products'], alpha=1.0, beta=1.0)
        >>> import numpy as np
        >>> bool(np.isfinite(result['final_energy']))
        True
    """
    # Validate inputs
    if not reactants:
        raise ValueError("Reactants list cannot be empty")
    if not products:
        raise ValueError("Products list cannot be empty")

    for i, mol in enumerate(reactants):
        if mol.GetNumConformers() == 0:
            raise ValueError(f"Reactant {i} has no conformer")
    for i, mol in enumerate(products):
        if mol.GetNumConformers() == 0:
            raise ValueError(f"Product {i} has no conformer")

    # Build atom mapping correspondence
    correspondence = get_atom_mapping_correspondence(reactants, products)
    if not correspondence:
        raise ValueError("No atom mapping correspondence found between reactants and products")

    # Build molecule states
    reactant_states = [_build_molecule_state(mol) for mol in reactants]
    product_states = [_build_molecule_state(mol) for mol in products]

    # Build pairwise parameters for each side in parallel (done ONCE, reused in optimization loop)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_r = pool.submit(_build_pairwise_params, reactants, preset_charges=reactant_charges)
        fut_p = pool.submit(_build_pairwise_params, products, preset_charges=product_charges)
        reactant_pp, reactant_atom_offsets = fut_r.result()
        product_pp, product_atom_offsets = fut_p.result()

    # Create system state
    n_reactant_params = 6 * len(reactants)
    n_product_params = 6 * len(products)
    system_state = SystemState(
        reactant_states=reactant_states,
        product_states=product_states,
        atom_mapping=correspondence,
        n_reactant_params=n_reactant_params,
        n_product_params=n_product_params,
        reactant_params=reactant_pp,
        product_params=product_pp,
        reactant_atom_offsets=reactant_atom_offsets,
        product_atom_offsets=product_atom_offsets,
    )

    # Initialize parameters
    initial_params = np.zeros(n_reactant_params + n_product_params, dtype=np.float64)

    # Set bounds for rotation angles to [-pi, pi]
    bounds: list[tuple[float | None, float | None]] = []
    for _ in range(len(reactants) + len(products)):
        # Translation bounds: None (unbounded)
        bounds.extend([(None, None), (None, None), (None, None)])
        # Rotation bounds: [-pi, pi]
        bounds.extend([(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)])

    # Run L-BFGS-B optimization
    result = minimize(
        _compute_rigid_body_energy,
        initial_params,
        args=(system_state, alpha, beta),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iters, "ftol": ftol},
    )

    # Extract final parameters
    final_params = result.x
    reactant_params = final_params[:n_reactant_params].reshape(-1, 6)
    product_params = final_params[n_reactant_params:].reshape(-1, 6)

    # Apply final transforms and create output molecules
    optimized_reactants: list[Chem.Mol] = []
    reactant_coords_list: list[NDArray[np.floating]] = []
    for i, state in enumerate(reactant_states):
        trans = reactant_params[i, :3]
        rot = reactant_params[i, 3:]
        new_coords = apply_rigid_transform(state.initial_coords, state.centroid, trans, rot)
        reactant_coords_list.append(new_coords)

        mol_copy = Chem.RWMol(state.mol)
        set_molecule_coordinates(mol_copy, new_coords)
        optimized_reactants.append(mol_copy.GetMol())

    optimized_products: list[Chem.Mol] = []
    product_coords_list: list[NDArray[np.floating]] = []
    for i, state in enumerate(product_states):
        trans = product_params[i, :3]
        rot = product_params[i, 3:]
        new_coords = apply_rigid_transform(state.initial_coords, state.centroid, trans, rot)
        product_coords_list.append(new_coords)

        mol_copy = Chem.RWMol(state.mol)
        set_molecule_coordinates(mol_copy, new_coords)
        optimized_products.append(mol_copy.GetMol())

    # Compute final geometric error
    final_geo_error = compute_geometric_error(
        reactant_coords_list, product_coords_list, correspondence
    )

    return {
        "reactants": optimized_reactants,
        "products": optimized_products,
        "final_energy": float(result.fun),
        "geometric_error": final_geo_error,
        "success": result.success,
    }


def parse_xyz_metadata(path: str | Path) -> list[str]:
    r"""Parse an XYZ file's comment line and return metadata.

    Reads the second line of an XYZ file, which is expected to contain
    semicolon-delimited key-value pairs (e.g., ``charge: 0; smiles: CCO.O;``).

    The ``smiles`` field is split on ``.`` and returned as a list of individual
    SMILES strings.

    Args:
        path: Path to the XYZ file

    Returns:
        List of SMILES as a ``list[str]``.

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the comment line cannot be parsed or lacks a smiles field

    Examples:
        >>> import tempfile, pathlib
        >>> p = pathlib.Path(tempfile.mktemp(suffix=".xyz"))
        >>> _ = p.write_text("1\nsmiles: CCO;\nC 0.0 0.0 0.0\n")
        >>> meta = parse_xyz_metadata(p)
        >>> meta
        ['CCO']
        >>> p.unlink()
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 2:  # noqa: PLR2004
        raise ValueError(f"XYZ file {path} has fewer than 2 lines")

    comment = lines[1]
    metadata: dict[str, str] = {}
    for raw_token in comment.split(";"):
        stripped = raw_token.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", maxsplit=1)
        key = key.strip().lower()
        value = value.strip()
        if key == "smiles":
            metadata[key] = value
        else:
            metadata[key] = value

    if "smiles" not in metadata:
        raise ValueError(f"XYZ file {path} comment line has no 'smiles' field")

    return metadata["smiles"].split(".")


def parse_xyz_file(path: str | Path) -> XYZData:
    """Parse a complete XYZ file including coordinates and optional charges.

    Reads atom count (line 0), metadata from comment line (line 1), and all
    coordinate lines.  If every atom line has a 5th column, it is interpreted
    as a per-atom partial charge.

    Args:
        path: Path to the XYZ file.

    Returns:
        XYZData with elements, coordinates, and optional charges.

    Raises:
        ValueError: If the file format is invalid or atom count mismatches.
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    n_atoms = int(lines[0].strip())
    if len(lines) < 2 + n_atoms:
        raise ValueError(
            f"XYZ file {path} claims {n_atoms} atoms but has only {len(lines) - 2} coordinate lines"
        )

    # Parse comment line for metadata (reuse logic from parse_xyz_metadata)
    comment = lines[1]
    metadata: dict[str, str | list[str]] = {}
    for raw_token in comment.split(";"):
        stripped = raw_token.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", maxsplit=1)
        key = key.strip().lower()
        value_str = value.strip()
        if key == "smiles":
            metadata[key] = value_str.split(".")
        else:
            metadata[key] = value_str

    if "smiles" not in metadata:
        raise ValueError(f"XYZ file {path} comment line has no 'smiles' field")

    elements: list[str] = []
    coords_list: list[list[float]] = []
    charges: list[float] = []
    has_charges: bool | None = None

    for i in range(n_atoms):
        tokens = lines[2 + i].split()
        if len(tokens) < 4:  # noqa: PLR2004
            raise ValueError(f"Line {2 + i} has fewer than 4 columns in {path}")

        elements.append(tokens[0])
        coords_list.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])

        line_has_charge = len(tokens) >= 5  # noqa: PLR2004
        if has_charges is None:
            has_charges = line_has_charge
        elif has_charges != line_has_charge:
            raise ValueError(
                f"Inconsistent column count in {path}: line {2 + i} "
                f"{'has' if line_has_charge else 'lacks'} a 5th column"
            )

        if line_has_charge:
            charges.append(float(tokens[4]))

    return XYZData(
        n_atoms=n_atoms,
        metadata=metadata,
        elements=elements,
        coords=np.array(coords_list, dtype=np.float64),
        charges=charges if has_charges else None,
    )


def split_xyz_by_molecules(
    xyz_data: XYZData,
    smiles_list: list[str],
) -> list[dict]:
    """Split concatenated XYZ data into per-molecule chunks.

    Uses the atom count (with explicit H) from each SMILES to determine
    molecule boundaries in the concatenated coordinate block.

    Args:
        xyz_data: Parsed XYZ data (all atoms concatenated).
        smiles_list: SMILES strings, one per molecule in concatenation order.

    Returns:
        List of dicts per molecule with keys ``'smiles'``, ``'elements'``,
        ``'coords'`` (N_i, 3), and ``'charges'`` (list[float] | None).

    Raises:
        ValueError: If atom counts do not sum to the total.
    """
    expected_counts: list[int] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES in XYZ comment: {smi}")
        mol_h = Chem.AddHs(mol)
        expected_counts.append(mol_h.GetNumAtoms())

    if sum(expected_counts) != xyz_data.n_atoms:
        raise ValueError(
            f"SMILES imply {sum(expected_counts)} atoms but XYZ has {xyz_data.n_atoms}"
        )

    result: list[dict] = []
    offset = 0
    for smi, count in zip(smiles_list, expected_counts, strict=True):
        chunk: dict = {
            "smiles": smi,
            "elements": xyz_data.elements[offset : offset + count],
            "coords": xyz_data.coords[offset : offset + count],
            "charges": (
                xyz_data.charges[offset : offset + count] if xyz_data.charges is not None else None
            ),
        }
        result.append(chunk)
        offset += count

    return result


def mol_from_xyz_block(
    elements: list[str],
    coords: NDArray[np.floating],
    smiles: str,
) -> Chem.Mol:
    """Build an RDKit molecule from XYZ data using SMILES as bond-order template.

    Creates a raw molecule from element symbols and coordinates, then applies
    bond orders from the SMILES-derived template via
    ``AllChem.AssignBondOrdersFromTemplate``.  The returned molecule preserves
    the original XYZ atom ordering.

    Args:
        elements: Element symbols (length N).
        coords: (N, 3) coordinates in Angstroms.
        smiles: SMILES string for this molecule (bond-order source).

    Returns:
        RDKit Mol with correct bond orders, explicit Hs, and XYZ coordinates.

    Raises:
        ValueError: If template matching fails.
    """
    n = len(elements)
    xyz_block = f"{n}\n\n"
    for elem, (x, y, z) in zip(elements, coords, strict=True):
        xyz_block += f"{elem} {x:.8f} {y:.8f} {z:.8f}\n"

    raw_mol = Chem.MolFromXYZBlock(xyz_block)
    if raw_mol is None:
        raise ValueError("RDKit failed to parse XYZ block")

    # MolFromXYZBlock creates atoms without bonds; determine connectivity
    # from interatomic distances before template matching.
    rdDetermineBonds.DetermineConnectivity(raw_mol)

    template = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if template is None:
        raise ValueError(f"Invalid template SMILES: {smiles}")

    mol = AllChem.AssignBondOrdersFromTemplate(template, raw_mol)

    # The returned molecule needs full sanitization (ring info, valence
    # calculation, aromaticity) for downstream code (MMFF, OpenFF, etc.).
    Chem.SanitizeMol(mol)

    return mol


def mol_to_xyz_string(mol: Chem.Mol, comment: str = "") -> str:
    """Convert an RDKit molecule with a conformer to an XYZ format string.

    Args:
        mol: RDKit Mol object with a conformer
        comment: Optional comment for the second line of the XYZ file

    Returns:
        XYZ format string

    Raises:
        ValueError: If molecule has no conformer

    Examples:
        >>> mol = prepare_molecule_from_smiles("O")
        >>> xyz = mol_to_xyz_string(mol, comment="water")
        >>> lines = xyz.splitlines()
        >>> lines[0].strip()
        '3'
        >>> "water" in lines[1]
        True
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer")

    coords = get_molecule_coordinates(mol)
    n_atoms = mol.GetNumAtoms()

    lines = [str(n_atoms), comment]
    for i in range(n_atoms):
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        x, y, z = coords[i]
        lines.append(f"{symbol:>2s} {x:15.8f} {y:15.8f} {z:15.8f}")

    return "\n".join(lines) + "\n"


def write_xyz_files(
    reactants: list[Chem.Mol],
    products: list[Chem.Mol],
    output_dir: str | Path,
) -> dict[str, list[Path]]:
    """Write optimized reactant and product molecules as XYZ files.

    Creates two subdirectories under ``output_dir`` — ``reactants/`` and
    ``products/`` — and writes one XYZ file per molecule.

    Args:
        reactants: List of reactant RDKit Mol objects with 3D coordinates
        products: List of product RDKit Mol objects with 3D coordinates
        output_dir: Root directory for output. Will be created if it doesn't exist.

    Returns:
        Dictionary with keys ``'reactants'`` and ``'products'``, each containing
        a list of Path objects for the written files.

    Raises:
        ValueError: If any molecule lacks a conformer
    """
    root = Path(output_dir)
    reactant_dir = root / "reactants"
    product_dir = root / "products"
    reactant_dir.mkdir(parents=True, exist_ok=True)
    product_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, list[Path]] = {"reactants": [], "products": []}

    for i, mol in enumerate(reactants):
        path = reactant_dir / f"reactant_{i}.xyz"
        path.write_text(mol_to_xyz_string(mol, comment=f"reactant {i}"))
        written["reactants"].append(path)

    for i, mol in enumerate(products):
        path = product_dir / f"product_{i}.xyz"
        path.write_text(mol_to_xyz_string(mol, comment=f"product {i}"))
        written["products"].append(path)

    return written
