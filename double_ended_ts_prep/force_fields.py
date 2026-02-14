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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import openmm
import openmm.app
from openff.interchange import Interchange
from openff.toolkit import ForceField as OFFForceField
from openff.toolkit import Molecule as OFFMolecule
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Conversion factor: 1 Hartree = 627.5094740631 kcal/mol
KCAL_MOL_TO_HARTREE = 1.0 / 627.5094740631

# Coulomb constant in kcal*A/(mol*e^2) for vacuum electrostatics
COULOMB_K = 332.0637


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
class SystemState:
    """Complete state for rigid-body optimization of reactants and products.

    Reactants and products are completely isolated energy systems. Each side
    has its own OpenMM simulation for computing intramolecular + intermolecular
    interactions within that side only. The geometric error (via atom mapping)
    is the only coupling between the two sides.

    Attributes:
        reactant_states: List of MoleculeState for each reactant
        product_states: List of MoleculeState for each product
        atom_mapping: Correspondence between (r_mol_idx, atom_idx) -> (p_mol_idx, atom_idx)
        n_reactant_params: Total DOFs for reactant side (6 * num_reactants)
        n_product_params: Total DOFs for product side (6 * num_products)
        reactant_simulation: OpenMM simulation for reactant system only
        product_simulation: OpenMM simulation for product system only
        reactant_atom_offsets: Starting atom index of each reactant in the combined system
        product_atom_offsets: Starting atom index of each product in the combined system
        reactant_nb_params: (N_r_total, 3) nonbonded params [q, sigma, eps] for ghost cross-energy
        product_nb_params: (N_p_total, 3) nonbonded params [q, sigma, eps] for ghost cross-energy
    """

    reactant_states: list[MoleculeState]
    product_states: list[MoleculeState]
    atom_mapping: dict[tuple[int, int], tuple[int, int]]
    n_reactant_params: int
    n_product_params: int
    reactant_simulation: openmm.app.Simulation
    product_simulation: openmm.app.Simulation
    reactant_atom_offsets: list[int]
    product_atom_offsets: list[int]
    reactant_nb_params: NDArray[np.floating]
    product_nb_params: NDArray[np.floating]


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


def _build_openmm_simulation(
    mols: list[Chem.Mol],
) -> tuple[openmm.app.Simulation, list[int]]:
    """Build an OpenMM simulation from a list of RDKit molecules.

    Converts RDKit molecules to OpenFF molecules, assigns AM1-BCC partial
    charges, builds an Interchange with the OpenFF-2.2.1 force field, and
    creates an OpenMM simulation. This is called ONCE during setup and the
    simulation object is reused for repeated energy evaluations.

    Args:
        mols: List of RDKit Mol objects with conformers

    Returns:
        Tuple of (OpenMM Simulation, list of atom offsets for each molecule)

    Raises:
        ValueError: If OpenFF conversion or force field setup fails
    """
    forcefield = OFFForceField("openff_unconstrained-2.2.1.offxml")
    off_mols: list[OFFMolecule] = []
    atom_offsets: list[int] = []
    running_offset = 0

    for mol in mols:
        atom_offsets.append(running_offset)
        off_mol = OFFMolecule.from_rdkit(mol)
        off_mol.assign_partial_charges(partial_charge_method="am1bcc")
        off_mols.append(off_mol)
        running_offset += mol.GetNumAtoms()

    interchange = Interchange.from_smirnoff(forcefield, off_mols)
    integrator = openmm.VerletIntegrator(1 * openmm.unit.femtoseconds)  # type: ignore[unresolved-attribute]
    simulation = interchange.to_openmm_simulation(integrator)

    return simulation, atom_offsets


def _compute_openmm_energy(
    simulation: openmm.app.Simulation,
    coords_angstrom: NDArray[np.floating],
) -> float:
    """Compute OpenMM energy for given coordinates.

    Args:
        simulation: Pre-built OpenMM simulation (reused across iterations)
        coords_angstrom: (N, 3) coordinates in Angstroms

    Returns:
        Energy in kcal/mol
    """
    # OpenMM expects positions in nanometers
    simulation.context.setPositions(coords_angstrom / 10)
    state = simulation.context.getState(getEnergy=True)
    energy_kj = state.getPotentialEnergy() / openmm.unit.kilojoules_per_mole
    return float(energy_kj) * 0.239006  # kJ/mol → kcal/mol


def _extract_nonbonded_params(
    simulation: openmm.app.Simulation,
) -> NDArray[np.floating]:
    """Extract nonbonded parameters (charge, sigma, epsilon) for each atom.

    Reads from the NonbondedForce in the OpenMM system built by OpenFF.
    Parameters are converted to convenient units: charges in elementary charge,
    sigma in Angstroms, epsilon in kcal/mol.

    Args:
        simulation: OpenMM Simulation with a NonbondedForce

    Returns:
        (N, 3) array where columns are [charge_e, sigma_A, epsilon_kcal]

    Raises:
        ValueError: If no NonbondedForce is found in the system
    """
    system = simulation.context.getSystem()
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, openmm.NonbondedForce):
            n = force.getNumParticles()
            params = np.zeros((n, 3), dtype=np.float64)
            for j in range(n):
                q, s, e = force.getParticleParameters(j)
                params[j, 0] = q / openmm.unit.elementary_charge
                params[j, 1] = s / openmm.unit.nanometer * 10  # type: ignore[unresolved-attribute]  # nm → Å
                params[j, 2] = e / openmm.unit.kilojoules_per_mole * 0.239006  # kJ/mol → kcal/mol
            return params
    raise ValueError("No NonbondedForce found in the OpenMM system")


def _compute_ghost_cross_energy(
    reactant_coords: NDArray[np.floating],
    product_coords: NDArray[np.floating],
    reactant_params: NDArray[np.floating],
    product_params: NDArray[np.floating],
    alpha_sc: float = 0.5,
) -> float:
    """Compute ghost cross-interaction energy between reactant and product atoms.

    Uses soft-core potentials (Beutler et al., 1994) to compute attractive LJ
    (r^-6 only, no r^-12 repulsion) plus Coulomb electrostatics between all
    reactant-product atom pairs. Soft-core potentials replace bare r with an
    effective distance that remains finite at r=0, preventing the singularity
    that occurs when molecules overlap.

    Soft-core LJ:  -4 * eps * sigma^6 / (alpha_sc * sigma^6 + r^6)
    Soft-core Coulomb:  COULOMB_K * q_ij / sqrt(alpha_sc * sigma_ij^2 + r^2)

    Uses Lorentz-Berthelot combining rules:
        sigma_ij = (sigma_i + sigma_j) / 2
        eps_ij   = sqrt(eps_i * eps_j)

    Args:
        reactant_coords: (N_r, 3) coordinates in Angstroms
        product_coords: (N_p, 3) coordinates in Angstroms
        reactant_params: (N_r, 3) array of [charge_e, sigma_A, epsilon_kcal]
        product_params: (N_p, 3) array of [charge_e, sigma_A, epsilon_kcal]
        alpha_sc: Soft-core parameter controlling singularity damping (default 0.5)

    Returns:
        Ghost cross-interaction energy in kcal/mol
    """
    # Pairwise distance matrix: (N_r, N_p)
    diff = reactant_coords[:, None, :] - product_coords[None, :, :]
    r_sq = np.sum(diff**2, axis=-1)

    # Lorentz-Berthelot combining rules
    sigma_ij = (reactant_params[:, 1, None] + product_params[None, :, 1]) / 2
    eps_ij = np.sqrt(reactant_params[:, 2, None] * product_params[None, :, 2])

    # Soft-core attractive LJ: -4 * eps * sigma^6 / (alpha_sc * sigma^6 + r^6)
    sigma6 = sigma_ij**6
    r6 = r_sq**3
    v_disp = -4.0 * eps_ij * sigma6 / (alpha_sc * sigma6 + r6)

    # Soft-core Coulomb: COULOMB_K * q_ij / sqrt(alpha_sc * sigma_ij^2 + r^2)
    q_ij = reactant_params[:, 0, None] * product_params[None, :, 0]
    r_eff = np.sqrt(alpha_sc * sigma_ij**2 + r_sq)
    v_coul = COULOMB_K * q_ij / r_eff

    return float(np.sum(v_disp + v_coul))


def _compute_rigid_body_energy(
    params: NDArray[np.floating],
    system_state: SystemState,
    alpha: float,
    beta: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """Compute total energy for scipy optimizer using separated reactant/product systems.

    Energy = alpha * (E_reactants + E_products) + beta * geometric_error
             + gamma * E_ghost_cross

    Reactants and products are completely isolated energy systems:
    - E_reactants: OpenMM energy computed ONLY among reactant molecules
    - E_products: OpenMM energy computed ONLY among product molecules
    - geometric_error via atom mapping couples the two sides geometrically
    - E_ghost_cross: attractive LJ (r^-6) + Coulomb between sides (no repulsion)

    Args:
        params: 1D array of all rigid body parameters
        system_state: SystemState with pre-built OpenMM simulations
        alpha: Weight for force field energy term (kcal/mol)
        beta: Weight for geometric error term (kcal/mol per Angstrom^2)
        gamma: Weight for ghost cross-interaction term (default 0.0)

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

    reactant_energy = _compute_openmm_energy(
        system_state.reactant_simulation, combined_reactant_coords
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

    product_energy = _compute_openmm_energy(
        system_state.product_simulation, combined_product_coords
    )

    # Energy from isolated systems (no cross-interactions)
    ff_energy = alpha * (reactant_energy + product_energy)

    # Geometric error: coupling term between reactants and products
    geo_error = compute_geometric_error(
        reactant_coords_list, product_coords_list, system_state.atom_mapping
    )

    total = ff_energy + beta * geo_error

    # Ghost cross-interaction: attractive LJ + Coulomb between sides (no repulsion)
    if gamma != 0.0:
        cross_energy = _compute_ghost_cross_energy(
            combined_reactant_coords,
            combined_product_coords,
            system_state.reactant_nb_params,
            system_state.product_nb_params,
        )
        total += gamma * cross_energy

    return total


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
    beta: float = 1.0,
    gamma: float = 0.0,
    max_iters: int = 500,
    gtol: float = 1e-5,
) -> dict[str, list[Chem.Mol] | float | bool]:
    """Optimize reactant/product geometries for transition state search.

    Performs rigid-body optimization to minimize:
    E = alpha * (E_reactants + E_products) + beta * geometric_error
        + gamma * E_ghost_cross

    Each molecule is treated as a rigid body with 6 degrees of freedom
    (3 translations + 3 rotations). The geometric error penalizes
    distances between corresponding atoms (matched by atom mapping numbers).
    The ghost cross-interaction term provides attractive dispersion and
    electrostatic coupling between reactant and product molecules without
    repulsive overlap penalties.

    Args:
        reactants: List of reactant RDKit Mol objects with atom mapping and 3D coords
        products: List of product RDKit Mol objects with atom mapping and 3D coords
        alpha: Weight for force field energy term (default 1.0 kcal/mol)
        beta: Weight for geometric error term (default 1.0 kcal/mol per Angstrom^2)
        gamma: Weight for ghost cross-interaction term (default 0.0)
        max_iters: Maximum L-BFGS iterations
        gtol: Gradient tolerance for convergence

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
        >>> result['success']
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

    # Build OpenMM simulations for each side (done ONCE, reused in optimization loop)
    reactant_simulation, reactant_atom_offsets = _build_openmm_simulation(reactants)
    product_simulation, product_atom_offsets = _build_openmm_simulation(products)

    # Extract nonbonded parameters for ghost cross-interaction
    reactant_nb_params = _extract_nonbonded_params(reactant_simulation)
    product_nb_params = _extract_nonbonded_params(product_simulation)

    # Create system state
    n_reactant_params = 6 * len(reactants)
    n_product_params = 6 * len(products)
    system_state = SystemState(
        reactant_states=reactant_states,
        product_states=product_states,
        atom_mapping=correspondence,
        n_reactant_params=n_reactant_params,
        n_product_params=n_product_params,
        reactant_simulation=reactant_simulation,
        product_simulation=product_simulation,
        reactant_atom_offsets=reactant_atom_offsets,
        product_atom_offsets=product_atom_offsets,
        reactant_nb_params=reactant_nb_params,
        product_nb_params=product_nb_params,
    )

    # Initialize parameters: all zeros (molecules start overlapping)
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
        args=(system_state, alpha, beta, gamma),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iters, "gtol": gtol},
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
