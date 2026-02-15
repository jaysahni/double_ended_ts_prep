# Methodology: Rigid-Body Optimization for Transition State Preparation

This document provides a detailed technical explanation of how `double_ended_ts_prep` prepares molecular geometries for double-ended transition state searches.

## Overview

The package solves the following problem: given reactant and product molecules for a chemical reaction, find spatial arrangements of each molecule such that:

1. Atoms that correspond between reactants and products (via atom mapping) are close together in 3D space
2. Molecules within each side (reactant-side or product-side) do not sterically clash

This is formulated as a rigid-body optimization problem where each molecule can translate and rotate but its internal geometry is fixed.

## Pipeline

### Step 1: Reaction Definition and Atom Mapping

The process begins with reactant and product SMILES strings. These are combined into a SMIRKS reaction string using `build_smirks()`:

```
CCO.CC(=O)O >> CC(=O)OCC.O
```

The unmapped SMIRKS is then passed to `map_smirks()`, which uses [RXNMapper](https://github.com/rxn4chemistry/rxnmapper) to assign atom mapping numbers. RXNMapper is a transformer-based model that uses attention patterns to determine which atoms in the reactants correspond to which atoms in the products:

```
[CH3:1][CH2:2][OH:3].[CH3:4][C:5](=[O:6])[OH:7]>>[CH3:1][CH2:2][O:3][C:5]([CH3:4])=[O:6].[OH2:7]
```

The confidence score is checked against a minimum threshold (0.7); low-confidence mappings trigger a warning since incorrect atom correspondence will lead to poor optimization results.

### Step 2: 3D Coordinate Generation

`smirks_to_molecules()` parses the mapped SMIRKS and generates 3D coordinates for each molecule:

1. Parse each SMILES fragment preserving atom mapping numbers
2. Add explicit hydrogens (`Chem.AddHs`)
3. Embed using RDKit's ETKDGv3 distance geometry algorithm, which produces chemically reasonable 3D conformations

At this stage, each molecule has valid 3D coordinates but the molecules are not spatially related to each other.

### Step 3: System Setup

`optimize_ts_prep()` prepares the optimization system:

**Atom Mapping Correspondence:** `get_atom_mapping_correspondence()` builds a dictionary mapping `(reactant_mol_index, atom_index)` to `(product_mol_index, atom_index)` by matching atom mapping numbers across the reactant and product sides. Only atoms with nonzero mapping numbers that appear on both sides are included.

**OpenMM Simulations:** Two separate OpenMM simulations are built -- one for all reactant molecules and one for all product molecules. Each simulation uses:
- The [OpenFF 2.2.1 (Sage)](https://openforcefield.org/) force field for bonded and nonbonded parameters
- AM1-BCC partial charges computed by the OpenFF toolkit
- A combined system containing all molecules on that side

Building the simulations once and reusing them across optimization iterations avoids repeated force field parameterization.

### Step 4: Rigid-Body Optimization

Each molecule is parameterized by 6 degrees of freedom:
- **Translation:** `[tx, ty, tz]` in Angstroms (unbounded)
- **Rotation:** `[rx, ry, rz]` Euler angles in radians (bounded to [-pi, pi], XYZ convention)

Transformations are applied around each molecule's centroid: the molecule is centered, rotated using `scipy.spatial.transform.Rotation`, then translated back and offset.

All parameters are initialized to zero, meaning all molecules start at their original (overlapping) positions. The L-BFGS-B optimizer then minimizes the total energy functional.

## Energy Functional

The total energy being minimized is:

```
E_total = alpha * (E_reactants + E_products) + beta * E_geometric
```

### Term 1: Force Field Energy (alpha)

```
E_ff = alpha * (E_reactants + E_products)
```

Each side's energy is computed independently using OpenMM. This includes all bonded terms (bonds, angles, torsions) and nonbonded terms (Lennard-Jones + Coulomb) *within* that side's molecules. The reactant and product systems do not interact through this term.

This term serves two purposes:
- **Intramolecular stability:** Penalizes internal strain (though rigid-body transforms preserve internal geometry, so this is constant for single-molecule systems)
- **Intermolecular repulsion within each side:** When multiple reactant molecules (or multiple product molecules) are present, this term prevents them from occupying the same space

The energy is evaluated by updating positions in a pre-built OpenMM context and querying the potential energy. Coordinates are converted from Angstroms to nanometers (OpenMM's native unit), and energies are converted from kJ/mol to kcal/mol.

### Term 2: Geometric Error (beta)

```
E_geometric = beta * sum_i ||r_i - p_i||^2
```

This is the sum of squared Euclidean distances between all pairs of corresponding atoms (identified by atom mapping). It is the primary coupling between the reactant and product sides.

The sum-of-squared-distances form (rather than RMSD) is used because it provides better gradient behavior for the L-BFGS-B optimizer -- the gradient with respect to atomic position is simply `2 * (r_i - p_i)`, which is smooth and well-behaved everywhere.

## Optimization Details

### Optimizer

The L-BFGS-B algorithm (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bound constraints) is used via `scipy.optimize.minimize`. It is a quasi-Newton method well-suited for smooth, differentiable objectives with simple bounds:

- Translations are unbounded
- Rotations are bounded to [-pi, pi] to avoid redundant parameterization
- Gradients are computed by finite differences (SciPy's default for L-BFGS-B)
- Default convergence: gradient tolerance `gtol=1e-5`, max 500 iterations

### Initial Conditions

All rigid-body parameters start at zero, meaning molecules begin at their embedded positions. Since 3D embedding places each molecule independently, molecules from different fragments typically start overlapping -- the optimizer must separate them while satisfying the geometric correspondence constraint.

### Output

The optimizer returns:
- **Optimized molecules:** RDKit `Mol` objects with updated 3D coordinates
- **Final energy:** Total energy at the minimum (kcal/mol)
- **Geometric error:** Final sum of squared distances between mapped atoms (A^2)
- **Success flag:** Whether the optimizer converged

Optimized geometries can be exported to XYZ files using `write_xyz_files()`, producing one file per molecule under `reactants/` and `products/` subdirectories.

## Architecture

```
double_ended_ts_prep/
  labeling.py       -- SMIRKS construction, atom mapping (RXNMapper), SMILES-to-molecule parsing
  force_fields.py   -- Rigid-body optimization, OpenMM energy evaluation, XYZ export
```

### Data Structures

**`MoleculeState`**: Holds per-molecule data during optimization -- the original RDKit molecule, initial coordinates, centroid (rotation pivot), and atom count.

**`SystemState`**: Complete optimization state -- molecule states for both sides, atom mapping correspondence, OpenMM simulations, and atom offsets for indexing into the combined coordinate arrays.

### Computational Notes

- OpenMM simulations are built once per optimization call, not per energy evaluation. This avoids repeated force field parameterization and system construction.
- All energies are in kcal/mol. Coordinate units are Angstroms internally; OpenMM conversions (to nm and kJ/mol) are handled at the interface.

## References

- Schwaller, P., Hoover, B., Reymond, J. L., Strobelt, H., & Laino, T. (2021). Extraction of organic chemistry grammar from unsupervised learning of chemical reactions. *Sci. Adv.*, 7(15), eabe4166.
- Qiu, Y., Smith, D. G. A., Boothroyd, S., et al. (2021). Development and benchmarking of Open Force Field v2.0.0 -- the Sage small molecule force field. *J. Chem. Theory Comput.*, 17(10), 6262-6280.
