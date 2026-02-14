# double_ended_ts_prep

[![License](https://img.shields.io/github/license/jaysahni/double_ended_ts_prep)](https://github.com/jaysahni/double_ended_ts_prep/blob/master/LICENSE)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/jaysahni/double_ended_ts_prep/test.yml?branch=master&logo=github-actions)](https://github.com/jaysahni/double_ended_ts_prep/actions/)
[![Codecov](https://img.shields.io/codecov/c/github/jaysahni/double_ended_ts_prep)](https://codecov.io/gh/jaysahni/double_ended_ts_prep)

Automatic geometry preparation for double-ended transition state searches. Given reactant and product SMILES, this package generates atom-mapped 3D structures with optimized spatial alignment, ready for use with TS search methods like the growing string method (GSM) or nudged elastic band (NEB).

## Motivation

Double-ended transition state search methods require reasonable starting geometries for both reactants and products where corresponding atoms are spatially close. Poor initial alignment leads to failed searches or convergence to incorrect saddle points. This package automates the preparation step by:

1. Generating atom mappings between reactants and products using RXNMapper
2. Embedding 3D coordinates for each molecule
3. Optimizing rigid-body positions so that mapped atoms are close together while maintaining physically reasonable molecular geometries

## Installation

This project uses [Pixi](https://pixi.sh) for environment management:

```bash
pixi install
```

### Dependencies

- **RDKit** - Cheminformatics (SMILES parsing, 3D embedding, coordinate manipulation)
- **RXNMapper** - Attention-guided atom mapping for chemical reactions
- **OpenMM** - Molecular mechanics energy evaluation
- **OpenFF Toolkit** - Open Force Field parameterization (OpenFF-2.2.1)
- **SciPy** - L-BFGS-B optimization
- **NumPy** - Array operations

## Quick Start

```python
from double_ended_ts_prep.labeling import build_smirks, map_smirks, smirks_to_molecules
from double_ended_ts_prep.force_fields import optimize_ts_prep, write_xyz_files

# 1. Define a reaction: isocyanate + water -> carbamic acid
smirks = build_smirks(["N=C=O", "O"], ["NC(=O)O"])

# 2. Generate atom mapping
mapped_smirks = map_smirks(smirks)

# 3. Parse into 3D molecules
mols = smirks_to_molecules(mapped_smirks)

# 4. Optimize spatial alignment
result = optimize_ts_prep(
    mols["reactants"],
    mols["products"],
    alpha=1.0,   # force field weight
    beta=1.0,    # geometric alignment weight
    gamma=1.0,   # cross-interaction attraction weight
)

print(f"Converged: {result['success']}")
print(f"Final energy: {result['final_energy']:.2f} kcal/mol")
print(f"Geometric error: {result['geometric_error']:.4f} A^2")

# 5. Export to XYZ files for your TS search code
write_xyz_files(result["reactants"], result["products"], "output/")
```

## API Reference

### Labeling Module (`labeling`)

#### `build_smirks(reactants, products) -> str`

Constructs a SMIRKS reaction string from lists of reactant and product SMILES. Validates all SMILES and canonicalizes with explicit hydrogens.

```python
smirks = build_smirks(["CCO", "CC(=O)O"], ["CC(=O)OCC", "O"])
# "[CH3][CH2][OH].[CH3][C](=[O])[OH]>>[CH3][CH2][O][C]([CH3])=[O].[OH2]"
```

#### `map_smirks(unmapped_smirks) -> str`

Applies RXNMapper's attention-guided algorithm to assign atom mapping numbers. Issues a warning if mapping confidence falls below 0.7.

```python
mapped = map_smirks("CC(C)S>>CC(C)Sc1ncccc1F")
# Returns SMIRKS with :1, :2, ... atom labels
```

#### `smirks_to_molecules(smirks) -> dict`

Parses a mapped SMIRKS string into RDKit molecules with 3D coordinates. Each molecule gets explicit hydrogens and ETKDGv3-embedded coordinates.

```python
mols = smirks_to_molecules("[CH3:1][OH:2]>>[CH3:1][O:2][CH3:3]")
reactants = mols["reactants"]  # list[Chem.Mol]
products = mols["products"]    # list[Chem.Mol]
```

### Force Fields Module (`force_fields`)

#### `optimize_ts_prep(reactants, products, alpha, beta, gamma, ...) -> dict`

The main entry point. Optimizes rigid-body positions of reactant and product molecules by minimizing:

```
E = alpha * (E_reactants + E_products) + beta * geometric_error + gamma * E_ghost_cross
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reactants` | `list[Chem.Mol]` | required | Reactant molecules with atom mapping and 3D coords |
| `products` | `list[Chem.Mol]` | required | Product molecules with atom mapping and 3D coords |
| `alpha` | `float` | `1.0` | Weight for intramolecular force field energy (kcal/mol) |
| `beta` | `float` | `1.0` | Weight for geometric alignment penalty (kcal/mol per A^2) |
| `gamma` | `float` | `0.0` | Weight for ghost cross-interaction attraction |
| `max_iters` | `int` | `500` | Maximum L-BFGS-B iterations |
| `gtol` | `float` | `1e-5` | Gradient tolerance for convergence |

**Returns:** Dictionary with keys `reactants`, `products` (optimized `Mol` objects), `final_energy`, `geometric_error`, and `success`.

#### `write_xyz_files(reactants, products, output_dir) -> dict`

Exports molecules to XYZ files under `output_dir/reactants/` and `output_dir/products/`.

#### `prepare_molecule_from_smiles(smiles) -> Chem.Mol`

Standalone utility to create a 3D-embedded RDKit molecule from a SMILES string.

## Tuning the Energy Weights

The three weights (`alpha`, `beta`, `gamma`) control the balance between competing objectives:

- **`alpha` (force field):** Penalizes steric clashes and strain *within* each side. Higher values keep molecules from overlapping with their same-side neighbors but may resist geometric alignment.

- **`beta` (geometric error):** Drives mapped atoms together. This is the primary coupling between reactants and products. Higher values force tighter alignment at the cost of potentially strained geometries.

- **`gamma` (ghost cross-interaction):** Adds attractive dispersion and electrostatic forces *between* reactant and product molecules using soft-core potentials. This provides a physics-based attraction that complements the geometric penalty. Values around 0.5--2.0 are typical starting points.

A reasonable default is `alpha=1.0, beta=1.0, gamma=1.0`. For reactions with large geometric rearrangements, increasing `beta` or `gamma` may help. For systems where steric clashes are a concern, increase `alpha`.

## How It Works

See [METHODOLOGY.md](METHODOLOGY.md) for a detailed technical explanation of the optimization procedure, energy terms, and soft-core potential formulation.

## Development

```bash
pixi run fmt        # Format code with ruff
pixi run lint       # Lint with ruff
pixi run types      # Type-check with ty
pixi run test       # Run tests with pytest
pixi run all        # Run all of the above
```

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.
