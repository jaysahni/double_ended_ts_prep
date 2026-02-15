"""Command-line driver for double-ended transition state prep.

Reads reactant/product SMILES (directly or from XYZ files) and runs the
full labeling + optimization pipeline, writing optimized geometries as XYZ.

Usage (XYZ mode -- default):
    pixi run python scripts/run_ts_prep.py
    pixi run python scripts/run_ts_prep.py --reactants reactants.xyz --products product.xyz

Usage (SMILES mode):
    pixi run python scripts/run_ts_prep.py --smiles --reactants "CCCCCCN=C=O" "CCCCCCO" \
        --products "CCCCCCNC(=O)OCCCCCC"

Options:
    --name NAME        Base name for output files ({name}_reactants.xyz, {name}_products.xyz)
    --alpha, --beta    Weighting parameters
    --max-iters        Maximum optimizer iterations
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.optimize import minimize

from double_ended_ts_prep.force_fields import (
    SystemState,
    _build_molecule_state,
    _build_openmm_simulation,
    _compute_rigid_body_energy,
    apply_rigid_transform,
    compute_geometric_error,
    get_atom_mapping_correspondence,
    get_molecule_coordinates,
    mol_from_xyz_block,
    optimize_molecule_geometry,
    parse_xyz_file,
    set_molecule_coordinates,
    split_xyz_by_molecules,
)
from double_ended_ts_prep.labeling import (
    build_smirks,
    map_smirks,
    smirks_to_molecules,
    transfer_atom_mapping,
)

MOLECULES_DIR = Path(__file__).resolve().parent.parent / "molecules"


def _build_comment_line(
    mols: list[Chem.Mol],
    energy: float,
    name: str,
) -> str:
    """Build a comment line matching the input XYZ format."""
    smiles_parts = []
    for mol in mols:
        mol_no_map = Chem.RWMol(Chem.RemoveHs(mol))
        for atom in mol_no_map.GetAtoms():
            atom.SetAtomMapNum(0)
        smiles_parts.append(Chem.MolToSmiles(mol_no_map))
    smiles_str = ".".join(smiles_parts)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"charge: 0; multiplicity: 1; energy: {energy:.8f}; "
        f"smiles: {smiles_str}; generated_by: ts_prep; "
        f"timestamp: {timestamp}; name: {name};"
    )


def _matched_xyz(mols: list[Chem.Mol], comment: str) -> str:
    """Write molecules to XYZ with atoms ordered by mapping number.

    Mapped atoms (map_num > 0) are sorted by map_num ascending, then
    unmapped atoms follow.  This ensures that reactant and product XYZ
    files have the same atom at the same line index for all mapped atoms.
    """
    mapped: list[tuple[int, str, float, float, float]] = []
    unmapped: list[tuple[str, float, float, float]] = []
    for mol in mols:
        coords = get_molecule_coordinates(mol)
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()
            x, y, z = coords[i]
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                mapped.append((map_num, symbol, x, y, z))
            else:
                unmapped.append((symbol, x, y, z))

    mapped.sort(key=lambda t: t[0])

    lines: list[str] = []
    for _, sym, x, y, z in mapped:
        lines.append(f"{sym:>2s} {x:15.8f} {y:15.8f} {z:15.8f}")
    for sym, x, y, z in unmapped:
        lines.append(f"{sym:>2s} {x:15.8f} {y:15.8f} {z:15.8f}")

    total = len(mapped) + len(unmapped)
    return f"{total}\n{comment}\n" + "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run double-ended transition state prep.",
    )

    # Input mode
    parser.add_argument(
        "--smiles",
        action="store_true",
        help="SMILES mode: --reactants and --products are SMILES strings instead of XYZ filenames",
    )

    # Input sources (meaning depends on --smiles flag)
    parser.add_argument(
        "--reactants",
        nargs="+",
        default=None,
        help=(
            "XYZ mode (default): single XYZ filename in molecules/ dir. "
            "SMILES mode: one or more reactant SMILES strings."
        ),
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=None,
        help=(
            "XYZ mode (default): single XYZ filename in molecules/ dir. "
            "SMILES mode: one or more product SMILES strings."
        ),
    )

    # Output
    parser.add_argument(
        "--name",
        default=None,
        help="Base name for output files: {name}_reactants.xyz / {name}_products.xyz",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: molecules/output/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of appending _1, _2, etc.",
    )

    # Optimizer
    parser.add_argument("--alpha", type=float, default=1.0, help="FF energy weight (default: 1.0)")
    parser.add_argument(
        "--beta", type=float, default=5.0, help="Geometric error weight (default: 1.0)"
    )
    parser.add_argument(
        "--max-iters", type=int, default=500, help="Max optimizer iterations (default: 500)"
    )
    parser.add_argument(
        "--gtol", type=float, default=1e-5, help="Gradient tolerance (default: 1e-5)"
    )
    return parser.parse_args()


def main() -> None:  # noqa: D103
    args = _parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else MOLECULES_DIR / "output"

    # ── Resolve input ────────────────────────────────────────────────────
    # Charges: use XYZ 5th-column if available, otherwise Gasteiger
    r_charges: list[list[float]] | None = None
    p_charges: list[list[float]] | None = None

    if args.smiles:
        # SMILES mode: args are SMILES strings directly
        if not args.reactants:
            raise SystemExit("Error: --reactants required in SMILES mode")
        if not args.products:
            raise SystemExit("Error: --products required in SMILES mode")
        r_smiles = args.reactants
        p_smiles = args.products
        print("Mode:      SMILES input")
        print("  Charges: Gasteiger")
        xyz_mode = False
    else:
        # XYZ mode: args are filenames (or defaults)
        r_file = args.reactants[0] if args.reactants else "reactants.xyz"
        p_file = args.products[0] if args.products else "product.xyz"
        reactant_path = MOLECULES_DIR / r_file
        product_path = MOLECULES_DIR / p_file
        print("Mode:      XYZ input")
        print(f"Reactants: {reactant_path}")
        print(f"Products:  {product_path}")
        xyz_mode = True

        # Parse full XYZ files (coordinates + optional charges)
        r_xyz = parse_xyz_file(reactant_path)
        p_xyz = parse_xyz_file(product_path)

        r_smiles = r_xyz.metadata["smiles"]
        p_smiles = p_xyz.metadata["smiles"]
        assert isinstance(r_smiles, list), f"Expected list of SMILES, got: {type(r_smiles)}"
        assert isinstance(p_smiles, list), f"Expected list of SMILES, got: {type(p_smiles)}"

        # Split multi-molecule XYZ into per-molecule chunks
        r_mol_data = split_xyz_by_molecules(r_xyz, r_smiles)
        p_mol_data = split_xyz_by_molecules(p_xyz, p_smiles)

        # Build RDKit molecules from XYZ coordinates
        r_mols_from_xyz = [
            mol_from_xyz_block(d["elements"], d["coords"], d["smiles"]) for d in r_mol_data
        ]
        p_mols_from_xyz = [
            mol_from_xyz_block(d["elements"], d["coords"], d["smiles"]) for d in p_mol_data
        ]

        # Use XYZ charges if 5th column present, otherwise Gasteiger
        has_r_charges = all(d["charges"] is not None for d in r_mol_data)
        has_p_charges = all(d["charges"] is not None for d in p_mol_data)
        if has_r_charges and has_p_charges:
            r_charges = [d["charges"] for d in r_mol_data]
            p_charges = [d["charges"] for d in p_mol_data]
            print("  Charges: from XYZ 5th column")
        else:
            print("  Charges: Gasteiger (no 5th column in XYZ)")

    # Determine output filenames
    base_name = args.name or ""
    if base_name:
        r_base, p_base = f"{base_name}_reactants", f"{base_name}_products"
    else:
        r_base, p_base = "reactants", "products"

    r_filename = f"{r_base}.xyz"
    p_filename = f"{p_base}.xyz"

    if not args.overwrite:
        n = 1
        while (output_dir / r_filename).exists() or (output_dir / p_filename).exists():
            r_filename = f"{r_base}_{n}.xyz"
            p_filename = f"{p_base}_{n}.xyz"
            n += 1

    print(f"Output:    {output_dir}")
    print(f"Params:    alpha={args.alpha}, beta={args.beta}, max_iters={args.max_iters}")
    print(f"Tolerance: ftol=1e-12, gtol={args.gtol}")
    print()

    # ── Phase A: Atom labeling ──────────────────────────────────────────
    print("Phase A: Atom labeling...")
    t0 = time.perf_counter()

    print(f"  Reactant SMILES: {r_smiles}")
    print(f"  Product SMILES:  {p_smiles}")

    smirks = build_smirks(r_smiles, p_smiles)
    max_display = 80
    print(f"  Unmapped SMIRKS: {smirks[:max_display]}{'...' if len(smirks) > max_display else ''}")

    mapped = map_smirks(smirks)
    print(f"  Mapped SMIRKS:   {mapped[:max_display]}{'...' if len(mapped) > max_display else ''}")

    if xyz_mode:
        # Transfer atom mapping from RXNMapper onto XYZ-loaded molecules
        mapped_mols = transfer_atom_mapping(mapped, r_mols_from_xyz, p_mols_from_xyz)
        reactants = mapped_mols["reactants"]
        products = mapped_mols["products"]
    else:
        # SMILES mode: generate 3D coords from mapped SMIRKS (existing behavior)
        mols = smirks_to_molecules(mapped)
        reactants = mols["reactants"]
        products = mols["products"]

    # Optimize internal geometry (MMFF94) so each molecule is at its
    # force-field minimum before being treated as a rigid body.
    for mol in reactants + products:
        optimize_molecule_geometry(mol)

    t_label = time.perf_counter() - t0
    print(f"  -> {len(reactants)} reactant(s), {len(products)} product(s)")
    print(f"  -> Completed in {t_label:.2f}s")
    print()

    # ── Phase B: Simulation state setup ─────────────────────────────────
    print("Phase B: Simulation state setup...")
    t0 = time.perf_counter()

    reactant_states = [_build_molecule_state(mol) for mol in reactants]
    product_states = [_build_molecule_state(mol) for mol in products]

    reactant_simulation, reactant_atom_offsets = _build_openmm_simulation(
        reactants,
        preset_charges=r_charges,
    )
    product_simulation, product_atom_offsets = _build_openmm_simulation(
        products,
        preset_charges=p_charges,
    )

    correspondence = get_atom_mapping_correspondence(reactants, products)
    print(f"  Atom mapping pairs: {len(correspondence)}")

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
    )

    t_setup = time.perf_counter() - t0
    total_dofs = n_reactant_params + n_product_params
    print(f"  -> DOFs: {total_dofs} ({n_reactant_params} reactant, {n_product_params} product)")
    print(f"  -> Completed in {t_setup:.2f}s")
    print()

    # ── Phase C: Optimization ───────────────────────────────────────────
    print("Phase C: Optimization...")
    t0 = time.perf_counter()

    initial_params = np.zeros(n_reactant_params + n_product_params, dtype=np.float64)

    bounds: list[tuple[float | None, float | None]] = []
    for _ in range(len(reactants) + len(products)):
        bounds.extend([(None, None), (None, None), (None, None)])
        bounds.extend([(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)])

    result = minimize(
        _compute_rigid_body_energy,
        initial_params,
        args=(system_state, args.alpha, args.beta),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": args.max_iters, "ftol": 1e-12, "gtol": args.gtol},
    )

    t_opt = time.perf_counter() - t0

    if result.success:
        print(f"  -> Converged in {result.nit} iterations ({t_opt:.2f}s)")
    else:
        print(f"  -> Did NOT converge: {result.message}")
        print(f"  -> Reached {result.nit}/{args.max_iters} iterations in {t_opt:.2f}s")
    print()

    # ── Apply transforms and compute final metrics ──────────────────────
    final_params = result.x
    reactant_params = final_params[:n_reactant_params].reshape(-1, 6)
    product_params = final_params[n_reactant_params:].reshape(-1, 6)

    optimized_reactants: list[Chem.Mol] = []
    reactant_coords_list = []
    for i, state in enumerate(reactant_states):
        trans = reactant_params[i, :3]
        rot = reactant_params[i, 3:]
        new_coords = apply_rigid_transform(state.initial_coords, state.centroid, trans, rot)
        reactant_coords_list.append(new_coords)
        mol_copy = Chem.RWMol(state.mol)
        set_molecule_coordinates(mol_copy, new_coords)
        optimized_reactants.append(mol_copy.GetMol())

    optimized_products: list[Chem.Mol] = []
    product_coords_list = []
    for i, state in enumerate(product_states):
        trans = product_params[i, :3]
        rot = product_params[i, 3:]
        new_coords = apply_rigid_transform(state.initial_coords, state.centroid, trans, rot)
        product_coords_list.append(new_coords)
        mol_copy = Chem.RWMol(state.mol)
        set_molecule_coordinates(mol_copy, new_coords)
        optimized_products.append(mol_copy.GetMol())

    final_geo_error = compute_geometric_error(
        reactant_coords_list, product_coords_list, correspondence
    )

    # ── Write output XYZ files ──────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pure FF energy (no geometric error contribution)
    ff_energy = _compute_rigid_body_energy(result.x, system_state, alpha=1.0, beta=0.0)

    name_label = args.name or "optimized"
    r_comment = _build_comment_line(optimized_reactants, ff_energy, f"{name_label} reactants")
    p_comment = _build_comment_line(optimized_products, ff_energy, f"{name_label} products")

    r_path = output_dir / r_filename
    p_path = output_dir / p_filename

    r_path.write_text(_matched_xyz(optimized_reactants, r_comment))
    p_path.write_text(_matched_xyz(optimized_products, p_comment))

    print("Output written to:")
    print(f"  {r_path}")
    print(f"  {p_path}")
    print()

    # ── Summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Alpha:             {args.alpha}")
    print(f"  Beta:              {args.beta}")
    print(f"  FF energy:         {ff_energy:.4f} kcal/mol")
    print(f"  Geometric error:   {final_geo_error:.4f} A^2")
    print(f"  Total objective:   {result.fun:.4f} (alpha*FF + beta*geo)")
    print(f"  Converged:         {result.success}")
    print(f"  Iterations:        {result.nit}")
    print()
    print("Timing:")
    print(f"  a) Atom labeling:      {t_label:>8.2f}s")
    print(f"  b) Simulation setup:   {t_setup:>8.2f}s")
    print(f"  c) Optimization:       {t_opt:>8.2f}s")
    print(f"     Total:              {t_label + t_setup + t_opt:>8.2f}s")


if __name__ == "__main__":
    main()
