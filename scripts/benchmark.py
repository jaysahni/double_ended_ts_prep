"""Benchmark: rigid-body optimization vs random molecule placement.

Compares the library's L-BFGS-B optimization against randomly placing
molecules in 3D space, using both XYZ and SMILES inputs.

Usage:
    pixi run python scripts/benchmark.py
    pixi run python scripts/benchmark.py --trials 100 --box-size 15
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.optimize import minimize

from double_ended_ts_prep.force_fields import (
    SystemState,
    _build_molecule_state,
    _build_pairwise_params,
    _compute_pairwise_energy,
    _compute_rigid_body_energy,
    apply_rigid_transform,
    compute_geometric_error,
    get_atom_mapping_correspondence,
    get_molecule_coordinates,
    mol_from_xyz_block,
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


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class ReactionSpec:
    """Specification for a benchmark reaction."""

    name: str
    mode: str  # "smiles" or "xyz"
    reactant_smiles: list[str] | None = None
    product_smiles: list[str] | None = None
    reactant_xyz: str | None = None
    product_xyz: str | None = None


@dataclass
class Metrics:
    """Metrics from a placement evaluation."""

    geo_error: float
    pairwise_energy: float
    objective: float


# ---------------------------------------------------------------------------
# Reaction definitions
# ---------------------------------------------------------------------------
def define_reactions() -> list[ReactionSpec]:
    """Return the list of benchmark reactions."""
    return [
        ReactionSpec(
            "Urethane formation (XYZ)",
            "xyz",
            reactant_xyz=str(MOLECULES_DIR / "reactants.xyz"),
            product_xyz=str(MOLECULES_DIR / "product.xyz"),
        ),
        ReactionSpec(
            "Ester hydrolysis (2->2)",
            "smiles",
            reactant_smiles=["CC(=O)OC", "O"],
            product_smiles=["CC(=O)O", "CO"],
        ),
        ReactionSpec(
            "Diels-Alder (2->1)",
            "smiles",
            reactant_smiles=["C=CC=C", "C=C"],
            product_smiles=["C1CC=CCC1"],
        ),
        ReactionSpec(
            "Amide formation (2->2)",
            "smiles",
            reactant_smiles=["CC(=O)O", "NCC"],
            product_smiles=["CC(=O)NCC", "O"],
        ),
    ]


# ---------------------------------------------------------------------------
# Reaction preparation
# ---------------------------------------------------------------------------
def prepare_reaction(spec: ReactionSpec) -> tuple[SystemState, int]:
    """Build a SystemState from a ReactionSpec.

    Returns (system_state, n_mapped_atoms).
    """
    r_charges: list[list[float]] | None = None
    p_charges: list[list[float]] | None = None

    if spec.mode == "xyz":
        assert spec.reactant_xyz is not None
        assert spec.product_xyz is not None
        r_xyz = parse_xyz_file(spec.reactant_xyz)
        p_xyz = parse_xyz_file(spec.product_xyz)
        r_smiles = r_xyz.metadata["smiles"]
        p_smiles = p_xyz.metadata["smiles"]
        assert isinstance(r_smiles, list)
        assert isinstance(p_smiles, list)

        r_mol_data = split_xyz_by_molecules(r_xyz, r_smiles)
        p_mol_data = split_xyz_by_molecules(p_xyz, p_smiles)

        r_mols_from_xyz = [
            mol_from_xyz_block(d["elements"], d["coords"], d["smiles"]) for d in r_mol_data
        ]
        p_mols_from_xyz = [
            mol_from_xyz_block(d["elements"], d["coords"], d["smiles"]) for d in p_mol_data
        ]

        if all(d["charges"] is not None for d in r_mol_data) and all(
            d["charges"] is not None for d in p_mol_data
        ):
            r_charges = [d["charges"] for d in r_mol_data]
            p_charges = [d["charges"] for d in p_mol_data]

        smirks = build_smirks(r_smiles, p_smiles)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mapped = map_smirks(smirks)
        mapped_mols = transfer_atom_mapping(mapped, r_mols_from_xyz, p_mols_from_xyz)
        reactants = mapped_mols["reactants"]
        products = mapped_mols["products"]
    else:
        assert spec.reactant_smiles is not None
        assert spec.product_smiles is not None
        smirks = build_smirks(spec.reactant_smiles, spec.product_smiles)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mapped = map_smirks(smirks)
        mols = smirks_to_molecules(mapped)
        reactants = mols["reactants"]
        products = mols["products"]

    reactant_states = [_build_molecule_state(mol) for mol in reactants]
    product_states = [_build_molecule_state(mol) for mol in products]

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_r = pool.submit(_build_pairwise_params, reactants, preset_charges=r_charges)
        fut_p = pool.submit(_build_pairwise_params, products, preset_charges=p_charges)
        reactant_pp, reactant_atom_offsets = fut_r.result()
        product_pp, product_atom_offsets = fut_p.result()

    correspondence = get_atom_mapping_correspondence(reactants, products)

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
    return system_state, len(correspondence)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(
    params: np.ndarray, system_state: SystemState, alpha: float, beta: float
) -> Metrics:
    """Decompose params into geometric error, pairwise energy, and combined objective."""
    n_r = system_state.n_reactant_params
    r_params = params[:n_r].reshape(-1, 6)
    p_params = params[n_r:].reshape(-1, 6)

    # Reactant side
    r_coords_list = []
    total_r = sum(s.atom_count for s in system_state.reactant_states)
    combined_r = np.zeros((total_r, 3), dtype=np.float64)
    for i, state in enumerate(system_state.reactant_states):
        coords = apply_rigid_transform(
            state.initial_coords, state.centroid, r_params[i, :3], r_params[i, 3:]
        )
        r_coords_list.append(coords)
        off = system_state.reactant_atom_offsets[i]
        combined_r[off : off + state.atom_count] = coords

    # Product side
    p_coords_list = []
    total_p = sum(s.atom_count for s in system_state.product_states)
    combined_p = np.zeros((total_p, 3), dtype=np.float64)
    for i, state in enumerate(system_state.product_states):
        coords = apply_rigid_transform(
            state.initial_coords, state.centroid, p_params[i, :3], p_params[i, 3:]
        )
        p_coords_list.append(coords)
        off = system_state.product_atom_offsets[i]
        combined_p[off : off + state.atom_count] = coords

    r_energy = _compute_pairwise_energy(combined_r, system_state.reactant_params)
    p_energy = _compute_pairwise_energy(combined_p, system_state.product_params)
    pairwise = r_energy + p_energy

    geo_error = compute_geometric_error(r_coords_list, p_coords_list, system_state.atom_mapping)
    objective = alpha * pairwise + beta * geo_error

    return Metrics(geo_error=geo_error, pairwise_energy=pairwise, objective=objective)


# ---------------------------------------------------------------------------
# Random placement
# ---------------------------------------------------------------------------
def random_placement_trial(
    system_state: SystemState,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    box_size: float,
) -> tuple[Metrics, np.ndarray]:
    """Returns (metrics, params_vector) for one random placement."""
    n_total = system_state.n_reactant_params + system_state.n_product_params
    n_mols = n_total // 6
    params = np.zeros(n_total, dtype=np.float64)
    for m in range(n_mols):
        base = m * 6
        params[base : base + 3] = rng.uniform(-box_size, box_size, size=3)
        params[base + 3 : base + 6] = rng.uniform(-np.pi, np.pi, size=3)
    return compute_metrics(params, system_state, alpha, beta), params.copy()


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def run_optimization(
    system_state: SystemState,
    alpha: float,
    beta: float,
    max_iters: int = 500,
    ftol: float = 1e-12,
) -> tuple[np.ndarray, bool]:
    """Run L-BFGS-B optimization and return (params, converged)."""
    n_total = system_state.n_reactant_params + system_state.n_product_params
    initial_params = np.zeros(n_total, dtype=np.float64)

    n_mols = n_total // 6
    bounds: list[tuple[float | None, float | None]] = []
    for _ in range(n_mols):
        bounds.extend([(None, None)] * 3)
        bounds.extend([(-np.pi, np.pi)] * 3)

    result = minimize(
        _compute_rigid_body_energy,
        initial_params,
        args=(system_state, alpha, beta),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iters, "ftol": ftol},
    )
    return result.x, result.success


# ---------------------------------------------------------------------------
# XYZ output
# ---------------------------------------------------------------------------
def params_to_molecules(
    params: np.ndarray, system_state: SystemState
) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    """Apply rigid-body params to produce transformed molecule copies."""
    n_r = system_state.n_reactant_params
    r_params = params[:n_r].reshape(-1, 6)
    p_params = params[n_r:].reshape(-1, 6)

    reactants = []
    for i, state in enumerate(system_state.reactant_states):
        coords = apply_rigid_transform(
            state.initial_coords, state.centroid, r_params[i, :3], r_params[i, 3:]
        )
        mol = Chem.RWMol(state.mol)
        set_molecule_coordinates(mol, coords)
        reactants.append(mol.GetMol())

    products = []
    for i, state in enumerate(system_state.product_states):
        coords = apply_rigid_transform(
            state.initial_coords, state.centroid, p_params[i, :3], p_params[i, 3:]
        )
        mol = Chem.RWMol(state.mol)
        set_molecule_coordinates(mol, coords)
        products.append(mol.GetMol())

    return reactants, products


def _matched_xyz(mols: list[Chem.Mol], comment: str) -> str:
    """Combine molecules into a single XYZ block, atoms ordered by mapping number.

    Mapped atoms (map_num > 0) come first sorted by map_num, then unmapped.
    This ensures reactant and product files have corresponding atoms on the
    same line index.
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


def write_benchmark_xyz(
    output_dir: Path,
    label: str,
    reaction_name: str,
    reactants: list[Chem.Mol],
    products: list[Chem.Mol],
    metrics: Metrics,
) -> None:
    """Write reactant and product XYZ files for a benchmark result."""
    safe_name = (
        reaction_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("->", "to")
    )
    r_path = output_dir / f"{safe_name}_{label}_reactants.xyz"
    p_path = output_dir / f"{safe_name}_{label}_products.xyz"

    comment = (
        f"{label}; geo_error={metrics.geo_error:.4f}; "
        f"pairwise={metrics.pairwise_energy:.4f}; "
        f"objective={metrics.objective:.4f}"
    )

    r_path.write_text(_matched_xyz(reactants, comment))
    p_path.write_text(_matched_xyz(products, comment))
    print(f"  Wrote {r_path.name}, {p_path.name}")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_reaction_results(
    name: str,
    mode: str,
    n_reactants: int,
    n_products: int,
    n_mapped: int,
    random_metrics: list[Metrics],
    opt_metrics: Metrics,
    converged: bool,
    opt_time: float,
) -> None:
    """Print detailed results for a single reaction."""
    geo_vals = [m.geo_error for m in random_metrics]
    pw_vals = [m.pairwise_energy for m in random_metrics]
    obj_vals = [m.objective for m in random_metrics]

    mean_geo = statistics.mean(geo_vals)
    mean_pw = statistics.mean(pw_vals)
    mean_obj = statistics.mean(obj_vals)
    best_obj = min(obj_vals)

    print(f"  Reaction: {name}  [{mode.upper()}]")
    print(
        f"  Molecules: {n_reactants} reactant(s), {n_products} product(s)"
        f" | Mapped atoms: {n_mapped}"
    )
    print(f"  Converged: {converged} | Optimization time: {opt_time:.2f}s")
    print(f"  {'':20s} {'Geo Error (A^2)':>16s} {'Pairwise (kcal/mol)':>20s} {'Objective':>12s}")
    print(f"  {'-' * 72}")
    median_geo = statistics.median(geo_vals)
    median_pw = statistics.median(pw_vals)
    median_obj = statistics.median(obj_vals)

    print(f"  {'Random mean':20s} {mean_geo:16.2f} {mean_pw:20.2f} {mean_obj:12.2f}")
    print(f"  {'Random median':20s} {median_geo:16.2f} {median_pw:20.2f} {median_obj:12.2f}")
    print(f"  {'Random best':20s} {min(geo_vals):16.2f} {min(pw_vals):20.2f} {best_obj:12.2f}")
    print(
        f"  {'Random worst':20s} {max(geo_vals):16.2f} {max(pw_vals):20.2f} {max(obj_vals):12.2f}"
    )
    opt_geo = opt_metrics.geo_error
    opt_pw = opt_metrics.pairwise_energy
    opt_obj = opt_metrics.objective
    print(f"  {'Optimized':20s} {opt_geo:16.2f} {opt_pw:20.2f} {opt_obj:12.2f}")
    print(f"  {'-' * 72}")

    if opt_metrics.objective > 0 and mean_obj > 0:
        mean_imp = mean_obj / opt_metrics.objective
        best_imp = best_obj / opt_metrics.objective
        print(f"  Improvement vs mean: {mean_imp:.1f}x | vs best random: {best_imp:.1f}x")
    elif mean_obj > opt_metrics.objective:
        mean_red = mean_obj - opt_metrics.objective
        best_red = best_obj - opt_metrics.objective
        print(f"  Reduction vs mean: {mean_red:.1f} | vs best random: {best_red:.1f}")
    print()


def print_summary(results: list[dict]) -> None:
    """Print a summary table of all reactions."""
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    hdr = (
        f"  {'Reaction':<30s} {'Random Mean':>14s}"
        f" {'Random Best':>14s} {'Optimized':>12s} {'Improv.':>8s}"
    )
    print(hdr)
    print(f"  {'-' * 78}")
    for r in results:
        mean_obj = statistics.mean([m.objective for m in r["random"]])
        best_obj = min([m.objective for m in r["random"]])
        opt_obj = r["opt"].objective
        if opt_obj > 0 and mean_obj > 0:
            imp = f"{mean_obj / opt_obj:.1f}x"
        else:
            imp = f"{mean_obj - opt_obj:.1f} abs"
        print(f"  {r['name']:<30s} {mean_obj:14.2f} {best_obj:14.2f} {opt_obj:12.2f} {imp:>8s}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark optimization vs random placement")
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Force field weight (default: 1.0)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.1, help="Geometric error weight (default: 0.1)"
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of random trials (default: 50)"
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=6.0,
        help="Random translation range in Angstroms (default: 6.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--save-xyz",
        action="store_true",
        help="Write best-random and optimized XYZ files to molecules/benchmark_results/",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("Benchmark: Rigid-Body Optimization vs Random Placement")
    print("=" * 78)
    print(
        f"  alpha={args.alpha}, beta={args.beta}, trials={args.trials}, "
        f"box_size={args.box_size} A, seed={args.seed}"
    )
    print()

    rng = np.random.default_rng(args.seed)
    reactions = define_reactions()
    summary_data: list[dict] = []

    for i, spec in enumerate(reactions, 1):
        print(f"[{i}/{len(reactions)}] Preparing {spec.name}...")
        t0 = time.perf_counter()
        system_state, n_mapped = prepare_reaction(spec)
        t_prep = time.perf_counter() - t0
        print(f"  Prepared in {t_prep:.2f}s")

        # Random trials
        random_results = [
            random_placement_trial(system_state, args.alpha, args.beta, rng, args.box_size)
            for _ in range(args.trials)
        ]
        random_metrics = [r[0] for r in random_results]

        # Find best and median random trials
        best_idx = min(range(len(random_metrics)), key=lambda j: random_metrics[j].objective)
        best_random_params = random_results[best_idx][1]

        sorted_indices = sorted(
            range(len(random_metrics)), key=lambda j: random_metrics[j].objective
        )
        median_idx = sorted_indices[len(sorted_indices) // 2]
        median_random_params = random_results[median_idx][1]

        # Optimization
        t0 = time.perf_counter()
        opt_params, converged = run_optimization(system_state, args.alpha, args.beta)
        t_opt = time.perf_counter() - t0
        opt_metrics = compute_metrics(opt_params, system_state, args.alpha, args.beta)

        n_r = len(system_state.reactant_states)
        n_p = len(system_state.product_states)
        print_reaction_results(
            spec.name,
            spec.mode,
            n_r,
            n_p,
            n_mapped,
            random_metrics,
            opt_metrics,
            converged,
            t_opt,
        )

        # Write XYZ files if requested
        if args.save_xyz:
            output_dir = MOLECULES_DIR / "benchmark_results"
            output_dir.mkdir(parents=True, exist_ok=True)

            best_r, best_p = params_to_molecules(best_random_params, system_state)
            write_benchmark_xyz(
                output_dir, "random_best", spec.name, best_r, best_p, random_metrics[best_idx]
            )

            med_r, med_p = params_to_molecules(median_random_params, system_state)
            write_benchmark_xyz(
                output_dir, "random_median", spec.name, med_r, med_p, random_metrics[median_idx]
            )

            opt_r, opt_p = params_to_molecules(opt_params, system_state)
            write_benchmark_xyz(output_dir, "optimized", spec.name, opt_r, opt_p, opt_metrics)

        summary_data.append({"name": spec.name, "random": random_metrics, "opt": opt_metrics})

    print_summary(summary_data)


if __name__ == "__main__":
    main()
