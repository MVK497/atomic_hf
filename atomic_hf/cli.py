from __future__ import annotations

import argparse

from .atom import AtomicSpec, build_configuration_summary, resolve_spin
from .rhf import AtomicRHFResult, run_atomic_rhf
from .uhf import AtomicUHFResult, run_atomic_uhf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomic-only RHF/UHF teaching program based on PySCF/libcint."
    )
    parser.add_argument("symbol", type=str, help="Atomic symbol, e.g. He, O, Fe.")
    parser.add_argument("--basis", type=str, default="sto-3g", help="Any basis name supported by PySCF.")
    parser.add_argument("--charge", type=int, default=0, help="Atomic charge.")
    parser.add_argument(
        "--spin",
        type=int,
        default=None,
        help="2S = N(alpha) - N(beta). Required for UHF. Optional for RHF closed-shell atoms.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rhf",
        choices=["rhf", "uhf"],
        help="Atomic Hartree-Fock method.",
    )
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum SCF iterations.")
    parser.add_argument("--energy-tol", type=float, default=1.0e-10, help="SCF energy tolerance.")
    parser.add_argument("--no-diis", action="store_true", help="Disable DIIS.")
    parser.add_argument("--diis-space", type=int, default=6, help="Number of DIIS vectors.")
    parser.add_argument("--show-history", action="store_true", help="Print SCF energy history.")
    return parser


def print_configuration_summary(summary: dict[str, object]) -> None:
    print("Atomic Information")
    print(f"Element: {summary['symbol']}")
    print(f"Atomic number: {summary['atomic_number']}")
    print(f"Electrons: {summary['electrons']}")
    print(f"Charge: {summary['charge']}")
    print(f"Estimated Aufbau configuration: {summary['estimated_configuration']}")
    print(f"Estimated unpaired electrons: {summary['estimated_unpaired_electrons']}")


def print_basis_summary(basis_summary: list[dict[str, int | str]]) -> None:
    print()
    print("Basis-shell summary")
    for entry in basis_summary:
        print(
            f"  {entry['label']}-type: shells = {entry['shells']}, "
            f"contracted radial functions = {entry['contracted_functions']}, "
            f"primitives = {entry['primitive_functions']}, "
            f"AOs = {entry['aos']}"
        )


def print_one_center_integral_summary(summary: dict[str, object]) -> None:
    print()
    print("One-center integral reduction")
    print(f"  Full AO dimension: {summary['full_ao_dimension']}")
    print(f"  Reduced radial dimension: {summary['reduced_radial_dimension']}")
    print(f"  One-electron compression ratio: {summary['one_electron_compression_ratio']:.3f}")
    print("  Angular-momentum blocks:")
    for block in summary["block_summaries"]:
        print(
            f"    {block['label']}-block: ao = {block['ao_count']}, radial = {block['radial_functions']}, "
            f"degeneracy = {block['degeneracy']}, "
            f"max|S_offdiag| = {block['overlap_offdiag_max_abs']:.3e}, "
            f"max|H_offdiag| = {block['hcore_offdiag_max_abs']:.3e}"
        )


def print_symmetry_blocks(blocks: list[dict[str, int | str]]) -> None:
    print()
    print("Symmetry-blocked diagonalization")
    for block in blocks:
        print(
            f"  {block['label']}-block: degeneracy = {block['degeneracy']}, "
            f"AO count = {block['ao_count']}, radial functions = {block['radial_functions']}"
        )


def print_rhf_result(result: AtomicRHFResult, show_history: bool) -> None:
    print()
    print("Atomic RHF Result")
    print(f"Basis: {result.basis}")
    print(f"Spin (2S): {result.spin}")
    print(f"Spherical-average atomic solver: {result.spherical_average}")
    print(f"SCF iterations: {result.iterations}")
    print(f"Total energy: {result.energy:.12f} Eh")
    print()
    print("MO occupations / energies")
    for index, (occ, energy) in enumerate(zip(result.mo_occupations, result.orbital_energies, strict=True), start=1):
        print(f"  MO {index:>2d}: occ = {occ:>4.1f}   eps = {energy: .12f} Eh")

    print_basis_summary(result.basis_summary)
    print_symmetry_blocks(result.symmetry_blocks)
    print_one_center_integral_summary(result.one_center_integral_summary)

    if show_history:
        print()
        print("SCF history")
        for iteration, energy in enumerate(result.history, start=1):
            print(f"  Iter {iteration:>2d}: {energy:.12f} Eh")


def print_uhf_result(result: AtomicUHFResult, show_history: bool) -> None:
    print()
    print("Atomic UHF Result")
    print(f"Basis: {result.basis}")
    print(f"Spin (2S): {result.spin}")
    print(f"Alpha electrons: {result.nalpha}")
    print(f"Beta electrons: {result.nbeta}")
    print(f"SCF iterations: {result.iterations}")
    print(f"Total energy: {result.energy:.12f} Eh")
    print(f"<S^2>: {result.s2:.12f}")
    print(f"Expected <S^2>: {result.expected_s2:.12f}")
    print(f"Spin contamination: {result.spin_contamination:.12f}")
    print()
    print("Alpha occupations / energies")
    for index, (occ, energy) in enumerate(
        zip(result.mo_occupations_alpha, result.orbital_energies_alpha, strict=True),
        start=1,
    ):
        print(f"  aMO {index:>2d}: occ = {occ:>4.1f}   eps = {energy: .12f} Eh")
    print()
    print("Beta occupations / energies")
    for index, (occ, energy) in enumerate(
        zip(result.mo_occupations_beta, result.orbital_energies_beta, strict=True),
        start=1,
    ):
        print(f"  bMO {index:>2d}: occ = {occ:>4.1f}   eps = {energy: .12f} Eh")

    print_basis_summary(result.basis_summary)
    print_one_center_integral_summary(result.one_center_integral_summary)

    if show_history:
        print()
        print("SCF history")
        for iteration, energy in enumerate(result.history, start=1):
            print(f"  Iter {iteration:>2d}: {energy:.12f} Eh")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.diis_space < 2:
        parser.error("--diis-space must be at least 2.")

    spec = AtomicSpec(
        symbol=args.symbol,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        title=f"{args.symbol} atom",
    )
    configuration_summary = build_configuration_summary(spec)
    print_configuration_summary(configuration_summary)
    resolved_spin = resolve_spin(spec)
    print(f"Resolved spin (2S): {resolved_spin}")
    if args.spin is None and args.method == "uhf":
        print("Spin was not provided explicitly; UHF requires a deliberate state choice, so this run will stop.")

    if args.method == "rhf":
        result = run_atomic_rhf(
            spec,
            max_iter=args.max_iter,
            e_tol=args.energy_tol,
            use_diis=not args.no_diis,
            diis_space=args.diis_space,
        )
        print_rhf_result(result, args.show_history)
        return

    result = run_atomic_uhf(
        spec,
        max_iter=args.max_iter,
        e_tol=args.energy_tol,
        use_diis=not args.no_diis,
        diis_space=args.diis_space,
    )
    print_uhf_result(result, args.show_history)
