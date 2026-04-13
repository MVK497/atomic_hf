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
        help="2S = N(alpha) - N(beta). Optional; if omitted, an atomic high-spin default is inferred.",
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
    parser.add_argument("--diis-start", type=int, default=None, help="Iteration to start DIIS extrapolation.")
    parser.add_argument("--guess", type=str, default="atom", choices=["atom", "core"], help="Initial guess strategy.")
    parser.add_argument("--damping", type=float, default=0.2, help="Density damping factor used in early cycles.")
    parser.add_argument("--damping-cycles", type=int, default=None, help="Number of early iterations to damp.")
    parser.add_argument("--level-shift", type=float, default=0.5, help="Virtual-space level shift in Hartree.")
    parser.add_argument("--show-history", action="store_true", help="Print SCF energy history.")
    return parser


def print_configuration_summary(summary: dict[str, object]) -> None:
    print("Atomic Information")
    print(f"Element: {summary['symbol']}")
    print(f"Atomic number: {summary['atomic_number']}")
    print(f"Electrons: {summary['electrons']}")
    print(f"Charge: {summary['charge']}")
    print(f"Estimated Aufbau configuration: {summary['estimated_configuration']}")
    print(f"Reference angular-shell profile: {summary['reference_l_shell_configuration']}")
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


def print_basis_engineering_summary(summary: dict[str, object]) -> None:
    print()
    print("Basis engineering summary")
    print(f"  Total shells: {summary['total_shells']}")
    print(f"  General-contracted shells: {summary['general_shells']}")
    print(f"  Segmented shells: {summary['segmented_shells']}")
    print(f"  Has general contractions: {summary['has_general_contractions']}")
    print(f"  Total primitives: {summary['total_primitives']}")
    print(f"  Total contracted radial functions: {summary['total_contracted_radial_functions']}")
    print(f"  Global contraction ratio: {summary['global_contraction_ratio']:.3f}")
    print("  Angular-momentum contraction structure:")
    for block in summary["angular_momentum_summary"]:
        print(
            f"    {block['label']}-type: shells = {block['shells']}, "
            f"general = {block['general_shells']}, segmented = {block['segmented_shells']}, "
            f"primitives = {block['primitives']}, contractions = {block['contracted_radial_functions']}, "
            f"max shell signature = {block['max_primitives_per_shell']}->{block['max_contractions_per_shell']}"
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


def print_two_electron_integral_summary(summary: dict[str, object]) -> None:
    print()
    print("One-center two-electron integral structure")
    print(f"  Threshold: {summary['threshold']:.1e}")
    print(f"  Total angular quartets: {summary['total_angular_quartets']}")
    print(f"  Active angular quartets: {summary['active_angular_quartets']}")
    print(f"  Inactive angular quartets: {summary['inactive_angular_quartets']}")
    print(f"  Active ratio: {summary['active_ratio']:.3f}")
    print(f"  Unique canonical quartet blocks: {summary['unique_canonical_blocks']}")
    print(f"  Active reduced radial pair blocks: {summary['active_reduced_radial_pair_blocks']}")
    print(f"  Quartet-to-pair compression ratio: {summary['reduced_pair_compression_ratio']:.3f}")
    print("  Dominant active quartets:")
    for quartet in summary["dominant_active_quartets"]:
        labels = ",".join(quartet["labels"])
        print(
            f"    ({labels}): ao = {quartet['ao_shape']}, "
            f"reduced = {quartet['reduced_radial_shape']}, "
            f"max|eri| = {quartet['max_abs']:.3e}, "
            f"||eri|| = {quartet['frobenius_norm']:.3e}"
        )
    print("  Dominant reduced radial pair blocks:")
    for pair_block in summary["dominant_reduced_radial_pairs"]:
        labels = ",".join(pair_block["labels"])
        print(
            f"    ({labels}): reduced = {pair_block['reduced_radial_shape']}, "
            f"max|J| = {pair_block['coulomb_max_abs']:.3e}, "
            f"max|K| = {pair_block['exchange_max_abs']:.3e}, "
            f"||J|| = {pair_block['coulomb_frobenius_norm']:.3e}, "
            f"||K|| = {pair_block['exchange_frobenius_norm']:.3e}"
        )


def print_symmetry_blocks(blocks: list[dict[str, int | str]]) -> None:
    print()
    print("Symmetry-blocked diagonalization")
    for block in blocks:
        print(
            f"  {block['label']}-block: degeneracy = {block['degeneracy']}, "
            f"AO count = {block['ao_count']}, radial functions = {block['radial_functions']}"
        )


def print_spin_symmetry_blocks(
    blocks_alpha: list[dict[str, int | str]],
    blocks_beta: list[dict[str, int | str]],
) -> None:
    print()
    print("Symmetry-blocked UHF diagonalization")
    print("  Alpha blocks:")
    for block in blocks_alpha:
        print(
            f"    {block['label']}-block: degeneracy = {block['degeneracy']}, "
            f"AO count = {block['ao_count']}, radial functions = {block['radial_functions']}"
        )
    print("  Beta blocks:")
    for block in blocks_beta:
        print(
            f"    {block['label']}-block: degeneracy = {block['degeneracy']}, "
            f"AO count = {block['ao_count']}, radial functions = {block['radial_functions']}"
        )


def print_rhf_result(result: AtomicRHFResult, show_history: bool) -> None:
    print()
    print("Atomic RHF Result")
    print(f"Basis: {result.basis}")
    print(f"Spin (2S): {result.spin}")
    print(f"Spherical-average atomic solver: {result.spherical_average}")
    print(f"Initial guess: {result.initial_guess}")
    print(f"SCF iterations: {result.iterations}")
    print(f"Total energy: {result.energy:.12f} Eh")
    print(
        "Stabilization: "
        f"DIIS={result.stabilization_summary['use_diis']}, "
        f"diis_start={result.stabilization_summary['diis_start_cycle']}, "
        f"damping={result.stabilization_summary['damping_factor']}, "
        f"level_shift={result.stabilization_summary['level_shift']}"
    )
    print(
        "Fock builder: "
        f"{result.fock_build_summary['builder']} "
        f"(reduced pair blocks={result.fock_build_summary['active_reduced_radial_pair_blocks']})"
    )
    print()
    print("MO occupations / energies")
    for index, (occ, energy) in enumerate(zip(result.mo_occupations, result.orbital_energies, strict=True), start=1):
        print(f"  MO {index:>2d}: occ = {occ:>4.1f}   eps = {energy: .12f} Eh")

    print_basis_summary(result.basis_summary)
    print_basis_engineering_summary(result.basis_engineering_summary)
    print_symmetry_blocks(result.symmetry_blocks)
    print_one_center_integral_summary(result.one_center_integral_summary)
    print_two_electron_integral_summary(result.two_electron_integral_summary)

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
    print("Blocked atomic solver: True")
    print(f"Initial guess: {result.initial_guess}")
    print(f"Alpha electrons: {result.nalpha}")
    print(f"Beta electrons: {result.nbeta}")
    print(f"SCF iterations: {result.iterations}")
    print(f"Total energy: {result.energy:.12f} Eh")
    print(
        "Stabilization: "
        f"DIIS={result.stabilization_summary['use_diis']}, "
        f"diis_start={result.stabilization_summary['diis_start_cycle']}, "
        f"damping={result.stabilization_summary['damping_factor']}, "
        f"level_shift={result.stabilization_summary['level_shift']}"
    )
    print(
        "Fock builder: "
        f"{result.fock_build_summary['builder']} "
        f"(reduced pair blocks={result.fock_build_summary['active_reduced_radial_pair_blocks']}, "
        f"active quartets={result.fock_build_summary['active_angular_quartets']}, "
        f"unique blocks={result.fock_build_summary['unique_canonical_blocks']})"
    )
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
    print_basis_engineering_summary(result.basis_engineering_summary)
    print_spin_symmetry_blocks(result.symmetry_blocks_alpha, result.symmetry_blocks_beta)
    print_one_center_integral_summary(result.one_center_integral_summary)
    print_two_electron_integral_summary(result.two_electron_integral_summary)

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
    if args.method == "rhf":
        result = run_atomic_rhf(
            spec,
            max_iter=args.max_iter,
            e_tol=args.energy_tol,
            use_diis=not args.no_diis,
            diis_space=args.diis_space,
            initial_guess=args.guess,
            damping_factor=args.damping,
            damping_cycles=args.damping_cycles if args.damping_cycles is not None else 4,
            level_shift=args.level_shift,
            diis_start_cycle=args.diis_start if args.diis_start is not None else 2,
        )
        print_rhf_result(result, args.show_history)
        return

    result = run_atomic_uhf(
        spec,
        max_iter=args.max_iter,
        e_tol=args.energy_tol,
        use_diis=not args.no_diis,
        diis_space=args.diis_space,
        initial_guess=args.guess,
        damping_factor=args.damping,
        damping_cycles=args.damping_cycles if args.damping_cycles is not None else 6,
        level_shift=args.level_shift,
        diis_start_cycle=args.diis_start if args.diis_start is not None else 3,
    )
    print_uhf_result(result, args.show_history)
