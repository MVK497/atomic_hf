from __future__ import annotations

import argparse

from atomic_hf.benchmark import run_benchmark_sweep, write_benchmark_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the atomic_hf atomic-only RHF/UHF solvers.")
    parser.add_argument("--min-z", type=int, default=1, help="First atomic number in the sweep.")
    parser.add_argument("--max-z", type=int, default=102, help="Last atomic number in the sweep.")
    parser.add_argument("--basis", type=str, default="sto-3g", help="Basis set for all benchmark runs.")
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "rhf", "uhf"],
        help="Benchmark RHF, UHF, or choose automatically from the inferred spin.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip the PySCF reference calculation and only time the local implementation.",
    )
    parser.add_argument("--json-output", type=str, default=None, help="Optional path for a JSON benchmark report.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.min_z < 1 or args.max_z < args.min_z:
        parser.error("Require 1 <= --min-z <= --max-z.")

    summary = run_benchmark_sweep(
        min_z=args.min_z,
        max_z=args.max_z,
        basis=args.basis,
        method=args.method,
        compare_reference=not args.no_reference,
    )
    print("Atomic HF Benchmark")
    print(f"Range: Z = {summary['range'][0]} .. {summary['range'][1]}")
    print(f"Basis: {summary['basis']}")
    print(f"Method mode: {summary['method']}")
    print(f"Reference comparison: {summary['compare_reference']}")
    print(f"Successful cases: {summary['successful_cases']}")
    print(f"Failed cases: {summary['failed_cases']}")
    if summary["max_absolute_energy_error"] is not None:
        print(f"Max |E - E_ref|: {summary['max_absolute_energy_error']:.6e} Eh")
        print(f"Mean |E - E_ref|: {summary['mean_absolute_energy_error']:.6e} Eh")

    print()
    print("Per-element results")
    for entry in summary["entries"]:
        if entry["status"] == "ok":
            error_text = (
                f", |dE| = {entry['absolute_energy_error']:.3e} Eh"
                if entry["absolute_energy_error"] is not None
                else ""
            )
            print(
                f"  Z={entry['atomic_number']:>3d} {entry['symbol']:>2s}  "
                f"{entry['method']:>3s}  spin={entry['spin']:>2d}  "
                f"iter={entry['iterations']:>3d}  E={entry['energy']:.12f} Eh"
                f"{error_text}  t={entry['elapsed_seconds']:.2f}s"
            )
        else:
            print(
                f"  Z={entry['atomic_number']:>3d} {entry['symbol']:>2s}  "
                f"{entry['method']:>3s}  status=failed  t={entry['elapsed_seconds']:.2f}s  "
                f"reason={entry['message']}"
            )

    if args.json_output:
        write_benchmark_json(summary, args.json_output)
        print()
        print(f"JSON report written to {args.json_output}")


if __name__ == "__main__":
    main()
