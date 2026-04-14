from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from atomic_hf.benchmark import run_benchmark_sweep_checkpointed


@dataclass(frozen=True)
class BenchmarkStage:
    name: str
    min_z: int
    max_z: int
    timeout_seconds: int
    description: str


STAGES: dict[str, BenchmarkStage] = {
    "smoke": BenchmarkStage(
        name="smoke",
        min_z=1,
        max_z=10,
        timeout_seconds=20,
        description="Quick sanity-check benchmark for light elements.",
    ),
    "first20": BenchmarkStage(
        name="first20",
        min_z=1,
        max_z=20,
        timeout_seconds=30,
        description="Stable first-row-through-calcium sweep.",
    ),
    "transition3d": BenchmarkStage(
        name="transition3d",
        min_z=21,
        max_z=30,
        timeout_seconds=180,
        description="Dedicated 3d transition-metal sweep with a larger timeout budget.",
    ),
    "main54": BenchmarkStage(
        name="main54",
        min_z=1,
        max_z=54,
        timeout_seconds=120,
        description="Moderate full-block benchmark through xenon.",
    ),
    "full": BenchmarkStage(
        name="full",
        min_z=1,
        max_z=102,
        timeout_seconds=180,
        description="Full benchmark for the first 102 elements.",
    ),
}


def _safe_tag(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_").lower()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run named atomic_hf benchmark stages with checkpointed JSON output."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="first20",
        choices=sorted(STAGES),
        help="Named benchmark stage to run.",
    )
    parser.add_argument("--basis", type=str, default="sto-3g", help="Primary basis set for the sweep.")
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "rhf", "uhf"],
        help="Benchmark RHF, UHF, or choose automatically from the inferred spin.",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Also compute PySCF reference energies. This is slower and usually unnecessary for long sweeps.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Override the preset per-entry timeout in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks",
        help="Directory for the JSON checkpoint/report output.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit JSON output path. If omitted, a stage-based filename is generated.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a fresh run instead of resuming from an existing checkpoint file.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print the available benchmark stages and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_stages:
        print("Available benchmark stages")
        for stage in STAGES.values():
            print(
                f"  {stage.name:>12s}: Z={stage.min_z:>3d}..{stage.max_z:<3d}  "
                f"timeout={stage.timeout_seconds:>3d}s  {stage.description}"
            )
        return

    stage = STAGES[args.stage]
    timeout_seconds = args.timeout if args.timeout is not None else stage.timeout_seconds

    if args.output is None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_tag = "ref" if args.reference else "noref"
        filename = f"{stage.name}_{_safe_tag(args.basis)}_{ref_tag}.json"
        output_path = output_dir / filename
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Atomic HF Benchmark Suite")
    print(f"Stage: {stage.name}")
    print(f"Description: {stage.description}")
    print(f"Range: Z = {stage.min_z} .. {stage.max_z}")
    print(f"Basis: {args.basis}")
    print(f"Method mode: {args.method}")
    print(f"Reference comparison: {args.reference}")
    print(f"Entry timeout: {timeout_seconds}s")
    print(f"Resume mode: {not args.no_resume}")
    print(f"JSON output: {output_path}")
    print()

    summary = run_benchmark_sweep_checkpointed(
        output_path=output_path,
        min_z=stage.min_z,
        max_z=stage.max_z,
        basis=args.basis,
        method=args.method,
        compare_reference=args.reference,
        resume=not args.no_resume,
        timeout_seconds=timeout_seconds,
    )

    print("Benchmark finished")
    print(f"Successful cases: {summary['successful_cases']}")
    print(f"Failed cases: {summary['failed_cases']}")
    print(f"Last completed atomic number: {summary.get('last_completed_atomic_number')}")
    if summary["max_absolute_energy_error"] is not None:
        print(f"Max |E - E_ref|: {summary['max_absolute_energy_error']:.6e} Eh")
        print(f"Mean |E - E_ref|: {summary['mean_absolute_energy_error']:.6e} Eh")
    print(f"Saved JSON checkpoint/report: {output_path}")


if __name__ == "__main__":
    main()
