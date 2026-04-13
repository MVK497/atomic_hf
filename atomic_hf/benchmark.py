from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from pathlib import Path

from pyscf import gto, scf
from pyscf.data import elements
from pyscf.scf import atom_hf as pyscf_atom_hf

from .atom import AtomicSpec, atomic_number, build_atomic_molecule, resolve_spin
from .rhf import run_atomic_rhf
from .uhf import run_atomic_uhf


@dataclass
class BenchmarkEntry:
    symbol: str
    atomic_number: int
    method: str
    spin: int
    basis: str
    energy: float | None
    reference_energy: float | None
    absolute_energy_error: float | None
    iterations: int | None
    elapsed_seconds: float
    status: str
    message: str | None


BENCHMARK_BASIS_FALLBACKS = (
    "minao",
    "lanl2dz",
    "def2-svp",
    "cc-pvdz-dk",
    "dyall_v2z",
)


def _basis_is_available(symbol: str, basis: str) -> bool:
    try:
        gto.basis.load(basis, symbol)
        return True
    except Exception:
        return False


def _resolve_benchmark_basis(symbol: str, basis: str) -> str:
    candidates = (basis,) + tuple(candidate for candidate in BENCHMARK_BASIS_FALLBACKS if candidate != basis)
    for candidate in candidates:
        if _basis_is_available(symbol, candidate):
            return candidate
    raise ValueError(f"No available benchmark basis was found for {symbol}.")


def _reference_atomic_energy(spec: AtomicSpec, method: str) -> float:
    mol = build_atomic_molecule(spec)
    if method == "rhf":
        if mol.nelectron == 1:
            reference = pyscf_atom_hf.AtomHF1e(mol)
        else:
            reference = pyscf_atom_hf.AtomSphAverageRHF(mol)
    else:
        reference = scf.UHF(mol)
        reference.init_guess = "atom"
    reference.verbose = 0
    reference.conv_tol = 1.0e-10
    reference.diis_space = 6
    return float(reference.kernel())


def run_benchmark_entry(
    spec: AtomicSpec,
    method: str = "auto",
    compare_reference: bool = True,
) -> BenchmarkEntry:
    start = time.perf_counter()
    selected_method = method
    candidate_bases = [str(spec.basis)] + [basis for basis in BENCHMARK_BASIS_FALLBACKS if basis != str(spec.basis)]
    last_error: Exception | None = None
    last_basis = str(spec.basis)
    for benchmark_basis in candidate_bases:
        if not _basis_is_available(spec.symbol, benchmark_basis):
            continue
        benchmark_spec = AtomicSpec(
            symbol=spec.symbol,
            basis=benchmark_basis,
            charge=spec.charge,
            spin=spec.spin,
            title=spec.title,
        )
        try:
            selected_method = method
            if selected_method == "auto":
                selected_method = "rhf" if resolve_spin(benchmark_spec) == 0 else "uhf"

            if selected_method == "rhf":
                result = run_atomic_rhf(benchmark_spec)
            else:
                result = run_atomic_uhf(benchmark_spec)
            elapsed = time.perf_counter() - start

            reference_energy = _reference_atomic_energy(benchmark_spec, selected_method) if compare_reference else None
            error = None if reference_energy is None else abs(result.energy - reference_energy)
            return BenchmarkEntry(
                symbol=result.symbol,
                atomic_number=result.atomic_number,
                method=selected_method,
                spin=result.spin,
                basis=result.basis,
                energy=float(result.energy),
                reference_energy=reference_energy,
                absolute_energy_error=error,
                iterations=int(result.iterations),
                elapsed_seconds=elapsed,
                status="ok",
                message=None,
            )
        except Exception as exc:  # pragma: no cover - benchmark tool should keep sweeping after failures
            last_error = exc
            last_basis = benchmark_basis
            continue

    elapsed = time.perf_counter() - start
    return BenchmarkEntry(
        symbol=spec.symbol,
        atomic_number=atomic_number(spec.symbol),
        method=selected_method,
        spin=resolve_spin(spec),
        basis=last_basis,
        energy=None,
        reference_energy=None,
        absolute_energy_error=None,
        iterations=None,
        elapsed_seconds=elapsed,
        status="failed",
        message=str(last_error) if last_error is not None else "No compatible basis was found.",
    )


def run_benchmark_sweep(
    min_z: int = 1,
    max_z: int = 102,
    basis: str = "sto-3g",
    method: str = "auto",
    compare_reference: bool = True,
) -> dict[str, object]:
    entries: list[BenchmarkEntry] = []
    for atomic_number in range(min_z, max_z + 1):
        symbol = elements.ELEMENTS[atomic_number]
        spec = AtomicSpec(symbol=symbol, basis=basis)
        entries.append(run_benchmark_entry(spec, method=method, compare_reference=compare_reference))

    successful = [entry for entry in entries if entry.status == "ok"]
    failed = [entry for entry in entries if entry.status != "ok"]
    errors = [entry.absolute_energy_error for entry in successful if entry.absolute_energy_error is not None]
    return {
        "range": [min_z, max_z],
        "basis": basis,
        "method": method,
        "compare_reference": compare_reference,
        "successful_cases": len(successful),
        "failed_cases": len(failed),
        "max_absolute_energy_error": max(errors) if errors else None,
        "mean_absolute_energy_error": (sum(errors) / len(errors)) if errors else None,
        "entries": [asdict(entry) for entry in entries],
    }


def write_benchmark_json(summary: dict[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
