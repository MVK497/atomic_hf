from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import subprocess
import sys
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

LIGHTWEIGHT_BENCHMARK_BASIS_FALLBACKS = (
    "minao",
    "lanl2dz",
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


def _fallback_candidates_for_requested_basis(basis: str) -> tuple[str, ...]:
    requested_basis = str(basis).lower()
    if requested_basis == "sto-3g":
        return LIGHTWEIGHT_BENCHMARK_BASIS_FALLBACKS
    return BENCHMARK_BASIS_FALLBACKS


def _candidate_benchmark_bases(symbol: str, basis: str) -> list[str]:
    requested_basis = str(basis)
    if _basis_is_available(symbol, requested_basis):
        return [requested_basis]

    # When the requested benchmark basis is unavailable, fall back conservatively.
    # For lightweight sweeps such as sto-3g, avoid escalating into very heavy bases.
    candidates = [
        candidate
        for candidate in _fallback_candidates_for_requested_basis(requested_basis)
        if candidate != requested_basis and _basis_is_available(symbol, candidate)
    ]
    return candidates[:1]


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


_ENTRY_RUNNER = r"""
import json
import sys
from atomic_hf import AtomicSpec, run_atomic_rhf, run_atomic_uhf
from atomic_hf.benchmark import _reference_atomic_energy

symbol = sys.argv[1]
basis = sys.argv[2]
charge = int(sys.argv[3])
spin_arg = sys.argv[4]
method = sys.argv[5]
compare_reference = bool(int(sys.argv[6]))
spin = None if spin_arg == "None" else int(spin_arg)
spec = AtomicSpec(symbol=symbol, basis=basis, charge=charge, spin=spin)

try:
    if method == "rhf":
        result = run_atomic_rhf(spec, with_analysis=False)
        payload = {
            "symbol": result.symbol,
            "atomic_number": result.atomic_number,
            "spin": result.spin,
            "basis": result.basis,
            "energy": float(result.energy),
            "iterations": int(result.iterations),
        }
    else:
        result = run_atomic_uhf(spec, with_analysis=False)
        payload = {
            "symbol": result.symbol,
            "atomic_number": result.atomic_number,
            "spin": result.spin,
            "basis": result.basis,
            "energy": float(result.energy),
            "iterations": int(result.iterations),
        }
    if compare_reference:
        payload["reference_energy"] = float(_reference_atomic_energy(spec, method))
    else:
        payload["reference_energy"] = None
    payload["status"] = "ok"
    payload["message"] = None
except Exception as exc:
    payload = {
        "symbol": symbol,
        "atomic_number": None,
        "spin": spin,
        "basis": basis,
        "energy": None,
        "iterations": None,
        "reference_energy": None,
        "status": "failed",
        "message": str(exc),
    }
print(json.dumps(payload))
"""


def _run_entry_subprocess(
    spec: AtomicSpec,
    method: str,
    compare_reference: bool,
    timeout_seconds: int | None,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-c",
        _ENTRY_RUNNER,
        spec.symbol,
        str(spec.basis),
        str(spec.charge),
        "None" if spec.spin is None else str(spec.spin),
        method,
        "1" if compare_reference else "0",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=timeout_seconds,
        check=False,
    )
    stdout = completed.stdout.strip().splitlines()
    if not stdout:
        raise RuntimeError(completed.stderr.strip() or "Benchmark subprocess produced no output.")
    return json.loads(stdout[-1])


def run_benchmark_entry(
    spec: AtomicSpec,
    method: str = "auto",
    compare_reference: bool = True,
    timeout_seconds: int | None = 60,
) -> BenchmarkEntry:
    start = time.perf_counter()
    selected_method = method
    candidate_bases = _candidate_benchmark_bases(spec.symbol, str(spec.basis))
    last_error: Exception | None = None
    last_basis = str(spec.basis)

    if not candidate_bases:
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
            elapsed_seconds=time.perf_counter() - start,
            status="failed",
            message=f"Requested benchmark basis {spec.basis} is unavailable for {spec.symbol}.",
        )

    for benchmark_basis in candidate_bases:
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
            payload = _run_entry_subprocess(
                benchmark_spec,
                selected_method,
                compare_reference=compare_reference,
                timeout_seconds=timeout_seconds,
            )
            elapsed = time.perf_counter() - start
            if payload["status"] != "ok":
                raise RuntimeError(str(payload["message"]))

            reference_energy = payload["reference_energy"]
            error = None if reference_energy is None else abs(float(payload["energy"]) - float(reference_energy))
            return BenchmarkEntry(
                symbol=str(payload["symbol"]),
                atomic_number=int(payload["atomic_number"]),
                method=selected_method,
                spin=int(payload["spin"]),
                basis=str(payload["basis"]),
                energy=float(payload["energy"]),
                reference_energy=reference_energy,
                absolute_energy_error=error,
                iterations=int(payload["iterations"]),
                elapsed_seconds=elapsed,
                status="ok",
                message=None,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - long-running entries should not block the sweep
            last_error = TimeoutError(f"Benchmark entry exceeded the {timeout_seconds}s timeout.")
            last_basis = benchmark_basis
            continue
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
    timeout_seconds: int | None = 60,
) -> dict[str, object]:
    entries: list[BenchmarkEntry] = []
    for atomic_number in range(min_z, max_z + 1):
        symbol = elements.ELEMENTS[atomic_number]
        spec = AtomicSpec(symbol=symbol, basis=basis)
        entries.append(
            run_benchmark_entry(
                spec,
                method=method,
                compare_reference=compare_reference,
                timeout_seconds=timeout_seconds,
            )
        )

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


def load_benchmark_json(input_path: str | Path) -> dict[str, object]:
    path = Path(input_path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_benchmark_sweep_checkpointed(
    output_path: str | Path,
    min_z: int = 1,
    max_z: int = 102,
    basis: str = "sto-3g",
    method: str = "auto",
    compare_reference: bool = True,
    resume: bool = True,
    timeout_seconds: int | None = 60,
) -> dict[str, object]:
    path = Path(output_path)
    if resume and path.exists():
        summary = load_benchmark_json(path)
        entries = list(summary.get("entries", []))
        done = {int(entry["atomic_number"]) for entry in entries}
    else:
        entries = []
        done = set()
        summary = {
            "range": [min_z, max_z],
            "basis": basis,
            "method": method,
            "compare_reference": compare_reference,
            "successful_cases": 0,
            "failed_cases": 0,
            "max_absolute_energy_error": None,
            "mean_absolute_energy_error": None,
            "entries": entries,
        }

    for atomic_number in range(min_z, max_z + 1):
        if atomic_number in done:
            continue
        symbol = elements.ELEMENTS[atomic_number]
        spec = AtomicSpec(symbol=symbol, basis=basis)
        entry = run_benchmark_entry(
            spec,
            method=method,
            compare_reference=compare_reference,
            timeout_seconds=timeout_seconds,
        )
        entries.append(asdict(entry))

        successful = [item for item in entries if item["status"] == "ok"]
        failed = [item for item in entries if item["status"] != "ok"]
        errors = [item["absolute_energy_error"] for item in successful if item["absolute_energy_error"] is not None]
        summary.update(
            {
                "successful_cases": len(successful),
                "failed_cases": len(failed),
                "max_absolute_energy_error": max(errors) if errors else None,
                "mean_absolute_energy_error": (sum(errors) / len(errors)) if errors else None,
                "entries": entries,
                "last_completed_atomic_number": atomic_number,
            }
        )
        write_benchmark_json(summary, path)

    return summary


def write_benchmark_json(summary: dict[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
