from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from pyscf import gto
from pyscf.data import elements


ANGULAR_LABELS = "spdfghi"
ANGULAR_LABEL_TO_VALUE = {label: index for index, label in enumerate(ANGULAR_LABELS)}
AUFBAU_ORDER: list[tuple[int, str, int]] = [
    (1, "s", 2),
    (2, "s", 2),
    (2, "p", 6),
    (3, "s", 2),
    (3, "p", 6),
    (4, "s", 2),
    (3, "d", 10),
    (4, "p", 6),
    (5, "s", 2),
    (4, "d", 10),
    (5, "p", 6),
    (6, "s", 2),
    (4, "f", 14),
    (5, "d", 10),
    (6, "p", 6),
    (7, "s", 2),
    (5, "f", 14),
    (6, "d", 10),
    (7, "p", 6),
]


@dataclass(frozen=True)
class AtomicSpec:
    symbol: str
    basis: object = "sto-3g"
    charge: int = 0
    spin: int | None = None
    title: str | None = None


@dataclass(frozen=True)
class SubshellOccupation:
    n: int
    l_label: str
    occupancy: int
    capacity: int
    unpaired: int

    @property
    def label(self) -> str:
        return f"{self.n}{self.l_label}"


def canonical_symbol(symbol: str) -> str:
    try:
        return gto.mole._std_symbol(symbol.strip())
    except RuntimeError as exc:
        raise ValueError(f"Unknown element symbol '{symbol}'.") from exc


def atomic_number(symbol: str) -> int:
    z = int(gto.charge(canonical_symbol(symbol)))
    if z <= 0:
        raise ValueError(f"Unknown atomic number for symbol '{symbol}'.")
    return z


def electron_count(symbol: str, charge: int = 0) -> int:
    nelec = atomic_number(symbol) - charge
    if nelec <= 0:
        raise ValueError("Atomic HF currently requires at least one electron.")
    return nelec


def _subshell_unpaired_electrons(occupancy: int, capacity: int) -> int:
    degeneracy = capacity // 2
    return occupancy if occupancy <= degeneracy else capacity - occupancy


def _build_aufbau_subshell_occupations_for_electron_count(nelec: int) -> list[SubshellOccupation]:
    remaining = nelec
    occupations: list[SubshellOccupation] = []
    for n, l_label, capacity in AUFBAU_ORDER:
        if remaining <= 0:
            break
        occ = min(remaining, capacity)
        occupations.append(
            SubshellOccupation(
                n=n,
                l_label=l_label,
                occupancy=occ,
                capacity=capacity,
                unpaired=_subshell_unpaired_electrons(occ, capacity),
            )
        )
        remaining -= occ
    if remaining != 0:
        raise ValueError("Electron configuration exceeded the supported Aufbau table.")
    return occupations


def _aggregate_subshell_occupations_by_l(occupations: list[SubshellOccupation]) -> dict[str, int]:
    totals = {label: 0 for label in ANGULAR_LABELS[:4]}
    for orbital in occupations:
        if orbital.l_label in totals:
            totals[orbital.l_label] += orbital.occupancy
    return totals


def _reference_angular_occupations(nelec: int) -> dict[str, int]:
    if nelec >= len(elements.CONFIGURATION):
        raise ValueError(f"No PySCF atomic configuration table is available for {nelec} electrons.")
    reference = elements.CONFIGURATION[nelec]
    return {
        ANGULAR_LABELS[l_value]: int(reference[l_value])
        for l_value in range(min(4, len(reference)))
    }


def _reconcile_occupations_with_reference_l_totals(
    occupations: list[SubshellOccupation],
    nelec: int,
) -> list[SubshellOccupation]:
    mutable = [
        {
            "n": orbital.n,
            "l_label": orbital.l_label,
            "occupancy": orbital.occupancy,
            "capacity": orbital.capacity,
        }
        for orbital in occupations
    ]
    target = _reference_angular_occupations(nelec)

    def current_totals() -> dict[str, int]:
        totals = {label: 0 for label in target}
        for orbital in mutable:
            l_label = orbital["l_label"]
            if l_label in totals:
                totals[l_label] += int(orbital["occupancy"])
        return totals

    for _ in range(64):
        totals = current_totals()
        if totals == target:
            break

        deficits = {label: target[label] - totals[label] for label in target if target[label] > totals[label]}
        surpluses = {label: totals[label] - target[label] for label in target if totals[label] > target[label]}
        if not deficits and not surpluses:
            break
        if not deficits or not surpluses:
            break

        donor_index = next(
            (
                index
                for index in range(len(mutable) - 1, -1, -1)
                if mutable[index]["l_label"] in surpluses and mutable[index]["occupancy"] > 0
            ),
            None,
        )
        acceptor_index = next(
            (
                index
                for index, orbital in enumerate(mutable)
                if orbital["l_label"] in deficits and orbital["occupancy"] < orbital["capacity"]
            ),
            None,
        )
        if donor_index is None or acceptor_index is None:
            break

        donor = mutable[donor_index]
        acceptor = mutable[acceptor_index]
        move = min(
            surpluses[donor["l_label"]],
            deficits[acceptor["l_label"]],
            int(donor["occupancy"]),
            int(acceptor["capacity"] - acceptor["occupancy"]),
        )
        if move <= 0:
            break

        donor["occupancy"] -= move
        acceptor["occupancy"] += move

    reconciled = [
        SubshellOccupation(
            n=int(orbital["n"]),
            l_label=str(orbital["l_label"]),
            occupancy=int(orbital["occupancy"]),
            capacity=int(orbital["capacity"]),
            unpaired=_subshell_unpaired_electrons(int(orbital["occupancy"]), int(orbital["capacity"])),
        )
        for orbital in mutable
        if int(orbital["occupancy"]) > 0
    ]
    return reconciled


def build_subshell_occupations(symbol: str, charge: int = 0) -> list[SubshellOccupation]:
    nelec = electron_count(symbol, charge)
    occupations = _build_aufbau_subshell_occupations_for_electron_count(nelec)
    return _reconcile_occupations_with_reference_l_totals(occupations, nelec)


def resolve_spin(spec: AtomicSpec) -> int:
    if spec.spin is not None:
        return spec.spin
    occupations = build_subshell_occupations(spec.symbol, spec.charge)
    return sum(orbital.unpaired for orbital in occupations)


def build_configuration_summary(spec: AtomicSpec) -> dict[str, object]:
    symbol = canonical_symbol(spec.symbol)
    occupations = build_subshell_occupations(symbol, spec.charge)
    configuration = " ".join(f"{orbital.label}^{orbital.occupancy}" for orbital in occupations)
    unpaired = sum(orbital.unpaired for orbital in occupations)
    l_totals = _aggregate_subshell_occupations_by_l(occupations)
    reference_l_configuration = " ".join(
        f"{label}^{count}" for label, count in l_totals.items() if count > 0
    )
    return {
        "symbol": symbol,
        "atomic_number": atomic_number(symbol),
        "electrons": electron_count(symbol, spec.charge),
        "charge": spec.charge,
        "estimated_configuration": configuration,
        "reference_l_shell_configuration": reference_l_configuration,
        "estimated_unpaired_electrons": unpaired,
        "estimated_default_spin": unpaired,
        "subshell_occupations": occupations,
    }


def build_atomic_molecule(spec: AtomicSpec) -> gto.Mole:
    symbol = canonical_symbol(spec.symbol)
    spin = resolve_spin(spec)
    mol = gto.Mole()
    mol.atom = f"{symbol} 0.0 0.0 0.0"
    mol.unit = "Bohr"
    mol.basis = spec.basis
    mol.charge = spec.charge
    mol.spin = spin
    mol.cart = False
    mol.build()
    return mol


def angular_momentum_label(l_value: int) -> str:
    if l_value < len(ANGULAR_LABELS):
        return ANGULAR_LABELS[l_value]
    return f"l={l_value}"


def summarize_basis_shells(mol: gto.Mole) -> list[dict[str, int | str]]:
    summary: dict[int, dict[str, int | str]] = {}
    for bas_idx in range(mol.nbas):
        l_value = int(mol.bas_angular(bas_idx))
        nctr = int(mol.bas_nctr(bas_idx))
        nprim = int(mol.bas_nprim(bas_idx))
        degeneracy = 2 * l_value + 1
        if l_value not in summary:
            summary[l_value] = {
                "l": l_value,
                "label": angular_momentum_label(l_value),
                "shells": 0,
                "contracted_functions": 0,
                "primitive_functions": 0,
                "aos": 0,
            }
        entry = summary[l_value]
        entry["shells"] += 1
        entry["contracted_functions"] += nctr
        entry["primitive_functions"] += nprim
        entry["aos"] += nctr * degeneracy
    return [summary[key] for key in sorted(summary)]


def analyze_basis_engineering(mol: gto.Mole) -> dict[str, object]:
    symbol = canonical_symbol(mol.atom_symbol(0))
    raw_basis = mol._basis[symbol]
    shell_summaries: list[dict[str, object]] = []
    angular_summary: dict[int, dict[str, object]] = {}
    general_shells = 0
    segmented_shells = 0

    for shell_index, shell in enumerate(raw_basis, start=1):
        l_value = int(shell[0])
        primitive_start = 1
        while primitive_start < len(shell) and isinstance(shell[primitive_start], Real):
            primitive_start += 1
        primitives = shell[primitive_start:]
        nprim = len(primitives)
        nctr = len(primitives[0]) - 1 if primitives else 0
        contraction_type = "general" if nctr > 1 else "segmented"
        if contraction_type == "general":
            general_shells += 1
        else:
            segmented_shells += 1

        shell_summary = {
            "shell_index": shell_index,
            "l": l_value,
            "label": angular_momentum_label(l_value),
            "primitives": nprim,
            "contracted_radial_functions": nctr,
            "contraction_type": contraction_type,
            "coefficient_entries": nprim * nctr,
            "contraction_signature": f"{nprim}->{nctr}",
        }
        shell_summaries.append(shell_summary)

        if l_value not in angular_summary:
            angular_summary[l_value] = {
                "l": l_value,
                "label": angular_momentum_label(l_value),
                "shells": 0,
                "general_shells": 0,
                "segmented_shells": 0,
                "primitives": 0,
                "contracted_radial_functions": 0,
                "coefficient_entries": 0,
                "max_primitives_per_shell": 0,
                "max_contractions_per_shell": 0,
            }
        entry = angular_summary[l_value]
        entry["shells"] += 1
        entry["general_shells"] += int(contraction_type == "general")
        entry["segmented_shells"] += int(contraction_type == "segmented")
        entry["primitives"] += nprim
        entry["contracted_radial_functions"] += nctr
        entry["coefficient_entries"] += nprim * nctr
        entry["max_primitives_per_shell"] = max(int(entry["max_primitives_per_shell"]), nprim)
        entry["max_contractions_per_shell"] = max(int(entry["max_contractions_per_shell"]), nctr)

    total_primitives = sum(int(shell["primitives"]) for shell in shell_summaries)
    total_contracted = sum(int(shell["contracted_radial_functions"]) for shell in shell_summaries)
    total_entries = sum(int(shell["coefficient_entries"]) for shell in shell_summaries)
    return {
        "symbol": symbol,
        "has_general_contractions": general_shells > 0,
        "general_shells": general_shells,
        "segmented_shells": segmented_shells,
        "total_shells": len(shell_summaries),
        "total_primitives": total_primitives,
        "total_contracted_radial_functions": total_contracted,
        "total_coefficient_entries": total_entries,
        "global_contraction_ratio": float(total_contracted / total_primitives) if total_primitives else 0.0,
        "shell_summaries": shell_summaries,
        "angular_momentum_summary": [angular_summary[key] for key in sorted(angular_summary)],
    }
