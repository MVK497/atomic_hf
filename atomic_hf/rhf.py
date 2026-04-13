from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atom import (
    AtomicSpec,
    analyze_basis_engineering,
    build_atomic_molecule,
    build_configuration_summary,
    canonical_symbol,
    resolve_spin,
    summarize_basis_shells,
)
from .blocks import (
    DIISHelper,
    analyze_one_center_integrals,
    analyze_two_electron_integrals,
    blocked_generalized_eigh,
    build_atomic_mo_occupations,
    build_density_from_occupations,
    build_fock,
    compute_diis_error,
)


@dataclass
class AtomicRHFResult:
    symbol: str
    atomic_number: int
    charge: int
    spin: int
    basis: str
    energy: float
    orbital_energies: np.ndarray
    coefficients: np.ndarray
    density: np.ndarray
    mo_occupations: np.ndarray
    iterations: int
    history: list[float]
    basis_summary: list[dict[str, int | str]]
    basis_engineering_summary: dict[str, object]
    configuration_summary: dict[str, object]
    symmetry_blocks: list[dict[str, int | str]]
    one_center_integral_summary: dict[str, object]
    two_electron_integral_summary: dict[str, object]
    spherical_average: bool


def run_atomic_rhf(
    spec: AtomicSpec,
    max_iter: int = 100,
    e_tol: float = 1.0e-10,
    d_tol: float = 1.0e-8,
    use_diis: bool = True,
    diis_space: int = 6,
) -> AtomicRHFResult:
    spin = resolve_spin(spec)
    if spin != 0:
        raise ValueError(
            "Atomic RHF in this project is restricted to closed-shell atoms/ions. "
            "Use --method uhf or explicitly provide a closed-shell spin."
        )

    mol = build_atomic_molecule(spec)
    s = mol.intor("int1e_ovlp")
    t = mol.intor("int1e_kin")
    v = mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    h_core = t + v
    e_nuc = float(mol.energy_nuc())

    mo_occ = build_atomic_mo_occupations(mol)
    history: list[float] = []
    diis_helper = DIISHelper(max_vectors=diis_space)

    orbital_energies, coefficients, symmetry_blocks = blocked_generalized_eigh(h_core, s, mol)
    density = build_density_from_occupations(coefficients, mo_occ)
    previous_energy: float | None = None

    overlap_eigvals, overlap_eigvecs = np.linalg.eigh(s)
    x = overlap_eigvecs @ np.diag(overlap_eigvals ** -0.5) @ overlap_eigvecs.T

    for iteration in range(1, max_iter + 1):
        fock = build_fock(h_core, eri, density)
        if use_diis:
            diis_error = compute_diis_error(fock, density, s, x)
            diis_helper.push(fock, diis_error)
            fock_to_diagonalize = diis_helper.extrapolate()
        else:
            fock_to_diagonalize = fock

        orbital_energies, coefficients, symmetry_blocks = blocked_generalized_eigh(fock_to_diagonalize, s, mol)
        new_density = build_density_from_occupations(coefficients, mo_occ)
        new_fock = build_fock(h_core, eri, new_density)
        electronic_energy = 0.5 * float(np.sum(new_density * (h_core + new_fock)))
        total_energy = electronic_energy + e_nuc
        history.append(total_energy)

        density_change = float(np.linalg.norm(new_density - density))
        if previous_energy is not None and abs(total_energy - previous_energy) < e_tol and density_change < d_tol:
            orbital_energies, coefficients, symmetry_blocks = blocked_generalized_eigh(new_fock, s, mol)
            final_density = build_density_from_occupations(coefficients, mo_occ)
            return AtomicRHFResult(
                symbol=canonical_symbol(spec.symbol),
                atomic_number=int(build_configuration_summary(spec)["atomic_number"]),
                charge=spec.charge,
                spin=spin,
                basis=spec.basis,
                energy=float(total_energy),
                orbital_energies=np.array(orbital_energies, copy=True),
                coefficients=np.array(coefficients, copy=True),
                density=np.array(final_density, copy=True),
                mo_occupations=np.array(mo_occ, copy=True),
                iterations=iteration,
                history=history,
                basis_summary=summarize_basis_shells(mol),
                basis_engineering_summary=analyze_basis_engineering(mol),
                configuration_summary=build_configuration_summary(spec),
                symmetry_blocks=symmetry_blocks,
                one_center_integral_summary=analyze_one_center_integrals(mol, s, h_core),
                two_electron_integral_summary=analyze_two_electron_integrals(mol, eri),
                spherical_average=True,
            )

        density = new_density
        previous_energy = total_energy

    raise RuntimeError("Blocked atomic RHF did not converge within the iteration limit.")
