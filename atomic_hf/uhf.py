from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atom import (
    AtomicSpec,
    analyze_basis_engineering,
    build_atomic_molecule,
    build_configuration_summary,
    canonical_symbol,
    summarize_basis_shells,
)
from .blocks import (
    DIISHelper,
    analyze_one_center_integrals,
    analyze_two_electron_integrals,
    blocked_generalized_eigh,
    build_spin_density_from_count,
    build_uhf_fock,
    combine_spin_blocks,
    compute_diis_error,
    compute_uhf_s2,
    sort_orbitals_by_energy,
    split_spin_blocks,
)


@dataclass
class AtomicUHFResult:
    symbol: str
    atomic_number: int
    charge: int
    spin: int
    basis: str
    energy: float
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    coefficients_alpha: np.ndarray
    coefficients_beta: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    mo_occupations_alpha: np.ndarray
    mo_occupations_beta: np.ndarray
    iterations: int
    history: list[float]
    nalpha: int
    nbeta: int
    s2: float
    expected_s2: float
    spin_contamination: float
    basis_summary: list[dict[str, int | str]]
    basis_engineering_summary: dict[str, object]
    configuration_summary: dict[str, object]
    symmetry_blocks_alpha: list[dict[str, int | str]]
    symmetry_blocks_beta: list[dict[str, int | str]]
    one_center_integral_summary: dict[str, object]
    two_electron_integral_summary: dict[str, object]


def run_atomic_uhf(
    spec: AtomicSpec,
    max_iter: int = 100,
    e_tol: float = 1.0e-10,
    d_tol: float = 1.0e-8,
    use_diis: bool = True,
    diis_space: int = 6,
) -> AtomicUHFResult:
    if spec.spin is None:
        raise ValueError(
            "Atomic UHF currently requires an explicit spin (2S). "
            "This avoids silently choosing the wrong open-shell state."
        )

    mol = build_atomic_molecule(spec)
    nalpha, nbeta = mol.nelec

    s = mol.intor("int1e_ovlp")
    t = mol.intor("int1e_kin")
    v = mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    h_core = t + v
    e_nuc = float(mol.energy_nuc())

    overlap_eigvals, overlap_eigvecs = np.linalg.eigh(s)
    x = overlap_eigvecs @ np.diag(overlap_eigvals ** -0.5) @ overlap_eigvecs.T
    diis_helper = DIISHelper(max_vectors=diis_space)
    history: list[float] = []

    orbital_energies_alpha, coefficients_alpha, symmetry_blocks_alpha = blocked_generalized_eigh(h_core, s, mol)
    orbital_energies_beta, coefficients_beta, symmetry_blocks_beta = blocked_generalized_eigh(h_core, s, mol)
    orbital_energies_alpha, coefficients_alpha = sort_orbitals_by_energy(orbital_energies_alpha, coefficients_alpha)
    orbital_energies_beta, coefficients_beta = sort_orbitals_by_energy(orbital_energies_beta, coefficients_beta)
    density_alpha, mo_occ_alpha = build_spin_density_from_count(coefficients_alpha, nalpha)
    density_beta, mo_occ_beta = build_spin_density_from_count(coefficients_beta, nbeta)

    previous_energy: float | None = None
    for iteration in range(1, max_iter + 1):
        fock_alpha, fock_beta = build_uhf_fock(h_core, eri, density_alpha, density_beta)
        if use_diis:
            error_alpha = compute_diis_error(fock_alpha, density_alpha, s, x)
            error_beta = compute_diis_error(fock_beta, density_beta, s, x)
            diis_helper.push(
                combine_spin_blocks(fock_alpha, fock_beta),
                combine_spin_blocks(error_alpha, error_beta),
            )
            fock_alpha_to_diag, fock_beta_to_diag = split_spin_blocks(diis_helper.extrapolate())
        else:
            fock_alpha_to_diag, fock_beta_to_diag = fock_alpha, fock_beta

        orbital_energies_alpha, coefficients_alpha, symmetry_blocks_alpha = blocked_generalized_eigh(
            fock_alpha_to_diag, s, mol
        )
        orbital_energies_beta, coefficients_beta, symmetry_blocks_beta = blocked_generalized_eigh(
            fock_beta_to_diag, s, mol
        )
        orbital_energies_alpha, coefficients_alpha = sort_orbitals_by_energy(orbital_energies_alpha, coefficients_alpha)
        orbital_energies_beta, coefficients_beta = sort_orbitals_by_energy(orbital_energies_beta, coefficients_beta)

        new_density_alpha, mo_occ_alpha = build_spin_density_from_count(coefficients_alpha, nalpha)
        new_density_beta, mo_occ_beta = build_spin_density_from_count(coefficients_beta, nbeta)
        new_fock_alpha, new_fock_beta = build_uhf_fock(h_core, eri, new_density_alpha, new_density_beta)

        electronic_energy = 0.5 * (
            np.sum((new_density_alpha + new_density_beta) * h_core)
            + np.sum(new_density_alpha * new_fock_alpha)
            + np.sum(new_density_beta * new_fock_beta)
        )
        total_energy = float(electronic_energy + e_nuc)
        history.append(total_energy)

        density_change = float(
            np.sqrt(
                np.linalg.norm(new_density_alpha - density_alpha) ** 2
                + np.linalg.norm(new_density_beta - density_beta) ** 2
            )
        )
        if previous_energy is not None and abs(total_energy - previous_energy) < e_tol and density_change < d_tol:
            orbital_energies_alpha, coefficients_alpha, symmetry_blocks_alpha = blocked_generalized_eigh(
                new_fock_alpha, s, mol
            )
            orbital_energies_beta, coefficients_beta, symmetry_blocks_beta = blocked_generalized_eigh(
                new_fock_beta, s, mol
            )
            orbital_energies_alpha, coefficients_alpha = sort_orbitals_by_energy(
                orbital_energies_alpha, coefficients_alpha
            )
            orbital_energies_beta, coefficients_beta = sort_orbitals_by_energy(
                orbital_energies_beta, coefficients_beta
            )
            final_density_alpha, mo_occ_alpha = build_spin_density_from_count(coefficients_alpha, nalpha)
            final_density_beta, mo_occ_beta = build_spin_density_from_count(coefficients_beta, nbeta)
            s2, expected_s2, spin_contamination = compute_uhf_s2(
                s,
                coefficients_alpha,
                coefficients_beta,
                nalpha,
                nbeta,
            )
            configuration_summary = build_configuration_summary(spec)
            return AtomicUHFResult(
                symbol=canonical_symbol(spec.symbol),
                atomic_number=int(configuration_summary["atomic_number"]),
                charge=spec.charge,
                spin=mol.spin,
                basis=spec.basis,
                energy=total_energy,
                orbital_energies_alpha=np.array(orbital_energies_alpha, copy=True),
                orbital_energies_beta=np.array(orbital_energies_beta, copy=True),
                coefficients_alpha=np.array(coefficients_alpha, copy=True),
                coefficients_beta=np.array(coefficients_beta, copy=True),
                density_alpha=np.array(final_density_alpha, copy=True),
                density_beta=np.array(final_density_beta, copy=True),
                mo_occupations_alpha=np.array(mo_occ_alpha, copy=True),
                mo_occupations_beta=np.array(mo_occ_beta, copy=True),
                iterations=iteration,
                history=history,
                nalpha=nalpha,
                nbeta=nbeta,
                s2=float(s2),
                expected_s2=float(expected_s2),
                spin_contamination=float(spin_contamination),
                basis_summary=summarize_basis_shells(mol),
                basis_engineering_summary=analyze_basis_engineering(mol),
                configuration_summary=configuration_summary,
                symmetry_blocks_alpha=symmetry_blocks_alpha,
                symmetry_blocks_beta=symmetry_blocks_beta,
                one_center_integral_summary=analyze_one_center_integrals(mol, s, h_core),
                two_electron_integral_summary=analyze_two_electron_integrals(mol, eri),
            )

        density_alpha = new_density_alpha
        density_beta = new_density_beta
        previous_energy = total_energy

    raise RuntimeError("Blocked atomic UHF did not converge within the iteration limit.")
