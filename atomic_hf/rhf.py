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
    apply_level_shift,
    analyze_one_center_integrals,
    analyze_two_electron_integrals,
    blocked_generalized_eigh,
    build_atomic_mo_occupations_from_spec,
    build_atomic_reference_density,
    build_density_from_occupations,
    build_gaunt_channel_eri_repository,
    damp_density,
    build_rhf_fock_from_gaunt_channels,
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
    initial_guess: str
    stabilization_summary: dict[str, object]
    fock_build_summary: dict[str, object]


def run_atomic_rhf(
    spec: AtomicSpec,
    max_iter: int = 100,
    e_tol: float = 1.0e-10,
    d_tol: float = 1.0e-8,
    use_diis: bool = True,
    diis_space: int = 6,
    initial_guess: str = "atom",
    damping_factor: float = 0.2,
    damping_cycles: int = 4,
    level_shift: float = 0.5,
    diis_start_cycle: int = 2,
    with_analysis: bool = True,
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
    gaunt_channel_eri = build_gaunt_channel_eri_repository(mol, eri)

    mo_occ = build_atomic_mo_occupations_from_spec(spec, mol)
    history: list[float] = []
    diis_helper = DIISHelper(max_vectors=diis_space)

    orbital_energies, coefficients, symmetry_blocks = blocked_generalized_eigh(h_core, s, mol)
    if initial_guess == "atom":
        density = build_atomic_reference_density(mol)
    elif initial_guess == "core":
        density = build_density_from_occupations(coefficients, mo_occ)
    else:
        raise ValueError("RHF initial_guess must be 'atom' or 'core'.")
    density = 0.5 * (density + density.T)
    previous_energy: float | None = None

    overlap_eigvals, overlap_eigvecs = np.linalg.eigh(s)
    x = overlap_eigvecs @ np.diag(overlap_eigvals ** -0.5) @ overlap_eigvecs.T

    for iteration in range(1, max_iter + 1):
        fock = build_rhf_fock_from_gaunt_channels(h_core, density, mol, gaunt_channel_eri)
        if use_diis and iteration >= diis_start_cycle:
            diis_error = compute_diis_error(fock, density, s, x)
            diis_helper.push(fock, diis_error)
            fock_to_diagonalize = diis_helper.extrapolate()
        else:
            fock_to_diagonalize = fock
        fock_to_diagonalize = apply_level_shift(fock_to_diagonalize, s, coefficients, mo_occ, level_shift)

        orbital_energies, coefficients, symmetry_blocks = blocked_generalized_eigh(fock_to_diagonalize, s, mol)
        new_density = build_density_from_occupations(coefficients, mo_occ)
        if damping_factor > 0.0 and iteration <= damping_cycles:
            new_density = damp_density(density, new_density, damping_factor)
        new_fock = build_rhf_fock_from_gaunt_channels(h_core, new_density, mol, gaunt_channel_eri)
        electronic_energy = 0.5 * float(np.sum(new_density * (h_core + new_fock)))
        total_energy = electronic_energy + e_nuc
        history.append(total_energy)

        density_change = float(np.linalg.norm(new_density - density))
        if previous_energy is not None and abs(total_energy - previous_energy) < e_tol and density_change < d_tol:
            orbital_energies, coefficients, symmetry_blocks = blocked_generalized_eigh(new_fock, s, mol)
            final_density = build_density_from_occupations(coefficients, mo_occ)
            configuration_summary = build_configuration_summary(spec)
            return AtomicRHFResult(
                symbol=canonical_symbol(spec.symbol),
                atomic_number=int(configuration_summary["atomic_number"]),
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
                basis_summary=summarize_basis_shells(mol) if with_analysis else [],
                basis_engineering_summary=analyze_basis_engineering(mol) if with_analysis else {},
                configuration_summary=configuration_summary,
                symmetry_blocks=symmetry_blocks,
                one_center_integral_summary=analyze_one_center_integrals(mol, s, h_core) if with_analysis else {},
                two_electron_integral_summary=analyze_two_electron_integrals(mol, eri) if with_analysis else {},
                spherical_average=True,
                initial_guess=initial_guess,
                stabilization_summary={
                    "use_diis": use_diis,
                    "diis_space": diis_space,
                    "diis_start_cycle": diis_start_cycle,
                    "damping_factor": damping_factor,
                    "damping_cycles": damping_cycles,
                    "level_shift": level_shift,
                },
                fock_build_summary={
                    "builder": "gaunt_channel_reduced_radial",
                    "active_reduced_radial_pair_blocks": len(gaunt_channel_eri.pair_blocks),
                },
            )

        density = new_density
        previous_energy = total_energy

    raise RuntimeError("Blocked atomic RHF did not converge within the iteration limit.")
