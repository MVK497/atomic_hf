from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import scf

from .atom import (
    AtomicSpec,
    build_atomic_molecule,
    build_configuration_summary,
    canonical_symbol,
    summarize_basis_shells,
)
from .blocks import analyze_one_center_integrals


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
    configuration_summary: dict[str, object]
    one_center_integral_summary: dict[str, object]


def run_atomic_uhf(
    spec: AtomicSpec,
    max_iter: int = 100,
    e_tol: float = 1.0e-10,
    use_diis: bool = True,
    diis_space: int = 6,
) -> AtomicUHFResult:
    if spec.spin is None:
        raise ValueError(
            "Atomic UHF currently requires an explicit spin (2S). "
            "This avoids silently choosing the wrong open-shell state."
        )

    mol = build_atomic_molecule(spec)
    mf = scf.UHF(mol)
    mf.max_cycle = max_iter
    mf.conv_tol = e_tol
    mf.verbose = 0
    mf.init_guess = "atom"
    history: list[float] = []

    if use_diis:
        mf.diis_space = diis_space
    else:
        mf.DIIS = None
        mf.diis = None

    def callback(envs: dict[str, object]) -> None:
        energy = envs.get("e_tot")
        if energy is not None:
            history.append(float(energy))

    mf.callback = callback
    total_energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError("Atomic UHF did not converge within the iteration limit.")

    density_alpha, density_beta = mf.make_rdm1()
    mo_energy_alpha, mo_energy_beta = mf.mo_energy
    coeff_alpha, coeff_beta = mf.mo_coeff
    occ_alpha, occ_beta = mf.mo_occ
    nalpha, nbeta = mol.nelec
    s2, _ = mf.spin_square()
    expected_s2 = 0.5 * mol.spin * (0.5 * mol.spin + 1.0)
    configuration_summary = build_configuration_summary(spec)
    overlap = mol.intor("int1e_ovlp")
    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

    return AtomicUHFResult(
        symbol=canonical_symbol(spec.symbol),
        atomic_number=int(configuration_summary["atomic_number"]),
        charge=spec.charge,
        spin=mol.spin,
        basis=spec.basis,
        energy=total_energy,
        orbital_energies_alpha=np.array(mo_energy_alpha, copy=True),
        orbital_energies_beta=np.array(mo_energy_beta, copy=True),
        coefficients_alpha=np.array(coeff_alpha, copy=True),
        coefficients_beta=np.array(coeff_beta, copy=True),
        density_alpha=np.array(density_alpha, copy=True),
        density_beta=np.array(density_beta, copy=True),
        mo_occupations_alpha=np.array(occ_alpha, copy=True),
        mo_occupations_beta=np.array(occ_beta, copy=True),
        iterations=len(history),
        history=history,
        nalpha=nalpha,
        nbeta=nbeta,
        s2=float(s2),
        expected_s2=float(expected_s2),
        spin_contamination=float(s2 - expected_s2),
        basis_summary=summarize_basis_shells(mol),
        configuration_summary=configuration_summary,
        one_center_integral_summary=analyze_one_center_integrals(mol, overlap, h_core),
    )
