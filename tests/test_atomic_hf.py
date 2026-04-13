from __future__ import annotations

import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf.scf import atom_hf as pyscf_atom_hf

from atomic_hf import (
    AtomicSpec,
    analyze_basis_engineering,
    build_atomic_molecule,
    build_configuration_summary,
    resolve_spin,
    run_benchmark_sweep,
    run_atomic_rhf,
    run_atomic_uhf,
)
from atomic_hf.blocks import (
    build_active_eri_quartets,
    build_fock,
    build_fock_from_active_quartets,
    build_reduced_radial_eri_repository,
    build_rhf_fock_from_reduced_radial_eri,
    build_uhf_fock,
    build_uhf_fock_from_active_quartets,
)


def test_default_spin_estimate_for_oxygen() -> None:
    spec = AtomicSpec(symbol="O")
    summary = build_configuration_summary(spec)
    assert summary["estimated_configuration"] == "1s^2 2s^2 2p^4"
    assert summary["estimated_default_spin"] == 2
    assert resolve_spin(spec) == 2


def test_reference_configuration_captures_chromium_exception() -> None:
    spec = AtomicSpec(symbol="Cr")
    summary = build_configuration_summary(spec)
    assert summary["estimated_configuration"] == "1s^2 2s^2 2p^6 3s^2 3p^6 4s^1 3d^5"
    assert summary["estimated_default_spin"] == 6
    assert resolve_spin(spec) == 6


def test_atomic_rhf_for_helium() -> None:
    result = run_atomic_rhf(AtomicSpec(symbol="He", basis="sto-3g", spin=0))
    assert abs(result.energy - (-2.807783957539974)) < 1.0e-10
    assert result.spherical_average is True
    assert result.iterations >= 1
    assert result.one_center_integral_summary["full_ao_dimension"] == 1
    assert result.one_center_integral_summary["reduced_radial_dimension"] == 1


def test_blocked_atomic_rhf_matches_pyscf_atomic_solver_for_neon() -> None:
    spec = AtomicSpec(symbol="Ne", basis="sto-3g", spin=0)
    result = run_atomic_rhf(spec)

    mol = gto.M(atom="Ne 0 0 0", basis="sto-3g", spin=0, charge=0, cart=False)
    reference = pyscf_atom_hf.AtomSphAverageRHF(mol)
    reference.verbose = 0
    reference.conv_tol = 1.0e-10
    reference.diis_space = 6
    reference_energy = reference.kernel()

    assert abs(result.energy - reference_energy) < 1.0e-9
    assert result.one_center_integral_summary["full_ao_dimension"] == 5
    assert result.one_center_integral_summary["reduced_radial_dimension"] == 3
    assert len(result.symmetry_blocks) == 2
    assert result.two_electron_integral_summary["total_angular_quartets"] == 16
    assert result.two_electron_integral_summary["active_angular_quartets"] >= 1


def test_atomic_uhf_for_oxygen() -> None:
    result = run_atomic_uhf(AtomicSpec(symbol="O", basis="sto-3g", spin=2))
    mol = gto.M(atom="O 0 0 0", basis="sto-3g", spin=2, charge=0, cart=False)
    reference = scf.UHF(mol)
    reference.verbose = 0
    reference.conv_tol = 1.0e-10
    reference.diis_space = 6
    reference.init_guess = "atom"
    reference_energy = reference.kernel()
    reference_s2 = reference.spin_square()[0]

    assert abs(result.energy - reference_energy) < 1.0e-9
    assert abs(result.s2 - reference_s2) < 1.0e-9
    assert result.nalpha == 5
    assert result.nbeta == 3
    assert result.one_center_integral_summary["reduced_radial_dimension"] == 3
    assert result.two_electron_integral_summary["total_angular_quartets"] == 16
    assert len(result.symmetry_blocks_alpha) == 2
    assert len(result.symmetry_blocks_beta) == 2


def test_basis_engineering_detects_general_contraction() -> None:
    general_basis = {
        "He": [
            [
                0,
                [6.36242139, 0.15432897, -0.09996723],
                [1.15892300, 0.53532814, 0.39951283],
                [0.31364979, 0.44463454, 0.70011547],
            ]
        ]
    }
    mol = build_atomic_molecule(AtomicSpec(symbol="He", basis=general_basis, spin=0))
    summary = analyze_basis_engineering(mol)

    assert summary["has_general_contractions"] is True
    assert summary["general_shells"] == 1
    assert summary["total_contracted_radial_functions"] == 2
    assert summary["shell_summaries"][0]["contraction_signature"] == "3->2"


def test_small_benchmark_sweep_runs() -> None:
    summary = run_benchmark_sweep(min_z=1, max_z=3, basis="sto-3g", method="auto", compare_reference=False)
    assert summary["successful_cases"] == 3
    assert summary["failed_cases"] == 0
    assert len(summary["entries"]) == 3


def test_quartet_screened_rhf_fock_matches_dense_builder() -> None:
    mol = build_atomic_molecule(AtomicSpec(symbol="Ne", basis="sto-3g", spin=0))
    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    density = np.array(
        [
            [1.02, 0.03, -0.01, 0.00, 0.00],
            [0.03, 0.91, 0.02, 0.01, -0.01],
            [-0.01, 0.02, 0.44, 0.00, 0.02],
            [0.00, 0.01, 0.00, 0.45, 0.00],
            [0.00, -0.01, 0.02, 0.00, 0.43],
        ]
    )
    density = 0.5 * (density + density.T)
    active_quartets = build_active_eri_quartets(mol, eri)

    dense_fock = build_fock(h_core, eri, density)
    screened_fock = build_fock_from_active_quartets(h_core, eri, density, active_quartets)

    assert np.allclose(screened_fock, dense_fock, atol=1.0e-12, rtol=1.0e-12)


def test_reduced_radial_rhf_fock_matches_dense_builder_for_spherical_density() -> None:
    result = run_atomic_rhf(AtomicSpec(symbol="Ne", basis="sto-3g", spin=0))
    mol = build_atomic_molecule(AtomicSpec(symbol="Ne", basis="sto-3g", spin=0))
    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    reduced_radial_eri = build_reduced_radial_eri_repository(mol, eri)

    dense_fock = build_fock(h_core, eri, result.density)
    reduced_fock = build_rhf_fock_from_reduced_radial_eri(h_core, result.density, mol, reduced_radial_eri)

    assert np.allclose(reduced_fock, dense_fock, atol=1.0e-12, rtol=1.0e-12)


def test_quartet_screened_uhf_fock_matches_dense_builder() -> None:
    mol = build_atomic_molecule(AtomicSpec(symbol="O", basis="sto-3g", spin=2))
    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    density_alpha = np.array(
        [
            [1.01, 0.02, 0.00, 0.01, -0.01],
            [0.02, 0.88, 0.01, 0.00, 0.01],
            [0.00, 0.01, 0.36, 0.02, 0.00],
            [0.01, 0.00, 0.02, 0.35, -0.01],
            [-0.01, 0.01, 0.00, -0.01, 0.34],
        ]
    )
    density_beta = np.array(
        [
            [0.99, -0.01, 0.01, 0.00, 0.00],
            [-0.01, 0.79, 0.00, -0.01, 0.01],
            [0.01, 0.00, 0.25, 0.00, 0.01],
            [0.00, -0.01, 0.00, 0.24, 0.00],
            [0.00, 0.01, 0.01, 0.00, 0.23],
        ]
    )
    density_alpha = 0.5 * (density_alpha + density_alpha.T)
    density_beta = 0.5 * (density_beta + density_beta.T)
    active_quartets = build_active_eri_quartets(mol, eri)

    dense_alpha, dense_beta = build_uhf_fock(h_core, eri, density_alpha, density_beta)
    screened_alpha, screened_beta = build_uhf_fock_from_active_quartets(
        h_core,
        eri,
        density_alpha,
        density_beta,
        active_quartets,
    )

    assert np.allclose(screened_alpha, dense_alpha, atol=1.0e-12, rtol=1.0e-12)
    assert np.allclose(screened_beta, dense_beta, atol=1.0e-12, rtol=1.0e-12)
