from __future__ import annotations

from pyscf import gto
from pyscf import scf
from pyscf.scf import atom_hf as pyscf_atom_hf

from atomic_hf import (
    AtomicSpec,
    analyze_basis_engineering,
    build_atomic_molecule,
    build_configuration_summary,
    resolve_spin,
    run_atomic_rhf,
    run_atomic_uhf,
)


def test_default_spin_estimate_for_oxygen() -> None:
    spec = AtomicSpec(symbol="O")
    summary = build_configuration_summary(spec)
    assert summary["estimated_configuration"] == "1s^2 2s^2 2p^4"
    assert summary["estimated_default_spin"] == 2
    assert resolve_spin(spec) == 2


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
