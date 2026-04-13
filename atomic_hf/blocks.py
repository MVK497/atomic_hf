from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import gto
from pyscf.lib import param
from pyscf.scf import atom_hf

from .atom import angular_momentum_label


@dataclass
class DIISHelper:
    max_vectors: int = 6

    def __post_init__(self) -> None:
        self.fock_matrices: list[np.ndarray] = []
        self.error_matrices: list[np.ndarray] = []

    def push(self, fock: np.ndarray, error: np.ndarray) -> None:
        self.fock_matrices.append(np.array(fock, copy=True))
        self.error_matrices.append(np.array(error, copy=True))
        if len(self.fock_matrices) > self.max_vectors:
            self.fock_matrices.pop(0)
            self.error_matrices.pop(0)

    def extrapolate(self) -> np.ndarray:
        count = len(self.fock_matrices)
        if count < 2:
            return self.fock_matrices[-1]

        b_matrix = np.empty((count + 1, count + 1))
        b_matrix[:-1, :-1] = np.array(
            [
                [np.vdot(err_i, err_j).real for err_j in self.error_matrices]
                for err_i in self.error_matrices
            ]
        )
        b_matrix[-1, :-1] = -1.0
        b_matrix[:-1, -1] = -1.0
        b_matrix[-1, -1] = 0.0

        rhs = np.zeros(count + 1)
        rhs[-1] = -1.0
        try:
            coefficients = np.linalg.solve(b_matrix, rhs)[:-1]
        except np.linalg.LinAlgError:
            return self.fock_matrices[-1]
        return sum(coeff * fock for coeff, fock in zip(coefficients, self.fock_matrices, strict=True))


def compute_diis_error(fock: np.ndarray, density: np.ndarray, overlap: np.ndarray, x: np.ndarray) -> np.ndarray:
    commutator = fock @ density @ overlap - overlap @ density @ fock
    return x.T @ commutator @ x


def ao_angular_momentum_for_each_ao(mol: gto.Mole) -> np.ndarray:
    ao_ang = np.zeros(mol.nao_nr(), dtype=int)
    ao_loc = mol.ao_loc_nr()
    for bas_idx in range(mol.nbas):
        p0, p1 = ao_loc[bas_idx], ao_loc[bas_idx + 1]
        ao_ang[p0:p1] = mol.bas_angular(bas_idx)
    return ao_ang


def reduced_matrix_for_l(matrix: np.ndarray, mol: gto.Mole, l_value: int) -> tuple[np.ndarray, np.ndarray]:
    ao_ang = ao_angular_momentum_for_each_ao(mol)
    idx = np.where(ao_ang == l_value)[0]
    if idx.size == 0:
        return np.zeros((0, 0)), idx
    degeneracy = 2 * l_value + 1
    nrad = idx.size // degeneracy
    block = matrix[idx[:, None], idx].reshape(nrad, degeneracy, nrad, degeneracy)
    reduced = np.einsum("piqi->pq", block) / degeneracy
    return reduced, idx


def blocked_eigensystem(
    fock: np.ndarray,
    overlap: np.ndarray,
    mol: gto.Mole,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str]]]:
    ao_ang = ao_angular_momentum_for_each_ao(mol)
    nao = mol.nao_nr()
    mo_coeff_blocks: list[np.ndarray] = []
    mo_energy_blocks: list[np.ndarray] = []
    metadata: list[dict[str, int | str]] = []

    for l_value in range(param.L_MAX):
        idx = np.where(ao_ang == l_value)[0]
        if idx.size == 0:
            continue

        degeneracy = 2 * l_value + 1
        nrad = idx.size // degeneracy
        f_l, _ = reduced_matrix_for_l(fock, mol, l_value)
        s_l, _ = reduced_matrix_for_l(overlap, mol, l_value)
        energies, coeff_reduced = np.linalg.eigh(np.linalg.solve(s_l, f_l))
        sort_idx = np.argsort(energies)
        energies = energies[sort_idx]
        coeff_reduced = coeff_reduced[:, sort_idx]

        coeff_full = np.zeros((nao, idx.size))
        for m_index in range(degeneracy):
            coeff_full[idx[m_index::degeneracy], m_index::degeneracy] = coeff_reduced

        mo_coeff_blocks.append(coeff_full)
        mo_energy_blocks.append(np.repeat(energies, degeneracy))
        metadata.append(
            {
                "l": l_value,
                "label": angular_momentum_label(l_value),
                "degeneracy": degeneracy,
                "ao_count": int(idx.size),
                "radial_functions": int(nrad),
            }
        )

    return np.hstack(mo_energy_blocks), np.hstack(mo_coeff_blocks), metadata


def blocked_generalized_eigh(
    fock: np.ndarray,
    overlap: np.ndarray,
    mol: gto.Mole,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str]]]:
    ao_ang = ao_angular_momentum_for_each_ao(mol)
    nao = mol.nao_nr()
    mo_coeff_blocks: list[np.ndarray] = []
    mo_energy_blocks: list[np.ndarray] = []
    metadata: list[dict[str, int | str]] = []

    for l_value in range(param.L_MAX):
        idx = np.where(ao_ang == l_value)[0]
        if idx.size == 0:
            continue
        degeneracy = 2 * l_value + 1
        nrad = idx.size // degeneracy
        f_l, _ = reduced_matrix_for_l(fock, mol, l_value)
        s_l, _ = reduced_matrix_for_l(overlap, mol, l_value)
        energies, coeff_reduced = np.linalg.eigh(s_l)
        if np.min(energies) < 1.0e-12:
            raise RuntimeError("Near-linear dependence detected in the reduced overlap block.")
        x = coeff_reduced @ np.diag(energies ** -0.5) @ coeff_reduced.T
        f_orth = x.T @ f_l @ x
        eps, c_orth = np.linalg.eigh(f_orth)
        coeff_reduced = x @ c_orth

        coeff_full = np.zeros((nao, idx.size))
        for m_index in range(degeneracy):
            coeff_full[idx[m_index::degeneracy], m_index::degeneracy] = coeff_reduced

        mo_coeff_blocks.append(coeff_full)
        mo_energy_blocks.append(np.repeat(eps, degeneracy))
        metadata.append(
            {
                "l": l_value,
                "label": angular_momentum_label(l_value),
                "degeneracy": degeneracy,
                "ao_count": int(idx.size),
                "radial_functions": int(nrad),
            }
        )
    return np.hstack(mo_energy_blocks), np.hstack(mo_coeff_blocks), metadata


def build_atomic_mo_occupations(mol: gto.Mole) -> np.ndarray:
    symbol = mol.atom_symbol(0)
    occupations: list[np.ndarray] = []
    for l_value in range(param.L_MAX):
        degeneracy = 2 * l_value + 1
        n2occ, frac = atom_hf.frac_occ(symbol, l_value)
        idx = mol._bas[:, gto.ANG_OF] == l_value
        nrad = int(mol._bas[idx, gto.NCTR_OF].sum())
        if nrad == 0:
            continue
        if n2occ > nrad:
            raise ValueError(
                f"Basis {mol.basis} does not provide enough radial functions for the occupied {angular_momentum_label(l_value)} block."
            )
        occ_l = np.zeros(nrad)
        occ_l[:n2occ] = 2.0
        if frac > 0 and n2occ < nrad:
            occ_l[n2occ] = frac
        occupations.append(np.repeat(occ_l, degeneracy))
    return np.hstack(occupations)


def build_density_from_occupations(coefficients: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
    return coefficients @ np.diag(mo_occ) @ coefficients.T


def build_fock(h_core: np.ndarray, eri: np.ndarray, density: np.ndarray) -> np.ndarray:
    coulomb = np.einsum("ls,mnls->mn", density, eri, optimize=True)
    exchange = np.einsum("ls,mlns->mn", density, eri, optimize=True)
    return h_core + coulomb - 0.5 * exchange


def build_uhf_fock(
    h_core: np.ndarray,
    eri: np.ndarray,
    density_alpha: np.ndarray,
    density_beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    density_total = density_alpha + density_beta
    coulomb = np.einsum("ls,mnls->mn", density_total, eri, optimize=True)
    exchange_alpha = np.einsum("ls,mlns->mn", density_alpha, eri, optimize=True)
    exchange_beta = np.einsum("ls,mlns->mn", density_beta, eri, optimize=True)
    return h_core + coulomb - exchange_alpha, h_core + coulomb - exchange_beta


def combine_spin_blocks(matrix_alpha: np.ndarray, matrix_beta: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(matrix_alpha)
    return np.block([[matrix_alpha, zeros], [zeros, matrix_beta]])


def split_spin_blocks(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nbf = matrix.shape[0] // 2
    return matrix[:nbf, :nbf], matrix[nbf:, nbf:]


def sort_orbitals_by_energy(
    orbital_energies: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(orbital_energies)
    return orbital_energies[order], coefficients[:, order]


def build_spin_mo_occupations(num_occupied: int, num_orbitals: int) -> np.ndarray:
    occupations = np.zeros(num_orbitals)
    occupations[:num_occupied] = 1.0
    return occupations


def build_spin_density_from_count(coefficients: np.ndarray, num_occupied: int) -> tuple[np.ndarray, np.ndarray]:
    occupations = build_spin_mo_occupations(num_occupied, coefficients.shape[1])
    return build_density_from_occupations(coefficients, occupations), occupations


def compute_uhf_s2(
    overlap: np.ndarray,
    coeff_alpha: np.ndarray,
    coeff_beta: np.ndarray,
    nalpha: int,
    nbeta: int,
) -> tuple[float, float, float]:
    occ_alpha = coeff_alpha[:, :nalpha]
    occ_beta = coeff_beta[:, :nbeta]
    if nalpha == 0 or nbeta == 0:
        overlap_occ = 0.0
    else:
        spin_overlap = occ_alpha.T @ overlap @ occ_beta
        overlap_occ = float(np.sum(np.abs(spin_overlap) ** 2))

    sz = 0.5 * (nalpha - nbeta)
    s2 = sz * (sz + 1.0) + nbeta - overlap_occ
    expected_s2 = sz * (sz + 1.0)
    return s2, expected_s2, s2 - expected_s2


def analyze_one_center_integrals(
    mol: gto.Mole,
    overlap: np.ndarray,
    h_core: np.ndarray,
) -> dict[str, object]:
    ao_ang = ao_angular_momentum_for_each_ao(mol)
    unique_l = sorted(set(int(value) for value in ao_ang))
    block_summaries: list[dict[str, object]] = []

    for l_value in unique_l:
        idx = np.where(ao_ang == l_value)[0]
        degeneracy = 2 * l_value + 1
        nrad = idx.size // degeneracy
        mask_other = ao_ang != l_value
        offdiag_s = overlap[idx[:, None], np.where(mask_other)[0]]
        offdiag_h = h_core[idx[:, None], np.where(mask_other)[0]]
        block_summaries.append(
            {
                "l": l_value,
                "label": angular_momentum_label(l_value),
                "degeneracy": degeneracy,
                "ao_count": int(idx.size),
                "radial_functions": int(nrad),
                "reduced_one_electron_block_shape": [int(nrad), int(nrad)],
                "overlap_offdiag_max_abs": float(np.max(np.abs(offdiag_s))) if offdiag_s.size else 0.0,
                "hcore_offdiag_max_abs": float(np.max(np.abs(offdiag_h))) if offdiag_h.size else 0.0,
            }
        )

    reduced_dimension = sum(int(item["radial_functions"]) for item in block_summaries)
    nao = mol.nao_nr()
    return {
        "full_ao_dimension": nao,
        "reduced_radial_dimension": reduced_dimension,
        "one_electron_compression_ratio": float(nao / reduced_dimension) if reduced_dimension else 1.0,
        "block_summaries": block_summaries,
    }


def analyze_two_electron_integrals(
    mol: gto.Mole,
    eri: np.ndarray,
    threshold: float = 1.0e-12,
) -> dict[str, object]:
    ao_ang = ao_angular_momentum_for_each_ao(mol)
    unique_l = sorted(set(int(value) for value in ao_ang))
    l_indices = {l_value: np.where(ao_ang == l_value)[0] for l_value in unique_l}
    radial_counts = {
        l_value: int(l_indices[l_value].size // (2 * l_value + 1))
        for l_value in unique_l
    }

    quartet_summaries: list[dict[str, object]] = []
    active_quartets = 0
    for l1 in unique_l:
        idx1 = l_indices[l1]
        for l2 in unique_l:
            idx2 = l_indices[l2]
            for l3 in unique_l:
                idx3 = l_indices[l3]
                for l4 in unique_l:
                    idx4 = l_indices[l4]
                    block = eri[np.ix_(idx1, idx2, idx3, idx4)]
                    max_abs = float(np.max(np.abs(block))) if block.size else 0.0
                    fro_norm = float(np.linalg.norm(block)) if block.size else 0.0
                    is_active = max_abs > threshold
                    active_quartets += int(is_active)
                    quartet_summaries.append(
                        {
                            "labels": (
                                angular_momentum_label(l1),
                                angular_momentum_label(l2),
                                angular_momentum_label(l3),
                                angular_momentum_label(l4),
                            ),
                            "l_values": [l1, l2, l3, l4],
                            "ao_shape": [int(idx1.size), int(idx2.size), int(idx3.size), int(idx4.size)],
                            "reduced_radial_shape": [
                                radial_counts[l1],
                                radial_counts[l2],
                                radial_counts[l3],
                                radial_counts[l4],
                            ],
                            "max_abs": max_abs,
                            "frobenius_norm": fro_norm,
                            "numerically_active": is_active,
                        }
                    )

    active_sorted = sorted(
        (quartet for quartet in quartet_summaries if quartet["numerically_active"]),
        key=lambda quartet: quartet["frobenius_norm"],
        reverse=True,
    )
    return {
        "threshold": threshold,
        "total_angular_quartets": len(quartet_summaries),
        "active_angular_quartets": active_quartets,
        "inactive_angular_quartets": len(quartet_summaries) - active_quartets,
        "active_ratio": float(active_quartets / len(quartet_summaries)) if quartet_summaries else 0.0,
        "dominant_active_quartets": active_sorted[:12],
        "quartet_summaries": quartet_summaries,
    }
