from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import gto
from pyscf.lib import param
from pyscf.scf import atom_hf, hf

from .atom import AtomicSpec, angular_momentum_label, build_subshell_occupations


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


@dataclass(frozen=True)
class ActiveERIQuartet:
    l_values: tuple[int, int, int, int]
    labels: tuple[str, str, str, str]
    idx_p: np.ndarray
    idx_q: np.ndarray
    idx_r: np.ndarray
    idx_s: np.ndarray
    ao_shape: tuple[int, int, int, int]
    reduced_radial_shape: tuple[int, int, int, int]
    max_abs: float
    frobenius_norm: float
    coulomb_ref: "QuartetBlockReference"
    exchange_ref: "QuartetBlockReference"


@dataclass(frozen=True)
class QuartetBlockReference:
    key: tuple[tuple[int, int], tuple[int, int]]
    requested_to_canonical_axes: tuple[int, int, int, int]
    canonical_to_requested_axes: tuple[int, int, int, int]


@dataclass
class StructuredERIRepository:
    threshold: float
    block_map: dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray]
    active_quartets: list[ActiveERIQuartet]


@dataclass(frozen=True)
class ReducedRadialPairBlock:
    l_output: int
    l_density: int
    output_label: str
    density_label: str
    output_radial_functions: int
    density_radial_functions: int
    coulomb_tensor: np.ndarray
    exchange_tensor: np.ndarray
    coulomb_max_abs: float
    exchange_max_abs: float
    coulomb_frobenius_norm: float
    exchange_frobenius_norm: float


@dataclass
class ReducedRadialERIRepository:
    threshold: float
    pair_blocks: dict[tuple[int, int], ReducedRadialPairBlock]


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


def expand_reduced_matrix_for_l(
    reduced_matrix: np.ndarray,
    mol: gto.Mole,
    l_value: int,
) -> np.ndarray:
    ao_ang = ao_angular_momentum_for_each_ao(mol)
    idx = np.where(ao_ang == l_value)[0]
    full = np.zeros((mol.nao_nr(), mol.nao_nr()))
    if idx.size == 0:
        return full
    degeneracy = 2 * l_value + 1
    block = np.zeros((idx.size, idx.size))
    for m_index in range(degeneracy):
        block[m_index::degeneracy, m_index::degeneracy] = reduced_matrix
    full[np.ix_(idx, idx)] = block
    return full


def build_reduced_density_by_l(
    density: np.ndarray,
    mol: gto.Mole,
) -> dict[int, np.ndarray]:
    ao_ang = ao_angular_momentum_for_each_ao(mol)
    reduced_density: dict[int, np.ndarray] = {}
    for l_value in sorted(set(int(value) for value in ao_ang)):
        reduced_density[l_value], _ = reduced_matrix_for_l(density, mol, l_value)
    return reduced_density


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
    orbital_start = 0

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
        orbital_stop = orbital_start + idx.size
        metadata.append(
            {
                "l": l_value,
                "label": angular_momentum_label(l_value),
                "degeneracy": degeneracy,
                "ao_count": int(idx.size),
                "radial_functions": int(nrad),
                "orbital_start": orbital_start,
                "orbital_stop": orbital_stop,
            }
        )
        orbital_start = orbital_stop

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
    orbital_start = 0

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
        orbital_stop = orbital_start + idx.size
        metadata.append(
            {
                "l": l_value,
                "label": angular_momentum_label(l_value),
                "degeneracy": degeneracy,
                "ao_count": int(idx.size),
                "radial_functions": int(nrad),
                "orbital_start": orbital_start,
                "orbital_stop": orbital_stop,
            }
        )
        orbital_start = orbital_stop
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


def build_atomic_reference_density(mol: gto.Mole) -> np.ndarray:
    return np.array(hf.init_guess_by_atom(mol), copy=True)


def build_atomic_mo_occupations_from_spec(spec: AtomicSpec, mol: gto.Mole) -> np.ndarray:
    symbol = mol.atom_symbol(0)
    occupations = build_subshell_occupations(symbol, spec.charge)
    grouped_by_l: dict[int, int] = {}
    for orbital in occupations:
        l_value = "spdfghi".index(orbital.l_label)
        grouped_by_l[l_value] = grouped_by_l.get(l_value, 0) + orbital.occupancy

    mo_occ_blocks: list[np.ndarray] = []
    for l_value in range(param.L_MAX):
        degeneracy = 2 * l_value + 1
        idx = mol._bas[:, gto.ANG_OF] == l_value
        nrad = int(mol._bas[idx, gto.NCTR_OF].sum())
        if nrad == 0:
            continue
        electrons_l = grouped_by_l.get(l_value, 0)
        if electrons_l > 2 * degeneracy * nrad:
            raise ValueError(
                f"Basis {mol.basis} does not provide enough radial functions for the occupied {angular_momentum_label(l_value)} block."
            )
        occ_l = np.zeros(nrad * degeneracy)
        full_doubly_occupied = electrons_l // 2
        remainder = electrons_l % 2
        occ_l[:full_doubly_occupied] = 2.0
        if remainder:
            occ_l[full_doubly_occupied] = 1.0
        mo_occ_blocks.append(occ_l)
    return np.hstack(mo_occ_blocks)


def build_spin_population_by_l(spec: AtomicSpec) -> tuple[dict[int, int], dict[int, int]]:
    occupations = build_subshell_occupations(spec.symbol, spec.charge)
    alpha_by_l: dict[int, int] = {}
    beta_by_l: dict[int, int] = {}
    for orbital in occupations:
        l_value = "spdfghi".index(orbital.l_label)
        degeneracy = orbital.capacity // 2
        alpha = min(orbital.occupancy, degeneracy)
        beta = orbital.occupancy - alpha
        alpha_by_l[l_value] = alpha_by_l.get(l_value, 0) + alpha
        beta_by_l[l_value] = beta_by_l.get(l_value, 0) + beta
    return alpha_by_l, beta_by_l


def rebalance_spin_population_by_l(
    alpha_by_l: dict[int, int],
    beta_by_l: dict[int, int],
    target_nalpha: int,
    target_nbeta: int,
) -> tuple[dict[int, int], dict[int, int]]:
    alpha = dict(alpha_by_l)
    beta = dict(beta_by_l)
    current_nalpha = sum(alpha.values())
    current_nbeta = sum(beta.values())
    if current_nalpha == target_nalpha and current_nbeta == target_nbeta:
        return alpha, beta

    if current_nalpha + current_nbeta != target_nalpha + target_nbeta:
        raise ValueError("Spin-population rebalance cannot change the total number of electrons.")

    while current_nalpha > target_nalpha:
        updated = False
        for l_value in sorted(alpha, reverse=True):
            if alpha.get(l_value, 0) > 0:
                alpha[l_value] -= 1
                beta[l_value] = beta.get(l_value, 0) + 1
                current_nalpha -= 1
                current_nbeta += 1
                updated = True
                break
        if not updated:
            raise ValueError("Unable to lower alpha electron count to the requested spin state.")

    while current_nalpha < target_nalpha:
        updated = False
        for l_value in sorted(beta, reverse=True):
            if beta.get(l_value, 0) > 0:
                beta[l_value] -= 1
                alpha[l_value] = alpha.get(l_value, 0) + 1
                current_nalpha += 1
                current_nbeta -= 1
                updated = True
                break
        if not updated:
            raise ValueError("Unable to raise alpha electron count to the requested spin state.")

    return alpha, beta


def build_spin_mo_occupations_by_blocks(
    symmetry_blocks: list[dict[str, int | str]],
    electrons_by_l: dict[int, int],
) -> np.ndarray:
    num_orbitals = 0
    for block in symmetry_blocks:
        num_orbitals = max(num_orbitals, int(block["orbital_stop"]))
    occupations = np.zeros(num_orbitals)
    for block in symmetry_blocks:
        start = int(block["orbital_start"])
        stop = int(block["orbital_stop"])
        l_value = int(block["l"])
        count = electrons_by_l.get(l_value, 0)
        block_size = stop - start
        if count > block_size:
            raise ValueError(
                f"Too many spin-{angular_momentum_label(l_value)} electrons ({count}) for block size {block_size}."
            )
        occupations[start : start + count] = 1.0
    return occupations


def build_spin_density_from_block_occupations(
    coefficients: np.ndarray,
    symmetry_blocks: list[dict[str, int | str]],
    electrons_by_l: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    occupations = build_spin_mo_occupations_by_blocks(symmetry_blocks, electrons_by_l)
    return build_density_from_occupations(coefficients, occupations), occupations


def damp_density(
    previous_density: np.ndarray,
    new_density: np.ndarray,
    damping_factor: float,
) -> np.ndarray:
    if damping_factor <= 0.0:
        return np.array(new_density, copy=True)
    return damping_factor * previous_density + (1.0 - damping_factor) * new_density


def apply_level_shift(
    fock: np.ndarray,
    overlap: np.ndarray,
    coefficients: np.ndarray,
    mo_occ: np.ndarray,
    level_shift: float,
) -> np.ndarray:
    if level_shift <= 0.0:
        return np.array(fock, copy=True)
    occupied_mask = mo_occ > 1.0e-12
    virtual_coeff = coefficients[:, ~occupied_mask]
    if virtual_coeff.size == 0:
        return np.array(fock, copy=True)
    return fock + level_shift * overlap @ virtual_coeff @ virtual_coeff.T @ overlap


def build_fock(h_core: np.ndarray, eri: np.ndarray, density: np.ndarray) -> np.ndarray:
    coulomb = np.einsum("ls,mnls->mn", density, eri, optimize=True)
    exchange = np.einsum("ls,mlns->mn", density, eri, optimize=True)
    return h_core + coulomb - 0.5 * exchange


def _canonicalize_quartet_axes(
    l1: int,
    l2: int,
    l3: int,
    l4: int,
) -> QuartetBlockReference:
    pair_left = ((0, l1), (1, l2))
    pair_right = ((2, l3), (3, l4))

    if pair_left[0][1] > pair_left[1][1]:
        pair_left = (pair_left[1], pair_left[0])
    if pair_right[0][1] > pair_right[1][1]:
        pair_right = (pair_right[1], pair_right[0])
    if (pair_left[0][1], pair_left[1][1]) > (pair_right[0][1], pair_right[1][1]):
        canonical_tokens = pair_right + pair_left
    else:
        canonical_tokens = pair_left + pair_right

    requested_to_canonical_axes = tuple(token[0] for token in canonical_tokens)
    canonical_to_requested_axes = tuple(int(value) for value in np.argsort(requested_to_canonical_axes))
    canonical_left = (canonical_tokens[0][1], canonical_tokens[1][1])
    canonical_right = (canonical_tokens[2][1], canonical_tokens[3][1])
    return QuartetBlockReference(
        key=(canonical_left, canonical_right),
        requested_to_canonical_axes=requested_to_canonical_axes,
        canonical_to_requested_axes=canonical_to_requested_axes,
    )


def _orient_quartet_block(block: np.ndarray, reference: QuartetBlockReference) -> np.ndarray:
    return np.transpose(block, axes=reference.canonical_to_requested_axes)


def _intern_quartet_block(
    block_map: dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray],
    reference: QuartetBlockReference,
    requested_block: np.ndarray,
) -> QuartetBlockReference:
    if reference.key not in block_map:
        block_map[reference.key] = np.transpose(requested_block, axes=reference.requested_to_canonical_axes).copy()
    return reference


def build_structured_eri_repository(
    mol: gto.Mole,
    eri: np.ndarray,
    threshold: float = 1.0e-12,
) -> StructuredERIRepository:
    ao_ang = ao_angular_momentum_for_each_ao(mol)
    unique_l = sorted(set(int(value) for value in ao_ang))
    l_indices = {l_value: np.where(ao_ang == l_value)[0] for l_value in unique_l}
    radial_counts = {l_value: int(l_indices[l_value].size // (2 * l_value + 1)) for l_value in unique_l}

    block_map: dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray] = {}
    active_quartets: list[ActiveERIQuartet] = []
    for l1 in unique_l:
        idx1 = l_indices[l1]
        for l2 in unique_l:
            idx2 = l_indices[l2]
            for l3 in unique_l:
                idx3 = l_indices[l3]
                for l4 in unique_l:
                    idx4 = l_indices[l4]
                    coulomb_block = eri[np.ix_(idx1, idx2, idx3, idx4)]
                    max_abs = float(np.max(np.abs(coulomb_block))) if coulomb_block.size else 0.0
                    if max_abs <= threshold:
                        continue
                    exchange_block = eri[np.ix_(idx1, idx3, idx2, idx4)]
                    coulomb_ref = _intern_quartet_block(
                        block_map,
                        _canonicalize_quartet_axes(l1, l2, l3, l4),
                        coulomb_block,
                    )
                    exchange_ref = _intern_quartet_block(
                        block_map,
                        _canonicalize_quartet_axes(l1, l3, l2, l4),
                        exchange_block,
                    )
                    active_quartets.append(
                        ActiveERIQuartet(
                            l_values=(l1, l2, l3, l4),
                            labels=(
                                angular_momentum_label(l1),
                                angular_momentum_label(l2),
                                angular_momentum_label(l3),
                                angular_momentum_label(l4),
                            ),
                            idx_p=idx1,
                            idx_q=idx2,
                            idx_r=idx3,
                            idx_s=idx4,
                            ao_shape=(int(idx1.size), int(idx2.size), int(idx3.size), int(idx4.size)),
                            reduced_radial_shape=(
                                radial_counts[l1],
                                radial_counts[l2],
                                radial_counts[l3],
                                radial_counts[l4],
                            ),
                            max_abs=max_abs,
                            frobenius_norm=float(np.linalg.norm(coulomb_block)),
                            coulomb_ref=coulomb_ref,
                            exchange_ref=exchange_ref,
                        )
                    )
    return StructuredERIRepository(
        threshold=threshold,
        block_map=block_map,
        active_quartets=active_quartets,
    )


def build_reduced_radial_eri_repository(
    mol: gto.Mole,
    eri: np.ndarray,
    threshold: float = 1.0e-12,
) -> ReducedRadialERIRepository:
    ao_ang = ao_angular_momentum_for_each_ao(mol)
    unique_l = sorted(set(int(value) for value in ao_ang))
    l_indices = {l_value: np.where(ao_ang == l_value)[0] for l_value in unique_l}

    pair_blocks: dict[tuple[int, int], ReducedRadialPairBlock] = {}
    for l_output in unique_l:
        idx_out = l_indices[l_output]
        degeneracy_out = 2 * l_output + 1
        nrad_out = idx_out.size // degeneracy_out
        for l_density in unique_l:
            idx_den = l_indices[l_density]
            degeneracy_den = 2 * l_density + 1
            nrad_den = idx_den.size // degeneracy_den

            coulomb_block = eri[np.ix_(idx_out, idx_out, idx_den, idx_den)].reshape(
                nrad_out,
                degeneracy_out,
                nrad_out,
                degeneracy_out,
                nrad_den,
                degeneracy_den,
                nrad_den,
                degeneracy_den,
            )
            exchange_block = eri[np.ix_(idx_out, idx_den, idx_out, idx_den)].reshape(
                nrad_out,
                degeneracy_out,
                nrad_den,
                degeneracy_den,
                nrad_out,
                degeneracy_out,
                nrad_den,
                degeneracy_den,
            )

            reduced_coulomb = np.einsum("aibicjdj->abcd", coulomb_block, optimize=True) / degeneracy_out
            reduced_exchange = np.einsum("aicjbidj->abcd", exchange_block, optimize=True) / degeneracy_out

            coulomb_max_abs = float(np.max(np.abs(reduced_coulomb))) if reduced_coulomb.size else 0.0
            exchange_max_abs = float(np.max(np.abs(reduced_exchange))) if reduced_exchange.size else 0.0
            if max(coulomb_max_abs, exchange_max_abs) <= threshold:
                continue

            pair_blocks[(l_output, l_density)] = ReducedRadialPairBlock(
                l_output=l_output,
                l_density=l_density,
                output_label=angular_momentum_label(l_output),
                density_label=angular_momentum_label(l_density),
                output_radial_functions=int(nrad_out),
                density_radial_functions=int(nrad_den),
                coulomb_tensor=reduced_coulomb,
                exchange_tensor=reduced_exchange,
                coulomb_max_abs=coulomb_max_abs,
                exchange_max_abs=exchange_max_abs,
                coulomb_frobenius_norm=float(np.linalg.norm(reduced_coulomb)),
                exchange_frobenius_norm=float(np.linalg.norm(reduced_exchange)),
            )

    return ReducedRadialERIRepository(threshold=threshold, pair_blocks=pair_blocks)


def build_active_eri_quartets(
    mol: gto.Mole,
    eri: np.ndarray,
    threshold: float = 1.0e-12,
) -> list[ActiveERIQuartet]:
    return build_structured_eri_repository(mol, eri, threshold=threshold).active_quartets


def build_rhf_fock_from_reduced_radial_eri(
    h_core: np.ndarray,
    density: np.ndarray,
    mol: gto.Mole,
    reduced_eri: ReducedRadialERIRepository,
) -> np.ndarray:
    reduced_density = build_reduced_density_by_l(density, mol)
    reduced_hcore = {
        l_value: reduced_matrix_for_l(h_core, mol, l_value)[0]
        for l_value in reduced_density
    }
    reduced_fock = {l_value: np.array(reduced_hcore[l_value], copy=True) for l_value in reduced_density}

    for (l_output, l_density), pair_block in reduced_eri.pair_blocks.items():
        density_block = reduced_density.get(l_density)
        if density_block is None or density_block.size == 0:
            continue
        reduced_fock[l_output] += np.einsum(
            "rs,pqrs->pq",
            density_block,
            pair_block.coulomb_tensor,
            optimize=True,
        )
        reduced_fock[l_output] -= 0.5 * np.einsum(
            "rs,pqrs->pq",
            density_block,
            pair_block.exchange_tensor,
            optimize=True,
        )

    fock = np.zeros_like(h_core)
    for l_value, reduced_block in reduced_fock.items():
        fock += expand_reduced_matrix_for_l(reduced_block, mol, l_value)
    return fock


def build_fock_from_active_quartets(
    h_core: np.ndarray,
    density_or_eri: np.ndarray,
    structured_eri_or_density: StructuredERIRepository | np.ndarray,
    active_quartets: list[ActiveERIQuartet] | None = None,
) -> np.ndarray:
    if active_quartets is None:
        density = density_or_eri
        structured_eri = structured_eri_or_density
        eri = None
    else:
        eri = density_or_eri
        density = structured_eri_or_density
        structured_eri = None

    fock = np.array(h_core, copy=True)
    quartet_iterable = structured_eri.active_quartets if structured_eri is not None else active_quartets
    for quartet in quartet_iterable:
        if structured_eri is not None:
            coulomb_block = _orient_quartet_block(structured_eri.block_map[quartet.coulomb_ref.key], quartet.coulomb_ref)
            exchange_block = _orient_quartet_block(
                structured_eri.block_map[quartet.exchange_ref.key],
                quartet.exchange_ref,
            )
        else:
            coulomb_block = eri[np.ix_(quartet.idx_p, quartet.idx_q, quartet.idx_r, quartet.idx_s)]
            exchange_block = eri[np.ix_(quartet.idx_p, quartet.idx_r, quartet.idx_q, quartet.idx_s)]
        density_rs = density[np.ix_(quartet.idx_r, quartet.idx_s)]
        coulomb_contrib = np.einsum(
            "rs,pqrs->pq",
            density_rs,
            coulomb_block,
            optimize=True,
        )
        exchange_contrib = np.einsum(
            "rs,prqs->pq",
            density_rs,
            exchange_block,
            optimize=True,
        )
        fock[np.ix_(quartet.idx_p, quartet.idx_q)] += coulomb_contrib - 0.5 * exchange_contrib
    return fock


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


def build_uhf_fock_from_active_quartets(
    h_core: np.ndarray,
    density_alpha_or_eri: np.ndarray,
    density_beta_or_alpha: np.ndarray,
    structured_eri_or_beta: StructuredERIRepository | np.ndarray,
    active_quartets: list[ActiveERIQuartet] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if active_quartets is None:
        density_alpha = density_alpha_or_eri
        density_beta = density_beta_or_alpha
        structured_eri = structured_eri_or_beta
        eri = None
    else:
        eri = density_alpha_or_eri
        density_alpha = density_beta_or_alpha
        density_beta = structured_eri_or_beta
        structured_eri = None

    fock_alpha = np.array(h_core, copy=True)
    fock_beta = np.array(h_core, copy=True)
    density_total = density_alpha + density_beta

    quartet_iterable = structured_eri.active_quartets if structured_eri is not None else active_quartets
    for quartet in quartet_iterable:
        if structured_eri is not None:
            eri_coulomb = _orient_quartet_block(structured_eri.block_map[quartet.coulomb_ref.key], quartet.coulomb_ref)
            eri_exchange = _orient_quartet_block(
                structured_eri.block_map[quartet.exchange_ref.key],
                quartet.exchange_ref,
            )
        else:
            eri_coulomb = eri[np.ix_(quartet.idx_p, quartet.idx_q, quartet.idx_r, quartet.idx_s)]
            eri_exchange = eri[np.ix_(quartet.idx_p, quartet.idx_r, quartet.idx_q, quartet.idx_s)]
        density_total_rs = density_total[np.ix_(quartet.idx_r, quartet.idx_s)]
        density_alpha_rs = density_alpha[np.ix_(quartet.idx_r, quartet.idx_s)]
        density_beta_rs = density_beta[np.ix_(quartet.idx_r, quartet.idx_s)]

        coulomb_block = np.einsum("rs,pqrs->pq", density_total_rs, eri_coulomb, optimize=True)
        exchange_alpha_block = np.einsum("rs,prqs->pq", density_alpha_rs, eri_exchange, optimize=True)
        exchange_beta_block = np.einsum("rs,prqs->pq", density_beta_rs, eri_exchange, optimize=True)

        output_index = np.ix_(quartet.idx_p, quartet.idx_q)
        fock_alpha[output_index] += coulomb_block - exchange_alpha_block
        fock_beta[output_index] += coulomb_block - exchange_beta_block

    return fock_alpha, fock_beta


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
    mo_occ_alpha: np.ndarray | None = None,
    mo_occ_beta: np.ndarray | None = None,
) -> tuple[float, float, float]:
    if mo_occ_alpha is None:
        occ_alpha = coeff_alpha[:, :nalpha]
    else:
        occ_alpha = coeff_alpha[:, mo_occ_alpha > 1.0e-12]
    if mo_occ_beta is None:
        occ_beta = coeff_beta[:, :nbeta]
    else:
        occ_beta = coeff_beta[:, mo_occ_beta > 1.0e-12]
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
    structured_eri = build_structured_eri_repository(mol, eri, threshold=threshold)
    reduced_radial_eri = build_reduced_radial_eri_repository(mol, eri, threshold=threshold)
    active_quartet_map = {quartet.l_values: quartet for quartet in structured_eri.active_quartets}

    ao_ang = ao_angular_momentum_for_each_ao(mol)
    unique_l = sorted(set(int(value) for value in ao_ang))
    l_indices = {l_value: np.where(ao_ang == l_value)[0] for l_value in unique_l}
    radial_counts = {l_value: int(l_indices[l_value].size // (2 * l_value + 1)) for l_value in unique_l}

    quartet_summaries: list[dict[str, object]] = []
    active_count = 0
    for l1 in unique_l:
        idx1 = l_indices[l1]
        for l2 in unique_l:
            idx2 = l_indices[l2]
            for l3 in unique_l:
                idx3 = l_indices[l3]
                for l4 in unique_l:
                    idx4 = l_indices[l4]
                    quartet_key = (l1, l2, l3, l4)
                    active_quartet = active_quartet_map.get(quartet_key)
                    if active_quartet is None:
                        max_abs = 0.0
                        fro_norm = 0.0
                        is_active = False
                    else:
                        max_abs = active_quartet.max_abs
                        fro_norm = active_quartet.frobenius_norm
                        is_active = True
                        active_count += 1
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
    reduced_pair_summaries = [
        {
            "labels": [pair_block.output_label, pair_block.density_label],
            "l_values": [pair_block.l_output, pair_block.l_density],
            "reduced_radial_shape": [
                pair_block.output_radial_functions,
                pair_block.output_radial_functions,
                pair_block.density_radial_functions,
                pair_block.density_radial_functions,
            ],
            "coulomb_max_abs": pair_block.coulomb_max_abs,
            "exchange_max_abs": pair_block.exchange_max_abs,
            "coulomb_frobenius_norm": pair_block.coulomb_frobenius_norm,
            "exchange_frobenius_norm": pair_block.exchange_frobenius_norm,
        }
        for _, pair_block in sorted(reduced_radial_eri.pair_blocks.items())
    ]
    dominant_reduced_pairs = sorted(
        reduced_pair_summaries,
        key=lambda item: max(item["coulomb_frobenius_norm"], item["exchange_frobenius_norm"]),
        reverse=True,
    )
    return {
        "threshold": threshold,
        "total_angular_quartets": len(quartet_summaries),
        "active_angular_quartets": active_count,
        "inactive_angular_quartets": len(quartet_summaries) - active_count,
        "active_ratio": float(active_count / len(quartet_summaries)) if quartet_summaries else 0.0,
        "unique_canonical_blocks": len(structured_eri.block_map),
        "active_reduced_radial_pair_blocks": len(reduced_radial_eri.pair_blocks),
        "reduced_pair_compression_ratio": (
            float(active_count / len(reduced_radial_eri.pair_blocks))
            if reduced_radial_eri.pair_blocks
            else 1.0
        ),
        "dominant_reduced_radial_pairs": dominant_reduced_pairs[:12],
        "dominant_active_quartets": active_sorted[:12],
        "quartet_summaries": quartet_summaries,
        "reduced_radial_pair_summaries": reduced_pair_summaries,
    }
