# atomic_hf

`atomic_hf` is a small Python project for exploring atom-specific Hartree-Fock implementations on top of the PySCF/libcint integral stack.

The project focuses on single-center atoms and ions rather than general molecules. Its goal is to expose how atomic structure, angular-momentum blocking, basis contraction patterns, and one-/two-electron integral structure can be used in an atom-oriented HF code.

## Current Features

- Atomic-only `RHF` and `UHF`
- `DIIS` acceleration
- Blocked `RHF` diagonalization by angular momentum
- Blocked `UHF` diagonalization for separate alpha/beta channels
- One-center one-electron integral reduction analysis
- One-center two-electron integral quartet analysis
- Basis engineering analysis:
  - segmented vs general contraction detection
  - primitive / contraction statistics
  - angular-momentum-resolved contraction summaries
- Atomic electron-configuration summary based on an Aufbau-style teaching model
- Basic automated tests

## Project Layout

- `run_atomic_hf.py`: command-line entry point
- `atomic_hf/atom.py`: atomic specification, electron counting, configuration summary, basis-shell analysis
- `atomic_hf/blocks.py`: blocked linear algebra, DIIS helpers, one-center integral analysis
- `atomic_hf/rhf.py`: blocked atomic RHF solver
- `atomic_hf/uhf.py`: blocked atomic UHF solver
- `atomic_hf/cli.py`: command-line interface and formatted output
- `tests/test_atomic_hf.py`: regression and feature tests

## Installation

```bash
cd /Users/roxy/ROXY_Projects/projects/atomic_hf
python3 -m pip install -r requirements.txt
```

## Usage

Closed-shell atomic RHF:

```bash
python3 run_atomic_hf.py He --method rhf
```

Open-shell atomic UHF:

```bash
python3 run_atomic_hf.py O --method uhf --spin 2
```

Heavier atom with a larger basis:

```bash
python3 run_atomic_hf.py Fe --method uhf --spin 4 --basis def2-svp
```

Print SCF history:

```bash
python3 run_atomic_hf.py Ne --method rhf --show-history
```

## Output Highlights

For each run, the program reports:

- atomic number, charge, and electron count
- estimated electron configuration and default spin
- orbital energies and occupations
- basis-shell summary by angular momentum
- basis engineering summary, including contraction structure
- angular-momentum block structure used in the solver
- one-center one-electron reduction statistics
- dominant one-center two-electron integral quartets

## Tests

```bash
cd /Users/roxy/ROXY_Projects/projects/atomic_hf
pytest
```

## Current Limitations

- `RHF` is limited to closed-shell atoms and ions
- `UHF` currently requires an explicit `--spin`
- The electron-configuration summary is a teaching-oriented Aufbau model and does not yet include all heavy-element exceptions
- One-center ERI structure is analyzed, but not yet fully exploited to reduce Fock-build cost
- General-contraction structure is detected and summarized, but not yet fully used for contraction-aware acceleration
- The project is not yet benchmarked as a production tool for all elements up to `Z = 102`
