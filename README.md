# atomic_hf

`atomic_hf` is a Python project for exploring atom-specific Hartree-Fock implementations on top of the PySCF/libcint integral stack.

The code focuses on single-center atoms and ions rather than general molecules. It is designed to make atom-oriented ideas explicit: angular-momentum blocking, atomic occupations, contraction-aware basis analysis, one-center integral structure, and symmetry reuse in Fock builds.

## Current Features

- Atomic-only `RHF` and `UHF`
- `DIIS` acceleration
- Early-iteration `damping` and virtual-space `level shift`
- Automatic high-spin default `2S` inference from PySCF atomic configuration data
- Atomic initial guesses:
  - `atom`: PySCF atomic-density guess
  - `core`: blocked one-electron guess
- Blocked `RHF` diagonalization by angular momentum
- Blocked `UHF` diagonalization for separate alpha/beta channels
- One-center one-electron integral reduction analysis
- Structured one-center ERI quartet repository with canonical-block reuse
- Quartet-screened `RHF` / `UHF` Fock construction
- Basis engineering analysis:
  - segmented vs general contraction detection
  - primitive / contraction statistics
  - angular-momentum-resolved contraction summaries
- Atomic electron-configuration summary with PySCF-informed exception handling
- Benchmark sweep tooling with automatic basis fallback for heavier atoms
- Automated tests

## Project Layout

- `run_atomic_hf.py`: command-line entry point
- `run_atomic_benchmark.py`: benchmark sweep entry point
- `atomic_hf/atom.py`: atomic specification, electron counting, configuration summary, basis-shell analysis
- `atomic_hf/blocks.py`: blocked linear algebra, DIIS helpers, initial guesses, stabilization tools, one-center integral analysis, structured ERI blocks
- `atomic_hf/benchmark.py`: benchmark sweep helpers and JSON reporting
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
python3 run_atomic_hf.py O --method uhf
```

Heavier atom with a larger basis:

```bash
python3 run_atomic_hf.py Fe --method uhf --spin 4 --basis def2-svp
```

Use a one-electron guess instead of the atomic-density guess:

```bash
python3 run_atomic_hf.py Ne --method rhf --guess core
```

Tune SCF stabilization:

```bash
python3 run_atomic_hf.py Cr --method uhf --damping 0.25 --damping-cycles 8 --level-shift 0.7
```

Print SCF history:

```bash
python3 run_atomic_hf.py Ne --method rhf --show-history
```

Small benchmark sweep:

```bash
python3 run_atomic_benchmark.py --min-z 1 --max-z 10 --basis sto-3g
```

Benchmark with a JSON report:

```bash
python3 run_atomic_benchmark.py --min-z 1 --max-z 30 --basis sto-3g --json-output benchmarks_sto3g.json
```

## Output Highlights

For each run, the program reports:

- atomic number, charge, and electron count
- estimated electron configuration, angular-shell profile, and default spin
- orbital energies and occupations
- initial-guess mode and SCF stabilization settings
- basis-shell summary by angular momentum
- basis engineering summary, including contraction structure
- angular-momentum block structure used in the solver
- one-center one-electron reduction statistics
- dominant one-center two-electron integral quartets
- active quartet count and unique canonical quartet-block count

## Tests

```bash
cd /Users/roxy/ROXY_Projects/projects/atomic_hf
pytest
```

## Current Limitations

- `RHF` is limited to closed-shell atoms and ions
- Heavy atoms may require a larger or relativistic basis; benchmark mode can fall back automatically, but ordinary CLI runs still use the basis you request
- The one-center ERI acceleration currently reuses canonical quartet blocks and skips inactive quartets, but it does not yet implement a fully radial/Gaunt-factor atomic ERI formulation
- General-contraction structure is analyzed, but contraction-aware integral/Fock acceleration is still incomplete
- The project does not yet include relativistic Hamiltonians, ECP-aware workflows, or production-grade occupation control for every exotic ion/state
- A full `Z = 1 .. 102` production benchmark is still a workload for the benchmark tool, not a completed published data set inside the repository
