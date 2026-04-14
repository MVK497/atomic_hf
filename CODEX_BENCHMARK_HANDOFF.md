# Codex Benchmark Handoff

This file is for a Codex session running on another machine.

Its purpose is simple: continue the `atomic_hf` benchmark work on a faster computer, especially for the transition-metal and heavier-element region.

## Main Goal

Run the benchmark workflow for this project on a stronger machine and collect reliable JSON outputs for:

1. `Z = 1 .. 20` as a stable baseline
2. `Z = 21 .. 30` as the main 3d transition-metal stress test
3. `Z = 1 .. 102` as the long full sweep

## Important Context

- This repository contains an atomic-only Hartree-Fock project based on PySCF/libcint.
- The benchmark path already uses:
  - subprocess-isolated entries
  - checkpointed JSON output
  - resumable runs
  - per-entry timeout control
- The current bottleneck is mainly the `Sc` and later transition-metal region.
- On the original MacBook, some cases appear to be slow rather than fundamentally impossible.
- Local testing already showed:
  - `Sc` with `sto-3g` can converge, but may take about 55 seconds
  - `Ti` with `sto-3g` can also converge
- Therefore, the faster machine should be used mainly to finish the benchmark sweep more reliably.

## What To Do First

1. Make sure the repository is up to date.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Confirm that the benchmark suite script works:

```bash
python run_benchmark_suite.py --list-stages
python run_benchmark_suite.py --stage smoke --no-resume
```

## Recommended Benchmark Order

Run the stages in this order:

### 1. Baseline

```bash
python run_benchmark_suite.py --stage first20 --no-resume
```

Expected result:

- Should usually finish successfully.
- JSON output should be written into `benchmarks/first20_sto-3g_noref.json`.

### 2. Transition Metals

```bash
python run_benchmark_suite.py --stage transition3d --no-resume
```

If this is still too slow, retry with a larger timeout:

```bash
python run_benchmark_suite.py --stage transition3d --timeout 300 --no-resume
```

Expected result:

- Main target is to see how many of `Sc` through `Zn` can now complete.
- The most important output is `benchmarks/transition3d_sto-3g_noref.json`.

### 3. Full Sweep

```bash
python run_benchmark_suite.py --stage full --timeout 180
```

If needed:

```bash
python run_benchmark_suite.py --stage full --timeout 300
```

Notes:

- Do not use `--no-resume` unless you intentionally want to restart.
- The full run is expected to be long.
- The checkpoint JSON can be resumed after interruption.

## If A Benchmark Fails

When a stage fails or many entries time out, inspect the JSON file and summarize:

1. `last_completed_atomic_number`
2. `successful_cases`
3. `failed_cases`
4. which elements failed
5. whether failure was:
   - timeout
   - non-convergence
   - basis fallback issue
   - implementation error / traceback

Useful quick inspection command:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('benchmarks/transition3d_sto-3g_noref.json')
data = json.loads(path.read_text())
print('last_completed_atomic_number =', data.get('last_completed_atomic_number'))
print('successful_cases =', data.get('successful_cases'))
print('failed_cases =', data.get('failed_cases'))
for row in data.get('entries', []):
    if row.get('status') != 'ok':
        print(row.get('atomic_number'), row.get('symbol'), row.get('basis'), row.get('message'))
PY
```

## If More Work Is Needed

If `Sc` and later 3d atoms still remain problematic, the next coding target should be:

1. improve open-shell atomic UHF robustness further
2. improve benchmark timeout/fallback policy for slow-but-convergent transition metals
3. avoid wasting time on heavier fallback bases when `sto-3g` itself is the desired benchmark basis

## Files You Should Know

- `run_benchmark_suite.py`
  Stage-based benchmark runner
- `run_atomic_benchmark.py`
  Lower-level benchmark CLI
- `atomic_hf/benchmark.py`
  Benchmark logic, subprocess entry execution, checkpoint JSON writing
- `atomic_hf/uhf.py`
  Atomic UHF solver
- `BENCHMARKS.md`
  Human-facing benchmark usage guide

## Preferred Deliverables

After running on the stronger machine, report back with:

1. which stage(s) were run
2. the exact command(s) used
3. whether the run completed or resumed
4. the path(s) of the generated JSON files
5. summary statistics:
   - success count
   - failure count
   - last completed atomic number
6. a short diagnosis of the remaining hard cases

## Optional Windows Example

PowerShell:

```powershell
cd D:\path\to\atomic_hf
python -m pip install -r requirements.txt
python run_benchmark_suite.py --stage smoke --no-resume
python run_benchmark_suite.py --stage first20 --no-resume
python run_benchmark_suite.py --stage transition3d --timeout 300 --no-resume
python run_benchmark_suite.py --stage full --timeout 300
```
