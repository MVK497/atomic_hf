# Benchmark Guide

This file is a practical guide for running longer `atomic_hf` benchmarks on a faster machine.

## Recommended Workflow

1. Clone the repository on the target machine.
2. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Start with a smoke test:

```bash
python run_benchmark_suite.py --stage smoke
```

4. Then run the first stable block:

```bash
python run_benchmark_suite.py --stage first20
```

5. For the 3d transition metals, use the dedicated stage:

```bash
python run_benchmark_suite.py --stage transition3d
```

6. When the machine is stable enough, launch the full sweep:

```bash
python run_benchmark_suite.py --stage full --timeout 180
```

## Why Use the Suite Script?

- It generates checkpointed JSON files automatically.
- It uses stage-specific timeout defaults.
- It is easy to resume after interruption.
- It is easier to share with another Codex session or another computer.

## Typical Output Files

By default, the suite writes into the `benchmarks/` directory:

- `benchmarks/smoke_sto-3g_noref.json`
- `benchmarks/first20_sto-3g_noref.json`
- `benchmarks/transition3d_sto-3g_noref.json`
- `benchmarks/full_sto-3g_noref.json`

## Useful Variants

Compute references as well:

```bash
python run_benchmark_suite.py --stage first20 --reference
```

Use another basis:

```bash
python run_benchmark_suite.py --stage transition3d --basis def2-svp
```

Force a fresh run:

```bash
python run_benchmark_suite.py --stage full --no-resume
```

## Windows Notes

On Windows, the same commands work in PowerShell or `cmd`, as long as Python and the project dependencies are installed.

Example PowerShell session:

```powershell
cd D:\path\to\atomic_hf
python -m pip install -r requirements.txt
python run_benchmark_suite.py --stage smoke
python run_benchmark_suite.py --stage transition3d
```
