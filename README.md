# atomic_hf

`atomic_hf` is a Python project for exploring atom-specific Hartree-Fock implementations on top of the PySCF/libcint integral stack.  
`atomic_hf` 是一个基于 PySCF/libcint 积分栈、用于探索“单原子特化”Hartree-Fock 实现的 Python 项目。

The code focuses on single-center atoms and ions rather than general molecules. It is designed to make atom-oriented ideas explicit: angular-momentum blocking, atomic occupations, contraction-aware basis analysis, one-center integral structure, and symmetry reuse in Fock builds.  
这套代码聚焦于单中心原子与离子，而不是一般分子体系。它的目标是把“面向原子”的计算思想明确写出来，例如角动量分块、原子占据、考虑收缩结构的基组分析、单中心积分结构，以及在 Fock 构造中复用对称性。

## Current Features

## 当前特性

- Atomic-only `RHF` and `UHF`  
  仅面向原子的 `RHF` 与 `UHF`
- `DIIS` acceleration  
  支持 `DIIS` 加速
- Early-iteration `damping` and virtual-space `level shift`  
  支持早期迭代 `damping` 与虚轨道空间 `level shift`
- Automatic high-spin default `2S` inference from PySCF atomic configuration data  
  基于 PySCF 原子组态数据自动推断高自旋默认 `2S`
- Atomic initial guesses:  
  支持原子型初猜：
  - `atom`: PySCF atomic-density guess  
    `atom`：PySCF 的原子密度初猜
  - `core`: blocked one-electron guess  
    `core`：按角动量分块的一电子初猜
- Blocked `RHF` diagonalization by angular momentum  
  按角动量分块的 `RHF` 对角化
- Blocked `UHF` diagonalization for separate alpha/beta channels  
  对 alpha/beta 自旋通道分别进行分块 `UHF` 对角化
- Two UHF occupation modes:  
  两种 UHF 占据模式：
  - `integer`: conventional spin-orbital filling within each angular block  
    `integer`：在每个角动量块内按常规自旋轨道方式整数填充
  - `spherical_average`: fractional atomic-shell filling for more radial/open-shell-friendly behavior  
    `spherical_average`：按原子壳层进行分数平均填充，更适合径向化与开放壳层场景
- One-center one-electron integral reduction analysis  
  单中心一电子积分降维分析
- Structured one-center ERI quartet repository with canonical-block reuse  
  带 canonical block 复用的单中心双电子积分 quartet 结构仓库
- Gaunt/Wigner-channel one-center ERI decomposition for radial-plus-angular Fock construction  
  使用 Gaunt/Wigner 通道分解单中心 ERI，以支持“径向部分 + 角向部分”的 Fock 构造
- Exact reduced-radial RHF Fock construction for spherically averaged closed-shell atoms  
  面向球平均闭壳层原子的精确 reduced-radial RHF Fock 构造
- Exact UHF Fock decomposition into Gaunt-channel spherical part plus residual quartet correction  
  将 UHF Fock 精确分解为 Gaunt 通道球对称部分与 residual quartet 修正部分
- Quartet-screened `RHF` / `UHF` Fock construction  
  支持 quartet-screened 的 `RHF` / `UHF` Fock 构造
- Basis engineering analysis:  
  支持基组工程分析：
  - segmented vs general contraction detection  
    segmented 与 general contraction 的识别
  - primitive / contraction statistics  
    primitive / contraction 统计
  - angular-momentum-resolved contraction summaries  
    按角动量分辨的 contraction 汇总
- Atomic electron-configuration summary with PySCF-informed exception handling  
  带有 PySCF 例外组态信息修正的原子电子组态总结
- Benchmark sweep tooling with automatic basis fallback for heavier atoms  
  面向较重元素的 benchmark 扫描工具，并支持自动基组 fallback
- Checkpointed/resumable benchmark JSON output and per-entry timeout control  
  支持可断点续跑的 benchmark JSON 输出，以及逐元素 timeout 控制
- Automated tests  
  提供自动化测试

## Project Layout

## 项目结构

- `run_atomic_hf.py`: command-line entry point  
  `run_atomic_hf.py`：命令行入口
- `run_atomic_benchmark.py`: benchmark sweep entry point  
  `run_atomic_benchmark.py`：benchmark 扫描入口
- `run_benchmark_suite.py`: named benchmark-stage runner for smoke / transition / full sweeps  
  `run_benchmark_suite.py`：用于 smoke / 过渡金属 / 全量扫描的分阶段 benchmark 入口
- `atomic_hf/atom.py`: atomic specification, electron counting, configuration summary, basis-shell analysis  
  `atomic_hf/atom.py`：原子规格、电子数统计、电子组态总结、基组壳层分析
- `atomic_hf/blocks.py`: blocked linear algebra, DIIS helpers, initial guesses, stabilization tools, one-center integral analysis, structured ERI blocks  
  `atomic_hf/blocks.py`：分块线性代数、DIIS 辅助工具、初猜、稳定化工具、单中心积分分析、结构化 ERI 分块
- `atomic_hf/benchmark.py`: benchmark sweep helpers and JSON reporting  
  `atomic_hf/benchmark.py`：benchmark 扫描辅助函数与 JSON 报告输出
- `atomic_hf/rhf.py`: blocked atomic RHF solver  
  `atomic_hf/rhf.py`：分块原子 RHF 求解器
- `atomic_hf/uhf.py`: blocked atomic UHF solver  
  `atomic_hf/uhf.py`：分块原子 UHF 求解器
- `atomic_hf/cli.py`: command-line interface and formatted output  
  `atomic_hf/cli.py`：命令行接口与格式化输出
- `tests/test_atomic_hf.py`: regression and feature tests  
  `tests/test_atomic_hf.py`：回归测试与功能测试
- `BENCHMARKS.md`: practical guide for running longer sweeps on another machine  
  `BENCHMARKS.md`：在另一台机器上运行长程 benchmark 的实用指南

## Installation

## 安装

```bash
cd /Users/roxy/ROXY_Projects/projects/atomic_hf
python3 -m pip install -r requirements.txt
```

## Usage

## 用法

Closed-shell atomic RHF:  
闭壳层原子的 RHF：

```bash
python3 run_atomic_hf.py He --method rhf
```

Open-shell atomic UHF:  
开放壳层原子的 UHF：

```bash
python3 run_atomic_hf.py O --method uhf
```

Heavier atom with a larger basis:  
使用更大基组计算较重原子：

```bash
python3 run_atomic_hf.py Fe --method uhf --spin 4 --basis def2-svp
```

Use a one-electron guess instead of the atomic-density guess:  
使用一电子初猜而不是原子密度初猜：

```bash
python3 run_atomic_hf.py Ne --method rhf --guess core
```

Tune SCF stabilization:  
调整 SCF 稳定化参数：

```bash
python3 run_atomic_hf.py Cr --method uhf --damping 0.25 --damping-cycles 8 --level-shift 0.7
```

Use the more atomic, shell-averaged open-shell occupation model:  
使用更“原子化”的壳层平均开放壳层占据模型：

```bash
python3 run_atomic_hf.py O --method uhf --occupation-mode spherical_average
```

Print SCF history:  
打印 SCF 历史：

```bash
python3 run_atomic_hf.py Ne --method rhf --show-history
```

Small benchmark sweep:  
小规模 benchmark 扫描：

```bash
python3 run_atomic_benchmark.py --min-z 1 --max-z 10 --basis sto-3g
```

Benchmark with a JSON report:  
输出 JSON 报告的 benchmark：

```bash
python3 run_atomic_benchmark.py --min-z 1 --max-z 30 --basis sto-3g --json-output benchmarks_sto3g.json
```

Resume a long benchmark sweep with per-entry timeouts:  
对长程 benchmark 扫描启用断点续跑与逐元素超时控制：

```bash
python3 run_atomic_benchmark.py --min-z 1 --max-z 102 --basis sto-3g --no-reference --json-output benchmarks/z1_z102_auto_noref.json --entry-timeout 20
```

Use the benchmark suite runner on another machine:  
在另一台机器上使用 benchmark 套件入口：

```bash
python3 run_benchmark_suite.py --stage smoke
python3 run_benchmark_suite.py --stage transition3d
python3 run_benchmark_suite.py --stage full --timeout 180
```

## Output Highlights

## 输出内容概览

For each run, the program reports:  
对每一次运行，程序会输出：

- atomic number, charge, and electron count  
  原子序数、电荷与电子数
- estimated electron configuration, angular-shell profile, and default spin  
  估计电子组态、角动量壳层分布与默认自旋
- orbital energies and occupations  
  轨道能量与占据数
- initial-guess mode and SCF stabilization settings  
  初猜模式与 SCF 稳定化设置
- basis-shell summary by angular momentum  
  按角动量分类的基组壳层总结
- basis engineering summary, including contraction structure  
  基组工程总结，包括 contraction 结构
- angular-momentum block structure used in the solver  
  求解器中使用的角动量分块结构
- one-center one-electron reduction statistics  
  单中心一电子降维统计信息
- dominant one-center two-electron integral quartets  
  主导的单中心双电子积分 quartet
- dominant reduced-radial pair blocks for the one-center ERI tensor  
  单中心 ERI 张量中主导的 reduced-radial pair block
- active quartet count and unique canonical quartet-block count  
  active quartet 数量与唯一 canonical quartet block 数量
- dominant Gaunt/Wigner channel terms in the one-center ERI decomposition  
  单中心 ERI 分解中主导的 Gaunt/Wigner 通道项

## Tests

## 测试

```bash
cd /Users/roxy/ROXY_Projects/projects/atomic_hf
pytest
```

## Current Limitations

## 当前局限

- `RHF` is limited to closed-shell atoms and ions  
  `RHF` 目前仅支持闭壳层原子与离子
- Heavy atoms may require a larger or relativistic basis; benchmark mode can fall back automatically, but ordinary CLI runs still use the basis you request  
  重元素可能需要更大的基组或相对论基组；benchmark 模式可以自动 fallback，但普通 CLI 运行仍然只使用你指定的基组
- The one-center ERI acceleration now includes Gaunt/Wigner-channel factorization plus residual-quartet correction, but there is still room to push further toward a fully production-grade atomic ERI engine  
  单中心 ERI 加速目前已经包含 Gaunt/Wigner 通道分解与 residual quartet 修正，但距离完全工程化、生产级的 atomic ERI 引擎仍有提升空间
- General-contraction structure is analyzed, but contraction-aware integral/Fock acceleration is still incomplete  
  虽然已经能分析 general contraction 结构，但真正利用 contraction 结构来加速积分/Fock 构造仍未完全实现
- The project does not yet include relativistic Hamiltonians, ECP-aware workflows, or production-grade occupation control for every exotic ion/state  
  该项目尚未包含相对论哈密顿量、面向 ECP 的完整工作流，也还没有覆盖所有特殊离子/电子态的生产级占据控制
- A full `Z = 1 .. 102` production benchmark is still a workload for the benchmark tool, not a completed published data set inside the repository  
  完整的 `Z = 1 .. 102` 生产级 benchmark 目前仍然是 benchmark 工具要执行的任务，而不是仓库中已经固定发布完成的数据集
