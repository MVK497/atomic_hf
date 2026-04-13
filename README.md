# Atomic HF Teaching Project

这是一个和 `simple_hf_water` 分开的新项目，目标是逐步走向“单原子专用 Hartree-Fock 程序”。

当前第一版先完成前两步：

- 做一个最小可运行的单原子 `RHF/UHF` 程序
- 开始真正利用“单原子特性”
  - 输入只接受一个原子，不接受任意分子几何
  - 自动构造单中心几何 `X 0 0 0`
  - 给出基于 Aufbau 规则的电子组态摘要
  - 输出估计的默认自旋
  - 输出单中心基组按角动量分组后的统计信息
  - `RHF` 路线现在显式按角动量块做分块对角化
  - 对单中心一电子积分做径向块压缩，并输出压缩比例与跨角动量块耦合大小

## 安装

```bash
cd /Users/roxy/ROXY_Projects/projects/atomic_hf
python3 -m pip install -r requirements.txt
```

## 运行

闭壳层原子 `He` 的 `RHF`：

```bash
python3 run_atomic_hf.py He --method rhf
```

开壳层原子 `O` 的 `UHF`：

```bash
python3 run_atomic_hf.py O --method uhf --spin 2
```

铁原子的 `UHF`：

```bash
python3 run_atomic_hf.py Fe --method uhf --spin 4 --basis def2-svp
```

显示迭代历史：

```bash
python3 run_atomic_hf.py Ne --method rhf --show-history
```

## 当前限制

- `RHF` 只支持闭壳层
- `UHF` 当前要求显式给出 `--spin`
- 电子组态摘要目前基于 Aufbau 规则的教学型实现，还没有处理所有重元素的真实例外组态
- `RHF` 已经开始利用角动量分块，但 `UHF` 还主要依赖通用开壳层求解器
- 二电子积分目前仍然调用通用 `libcint` 接口，还没有进入更深的单中心积分公式专门化简
- 还没有做“前 102 号元素任意一个原子都稳定快速计算”的工程化验证

## 测试

```bash
cd /Users/roxy/ROXY_Projects/projects/atomic_hf
pytest
```
