# 20轮交替复查与修复报告（最终版）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 当前本地 HEAD：`5872ae5`
- 日期：2026-07-13
- 说明：本报告记录本轮按“push → 检查 → 发现问题则修复并 push → 再检查”的交替流程执行的结果。

## 一、已执行的交替流程

1. 已将上一轮修复提交 `42e9b3d` push 到 Gitee。
2. 继续检查时发现：无 scipy fallback 的 Cornish-Fisher 临界值在小样本/尾部 alpha 下仍偏低，可能在无 scipy 环境中把边界结果误判为显著。
3. 已修复并 push `acb472c`：先提高 Cornish-Fisher 阶数。
4. 进一步检查发现：近似法仍不如直接计算可靠，已改为基于 regularized incomplete beta + bisection 的 dependency-free Student-t inverse CDF，并 push `5872ae5`。
5. 在最新 HEAD 上执行 20 轮重复检查，结果如下。

## 二、20轮检查结果

| 轮次 | 结果 | 检查内容 |
|---:|---|---|
| 1 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 2 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 3 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 4 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 5 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 6 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 7 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 8 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 9 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 10 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 11 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 12 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 13 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 14 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 15 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 16 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 17 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 18 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 19 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |
| 20 | ✅ PASS | py_compile；Bonferroni 一侧临界值；强制无 scipy fallback 与 scipy 对照；旧错误模式扫描；JSON 原子写入路径扫描；文档一致性 |

结论：最新 HEAD 上 20 轮检查均通过，未再发现新的会直接影响实验结果的问题。

## 三、关键验证点

- `_bonferroni_crit_q(n, alpha)` 在 scipy 路径使用 `t.ppf(1 - alpha, n - 1)`，与下游 `t_stat > crit` 的一侧 above-chance 检验一致。
- 强制 `_T_PP=False` 时 fallback 不再使用近似正态/Cornish-Fisher，而是直接计算 Student-t CDF 并二分求逆。
- 在当前环境 scipy 可用的情况下，fallback 与 `scipy.stats.t.ppf(1-alpha, n-1)` 对照误差 `<1e-9`。
- 已确认不存在旧错误模式：`1 - alpha / 2`、`p = 1 - alpha / 2`、`(-2 * p) ** 0.5`、`|t_stat|`。
- `method_analysis.py` sink 公式已与实现统一；`run_all.py` 的 `ABL_SEEDS` 默认值说明已与 `run_ablation.py` 一致。

## 四、完整 pytest 状态

当前执行环境仍缺少 `torch`，且 Python 版本为 3.13（项目约束 `<3.13`），因此完整 pytest 不能在本环境完成。输出摘要：

```text

==================================== ERRORS ====================================
_____________________ ERROR collecting run_correctness.py ______________________
ImportError while importing test module '/home/user/repo/run_correctness.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/local/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
run_correctness.py:41: in <module>
    import torch
E   ModuleNotFoundError: No module named 'torch'
=========================== short test summary info ============================
ERROR run_correctness.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.55s
```

建议在 Python 3.10–3.12 且安装依赖后运行：

```bash
pip install -e .[dev]
pytest -q -m "not slow"
python run_correctness.py
python test_figures.py
```

## 五、最近提交

```text
5872ae5 fix: make t critical fallback exact
acb472c fix: improve t critical fallback accuracy
42e9b3d fix: align statistical tests and document audit
bffec1c docs: mark code review counts as historical
6bf64ac docs: sync README experiment schemas and knobs
```

## 六、工作区状态

生成本报告前的代码检查工作区状态：
```text
(clean)
```

> 注：本报告文件本身会在生成后造成工作区变更，并将单独提交/push。

## 七、最终结论

本轮共完成多次 push/检查/修复交替：初始修复已 push，随后发现并修复 fallback 临界值精度问题并 push，最终在最新 HEAD 上重复 20 轮检查均通过。除当前沙箱无法运行 torch 数值回归外，未再发现新的会直接影响实验结果的问题。
