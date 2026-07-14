# 实验结果影响 Bug 复查报告（追加20轮）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 本地 HEAD：`4c9bec9`
- 远程 master：`4c9bec9`
- 日期：2026-07-13
- 本轮范围：按用户要求，在第6轮之后继续执行 20 轮 `pull → 检查 → 如有问题则修复并 push`。

## 结论

本次追加 20 轮中，未发现新的会直接影响实验结果的 bug，因此没有代码修复提交。已保存本报告并 push。

## 20轮检查摘要

每轮均执行：

1. `git pull <gitee-url> master`
2. `python -m py_compile *.py`
3. `bash -n run_autodl_lm.sh`
4. 静态检查以下实验结果风险点：
   - 不存在旧的 `python train.py` / `evaluate.py` 入口；
   - 不存在旧 toy 脚本缺失模块导入：`config`、`model.hybrid_model`、`dataset`；
   - `train_lm_autodl.py` 的 next-token loss 对齐正确；
   - padding labels 置为 `-100`；
   - `max_steps` 按 optimizer steps 计数；
   - Kaggle/AutoDL CLI overrides 生效；
   - CUDA BF16/FP16 autocast 策略安全；
   - FP16 使用 `GradScaler` 且先 `unscale_` 再 clip；
   - LM 训练 seed 固定，DataLoader shuffle 使用独立 generator；
   - 生产实验 runner 不直接 `json.dump` 写结果；
   - Exp2/4/5/6 runner 有错误返回非零；
   - Bonferroni helper 使用 one-sided `target_p = 1.0 - alpha`，并保留 exact fallback 的 `_student_t_cdf` / `_betai` 路径。

| 轮次 | 结果 |
|---:|---|
| 1 | ✅ PASS |
| 2 | ✅ PASS |
| 3 | ✅ PASS |
| 4 | ✅ PASS |
| 5 | ✅ PASS |
| 6 | ✅ PASS |
| 7 | ✅ PASS |
| 8 | ✅ PASS |
| 9 | ✅ PASS |
| 10 | ✅ PASS |
| 11 | ✅ PASS |
| 12 | ✅ PASS |
| 13 | ✅ PASS |
| 14 | ✅ PASS |
| 15 | ✅ PASS |
| 16 | ✅ PASS |
| 17 | ✅ PASS |
| 18 | ✅ PASS |
| 19 | ✅ PASS |
| 20 | ✅ PASS |

## pytest 限制

当前沙箱仍无法完整运行 pytest，因为环境没有安装 `torch`，且 Python 为 3.13（项目要求 `<3.13`）。实际输出摘要：

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
1 error in 1.17s
```

建议在 Python 3.10–3.12 且安装依赖后运行：

```bash
pip install -e .[dev]
pytest -q -m "not slow"
python run_correctness.py
python test_figures.py
```

## 生成报告前工作区状态

```text
(clean)
```

## 最近提交

```text
4c9bec9 docs: record round 6 experiment bug audit
f2dbb83 fix: seed LM training reproducibly
f7182dd docs: record round 5 experiment bug audit
a6f878f fix: honor LM CLI training overrides
e83c4b7 docs: record round 4 experiment bug audit
2861d06 fix: choose safe LM autocast dtype
060c94e docs: record 3-round experiment bug audit
66f0ea9 fix: make LM training step accounting correct
```

## 最终说明

追加 20 轮均通过静态/轻量验证；未发现新的会直接影响实验结果的问题。本报告仅记录检查过程与结论。