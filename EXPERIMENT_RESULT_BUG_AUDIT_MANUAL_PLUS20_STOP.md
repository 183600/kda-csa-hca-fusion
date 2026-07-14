# 实验结果影响 Bug 复查报告（手工读码 + shell，追加20轮请求）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 本地 HEAD：`b4fb217`
- 远程 master：`b4fb217`
- 日期：2026-07-13
- 用户要求：再来 20 轮；如果没有发现问题就停下；检查和修复交替进行；检查不止 shell，也要看代码。

## 执行结论

本次执行第 1 轮 `pull → 人工读码 → shell/静态验证` 后，未发现新的会直接影响实验结果的 bug。按照“如果没有发现问题就停下”的要求，没有继续空跑后续 19 轮。

## 人工读码检查内容

### 1. `train_lm_autodl.py`

人工复查了以下逻辑：

- 数据集构造：`input_ids=tokens[:-1]`，`labels=tokens[1:]`；
- loss 对齐：使用全位置 `logits.reshape(-1, vocab)` 对 `labels.reshape(-1)`，没有再次错位切片；
- padding mask：按真实长度将 padding target 置为 `-100`，避免 GPT-2 `pad_token=eos_token` 时把 padding 当作真实 eos 训练；
- 训练步数：`max_steps` 表示 optimizer steps，内部每步执行 `grad_accum` 个 micro-batch；
- profile override：Kaggle/AutoDL/local profile 先给默认值，再统一应用 `--max_steps` / `--seq_len` / `--batch_size`；
- 混合精度：支持 BF16 才用 BF16，否则 CUDA 用 FP16 + GradScaler，CPU 禁用 autocast；
- 可复现性：`--seed` 同时设置 Python、torch、CUDA seed，并为 DataLoader shuffle 使用独立 generator；
- checkpoint：保存 `optimizer_step`、`micro_step`、`seed`。

未发现新的会影响训练结果的 bug。

### 2. `run_autodl_lm.sh`

人工检查确认：

- 脚本调用正式入口 `train_lm_autodl.py`；
- 不再调用旧的不存在入口；
- 显式安装 LM 训练依赖；
- `set -euo pipefail` 与 shell 语法正常。

### 3. `run_quality.py` / `run_ablation.py`

人工检查确认：

- Bonferroni 检验与 “above chance” 研究问题保持 one-sided 口径；
- fallback 日志与实际 exact beta-CDF/bisection 数值路径一致；
- 显著性字段 `t_crit_bonferroni` / `significant_bonferroni` 的写入逻辑一致；
- 结果 JSON 写入仍走严格/原子写入 helper。

### 4. 主要实验 runner

人工/静态检查确认：

- `run_benchmark.py`、`run_quality.py`、`run_ablation.py`、`run_decoding.py`、`run_kv_cache.py`、`run_all.py` 不直接用 `json.dump` 写生产结果；
- Exp2/4/5/6 runner 在错误或不完整结果时仍能返回非零，避免 run_all 误报成功。

## Shell / 静态验证

已执行并通过：

```bash
git pull <gitee-url> master
python -m py_compile *.py
bash -n run_autodl_lm.sh
```

并执行静态断言确认：

- 不存在旧 toy 缺失模块导入；
- 不存在旧训练/评估可执行入口命令；
- LM 训练关键不变量存在；
- 生产实验 runner 没有直接 `json.dump`；
- Bonferroni helper 保持 one-sided exact fallback 路径。

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
1 error in 1.74s
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
b4fb217 docs: record statistical fallback audit
8b5d762 fix: label statistical fallback accurately
474e3b5 docs: record no-issue audit stop
2042776 docs: record additional 20-round audit
4c9bec9 docs: record round 6 experiment bug audit
f2dbb83 fix: seed LM training reproducibly
f7182dd docs: record round 5 experiment bug audit
a6f878f fix: honor LM CLI training overrides
e83c4b7 docs: record round 4 experiment bug audit
2861d06 fix: choose safe LM autocast dtype
```

## 最终结论

本次通过人工读码和 shell/静态验证完成第 1 轮检查，未发现新的会直接影响实验结果的问题，因此按要求停止。
