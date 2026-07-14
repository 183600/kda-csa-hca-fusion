# 实验结果影响 Bug 复查与修复报告（最多3轮）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 本地 HEAD：`66f0ea9`
- 远程 master：`66f0ea9`
- 日期：2026-07-13

## 流程概述

按用户要求先 pull，然后逐文件/全仓检查可能影响实验结果的 bug；发现问题即修复并 push，再 pull/检查，最多 3 轮。本次共执行 3 轮：前两轮发现并修复问题，第三轮未发现新的会直接影响实验结果的问题。

## 第1轮：pull 后检查 → 发现并修复 → push

### 发现的问题

1. **LM 训练目标错位（影响 `train_lm_autodl.py` 真实 LM 训练结果）**
   - Dataset 已返回 `input_ids=tokens[:-1]`、`labels=tokens[1:]`。
   - 旧 loss 又使用 `logits[:, :-1]` 对 `labels[:, 1:]`，导致模型在位置 `t` 预测 `t+2`，跳过了真正的 next-token 目标。
   - 修复：loss 改为全位置对齐的 `cross_entropy(logits.reshape(...), labels.reshape(...), ignore_index=-100)`。

2. **padding token 被当作真实训练目标（影响 LM loss/perplexity）**
   - GPT-2 无原生 pad，脚本将 pad 设为 eos；如果不 mask padding，模型会大量学习 padding/eos。
   - 修复：按 padding 前真实长度把 padded target 位置置为 `-100`。

3. **AutoDL 脚本引用不存在入口（影响复现实验运行）**
   - `run_autodl_lm.sh` 调用不存在的 `train.py`、`evaluate.py`，且 `requirements.txt` 成功时不会安装 HF 训练依赖。
   - 修复：改为调用 `train_lm_autodl.py`，并显式安装 `transformers/datasets/accelerate/tqdm`。

4. **`train_toy_reference.py` 依赖仓库不存在的 `config/model/dataset` 模块**
   - 修复：改为兼容 wrapper，路由到正式的 `train_lm_autodl.py`。

### 推送

- commit：`5c1916e fix: repair LM training entrypoints`
- 已 push 到远程 master。

## 第2轮：pull 后检查 → 发现并修复 → push

### 发现的问题

1. **`max_steps` 语义与梯度累积不一致（影响 LM 训练预算与日志）**
   - 旧代码 `for step in range(max_steps)` 按 micro-batch 计数，但每 `grad_accum` 次才 optimizer step。
   - 用户看到的 `--max_steps 2000` 实际只有 `2000/grad_accum` 次参数更新，训练预算被低估；日志 loss 也记录的是除以 `grad_accum` 后的 loss。
   - 修复：`max_steps` 明确定义为 optimizer steps；内部循环执行 `grad_accum` 个 micro-batch；日志记录未缩放的平均 step loss；checkpoint 保存 `optimizer_step/micro_step`。
   - 同步修正文档中过于具体的“2h/3.6元”等估算，避免误导。

### 推送

- commit：`66f0ea9 fix: make LM training step accounting correct`
- 已 push 到远程 master。

## 第3轮：pull 后检查

第三轮 pull 后执行静态/轻量检查，未发现新的会直接影响实验结果的问题。

## 最终检查结果

| 检查项 | 结果 |
|---|---|
| `python -m py_compile *.py` | ✅ PASS |
| `bash -n run_autodl_lm.sh` | ✅ PASS |
| 旧入口/缺失模块引用扫描 | ✅ PASS（未发现旧的 `train.py`/`evaluate.py` 运行入口；仅剩正常的 `datasets` 导入和兼容 wrapper 说明文字） |
| Bonferroni fallback 与 scipy 对照 | PASS (fallback vs scipy <1e-9) |
| `git status --short`（生成本报告前） | `(clean)` |

相关扫描输出：

```text
./train_lm_autodl.py:30:    from datasets import load_dataset
./train_toy_reference.py:4:repository (``config``, ``model.hybrid_model``, ``dataset``) and therefore
```

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
1 error in 1.28s
```

建议在 Python 3.10–3.12 且安装依赖后运行：

```bash
pip install -e .[dev]
pytest -q -m "not slow"
python run_correctness.py
python test_figures.py
```

## 最近提交

```text
66f0ea9 fix: make LM training step accounting correct
5c1916e fix: repair LM training entrypoints
a2e917c verdict: keep rigorous ops, add real LM training for AutoDL <120 CNY
3f5d819 Fix softmax error-stub steps field (round 3)
acbbfc5 Fix HIGH-severity NaN gradient bug + tighten test tolerances (round 2)
127fc22 Fix bugs affecting experimental results (round 1)
adaf9cf Fix 4 bugs affecting experimental results (Round 1)
d02cebd fix experimental fairness and benchmark reliability
```

## 结论

最多 3 轮流程已完成：第 1、2 轮发现并修复了会影响 LM 训练实验结果/复现入口的问题并 push；第 3 轮未发现新的会直接影响实验结果的问题。当前远程 master 已包含修复。
