# 实验结果影响 Bug 复查报告（追加第5轮）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 日期：2026-07-13
- 本轮范围：在上一轮基础上，再执行 1 轮 `pull → 检查 → 修复（如有）→ push → 再检查`。

## 第5轮流程

1. 已执行 pull：远程 master 已是最新。
2. 重新检查新增 LM 训练入口、Kaggle/AutoDL 配置与全仓关键实验路径。
3. 发现 1 个会影响实验复现预算的问题，并已修复、提交、push。
4. 修复后再次 pull 和静态验证，未发现新的会直接影响实验结果的问题。

## 本轮发现并修复的问题

### `train_lm_autodl.py` 的 Kaggle profile 忽略 CLI 训练预算覆盖

**文件**：`train_lm_autodl.py`

**问题**：脚本原先 argparse 默认值为非 `None`：

```python
--max_steps default=2000
--seq_len default=1024
--batch_size default=2
```

但在 `--kaggle` 分支中固定写死：

```python
batch_size = 1
seq_len = 512
max_steps = 500
```

CLI 覆盖只在非 Kaggle 分支生效。因此用户若执行：

```bash
python train_lm_autodl.py --kaggle --max_steps 100 --seq_len 256
```

日志命令与实际训练预算不一致，可能导致复现实验时训练步数/上下文长度与用户预期不同。

**修复**：

- 将 `--max_steps` / `--seq_len` / `--batch_size` 默认值改为 `None`，表示使用环境 profile 默认值。
- 先选择 Kaggle 或 AutoDL/local profile 默认值，再统一应用 CLI overrides。
- 打印实际训练 profile：`batch_size`、`grad_accum`、`seq_len`、`optimizer_steps`。

**提交**：`a6f878f fix: honor LM CLI training overrides`

**Push**：已推送到 Gitee master。

## 修复后验证

已执行：

```bash
git pull <gitee-url> master
python -m py_compile *.py
bash -n run_autodl_lm.sh
```

并执行静态断言确认：

- argparse 默认值已为 `default=None`；
- 存在统一覆盖逻辑 `if args.max_steps is not None:`；
- 日志打印 `Training profile:`；
- shell 脚本语法通过。

验证结果：全部通过。

## 当前提交

```text
a6f878f fix: honor LM CLI training overrides
e83c4b7 docs: record round 4 experiment bug audit
2861d06 fix: choose safe LM autocast dtype
060c94e docs: record 3-round experiment bug audit
66f0ea9 fix: make LM training step accounting correct
```

## 结论

追加第5轮发现并修复了一个会影响 Kaggle/AutoDL LM 训练预算复现的 CLI override bug。修复后重新 pull 与检查通过，未再发现新的会直接影响实验结果的问题。
