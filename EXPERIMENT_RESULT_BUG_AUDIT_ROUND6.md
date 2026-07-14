# 实验结果影响 Bug 复查报告（追加第6轮）

- 仓库：`https://gitee.com/qwe12345678/kda-csa-hca-fusion.git`
- 日期：2026-07-13
- 本轮范围：在上一轮基础上，再执行 1 轮 `pull → 检查 → 修复（如有）→ push → 再检查`。

## 第6轮流程

1. 已执行 pull：远程 master 已是最新。
2. 重新检查新增 LM 训练入口的可复现性、训练预算与全仓关键实验路径。
3. 发现 1 个会影响 LM 训练结果复现的问题，并已修复、提交、push。
4. 修复后再次 pull 和静态验证，未发现新的会直接影响实验结果的问题。

## 本轮发现并修复的问题

### `train_lm_autodl.py` 未固定随机种子

**文件**：`train_lm_autodl.py`

**问题**：脚本没有固定随机种子，导致同一命令下：

- 模型初始化不同；
- DataLoader shuffle 顺序不同；
- CUDA 随机流不同。

因此两次运行相同 `--max_steps` / `--seq_len` / `--batch_size` 可能得到不同 loss 曲线和 checkpoint，影响实验结果复现。

**修复**：

- 新增 `--seed` 参数，默认 `42`；
- 在模型构建前设置：
  - `random.seed(args.seed)`
  - `torch.manual_seed(args.seed)`
  - CUDA 时 `torch.cuda.manual_seed_all(args.seed)`
- 为 DataLoader shuffle 使用独立 `torch.Generator()`，并设置 `args.seed + 1_000_000`；
- 在 checkpoint 中保存 `seed` 字段；
- 在训练 profile 日志中打印 seed。

**提交**：`f2dbb83 fix: seed LM training reproducibly`

**Push**：已推送到 Gitee master。

## 修复后验证

已执行：

```bash
git pull <gitee-url> master
python -m py_compile *.py
bash -n run_autodl_lm.sh
```

并执行静态断言确认：

- CLI 存在 `--seed`；
- 存在 `random.seed(args.seed)`；
- 存在 `torch.manual_seed(args.seed)`；
- CUDA 路径存在 `torch.cuda.manual_seed_all(args.seed)`；
- DataLoader generator 使用 `args.seed + 1_000_000`；
- checkpoint 写入 `"seed": args.seed`。

验证结果：全部通过。

## 当前提交

```text
f2dbb83 fix: seed LM training reproducibly
f7182dd docs: record round 5 experiment bug audit
a6f878f fix: honor LM CLI training overrides
e83c4b7 docs: record round 4 experiment bug audit
2861d06 fix: choose safe LM autocast dtype
```

## 结论

追加第6轮发现并修复了一个会影响 LM 训练结果复现性的随机种子 bug。修复后重新 pull 与检查通过，未再发现新的会直接影响实验结果的问题。
